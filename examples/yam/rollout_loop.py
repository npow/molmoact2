"""Shared YAM rollout loop (prediction, IK, prefetch, execution) for every policy family.

Run directly it drives the left-arm conveyor GR00T checkpoint through a local Orchestral Worker;
``managed_rollout.py`` loads this file for all other families.
"""
import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
from omegaconf import OmegaConf
from orchestral import ConnectGrant, Session
import rerun as rr
import rerun.blueprint as rrb

from chunk_fusion import MODES as CHUNK_FUSION_MODES, ChunkPlan
from committed_range import gr00t_conditioning_prefix
from frame_capture_time import session_capture_time
from gripper_aperture import clip_predicted_aperture
from orchestral_session_bridge import redial_session_transport
from rollout_trace import COMMAND_STAGES, PREDICTION_OUTCOMES, PREDICTION_STAGES, TICK_STAGE, RolloutTrace
import run_recording

CAMERAS = {"belt": "front_camera_rgb", "wrist": "left_camera_rgb"}
CONTROL_HZ = 15  # training rate of the YAM checkpoints; one action per 15 Hz step
ACTION_SHAPE = (16, 7)


def frames_for_policy(obs, camera_shapes, camera_bindings=CAMERAS):
    frames = {}
    for model_camera, rig_camera in camera_bindings.items():
        image = np.asarray(obs[rig_camera])
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"Invalid RGB frame from {rig_camera}: {image.shape} {image.dtype}")
        height, width = camera_shapes[model_camera]
        frames[model_camera] = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return frames


def checked_chunk(session, cameras, state, instruction, *, episode_start, action_shape=ACTION_SHAPE,
                  model_label="GR00T", gripper_tolerance=None, pose_actions=None, matrix_policy=False,
                  observation_time=None, max_plan_rows=None, history_inputs=None, trace=None, prediction_id=0,
                  state_padding=None, committed_actions=None, noise_seed=None, timings=None, observation=None,
                  predict_timeout=None):
    """Query the worker and return the checked joint chunk and the seconds it took.

    ``timings``, when given, is filled with monotonic timestamps of each stage (see
    ``prediction_latency`` and ``command_latency``) so the control loop can measure each stage.
    """
    start = time.monotonic()
    query_state = pose_actions.state(state, matrix_policy=matrix_policy) if pose_actions else state
    if state_padding is not None:
        # A bimanual checkpoint driving one arm: the other arm's channels are the constant
        # the checkpoint was trained with, and only this arm's action columns are used.
        query_state = np.concatenate([query_state, state_padding]).astype(np.float32)
    sensors = {} if observation_time is None else {"timestamp": np.array([observation_time], dtype=np.float64)}
    if history_inputs is not None:
        sensors.update(history_inputs)
        query_state = None
    queued_joints = committed_actions
    if committed_actions is not None:
        # Committed rows are in the worker's own action space: the queued joint rows for a joint worker,
        # and the same commands as poses (forward kinematics) for a pose worker.
        if pose_actions:
            committed_actions = pose_actions.committed(queued_joints, matrix_policy=matrix_policy)
        if state_padding is not None:
            # The inactive half of the model has the same fixed command as its input state.
            parked = np.broadcast_to(state_padding, (len(committed_actions), len(state_padding)))
            committed_actions = np.concatenate([committed_actions, parked], axis=1)
        executed_actions = committed_actions
        if model_label == "GR00T":
            committed_actions = gr00t_conditioning_prefix(executed_actions)
        sensors["committed_actions"] = committed_actions
    if noise_seed is not None:
        sensors["noise_seed"] = noise_seed
    if redial_session_transport(session, timeout_s=predict_timeout):
        print(f"[{model_label.lower()}] Re-opened QUIC transport onto the active Worker.", flush=True)
    if predict_timeout is not None:
        transport = getattr(session, "_transport", None)
        if isinstance(getattr(transport, "_timeout_s", None), (int, float)) and transport._timeout_s > predict_timeout:
            transport._timeout_s = float(predict_timeout)
    requested = time.monotonic()
    response = session.predict(cameras=cameras, state=query_state, instruction=instruction,
                               episode_start=episode_start, **sensors, **run_recording.predict_link(observation))
    responded = time.monotonic()
    chunk = np.asarray(response.actions, dtype=np.float32)
    # The chunk id is the rollout's prediction number; link it to the SDK's Prediction record.
    run_recording.current().chunk(prediction_id, outcome="returned", rows=len(chunk),
                                  prediction=getattr(session, "last_recorded_prediction", None))
    if trace is not None:
        trace.prediction(prediction_id, response, chunk, session, responded - start)
    if chunk.shape != action_shape or not np.isfinite(chunk).all():
        raise ValueError(f"{model_label} returned an invalid action chunk")
    if committed_actions is not None and not np.array_equal(chunk[:len(committed_actions)], committed_actions):
        raise ValueError(f"{model_label} did not preserve the committed action prefix")
    if committed_actions is not None:
        chunk[:len(committed_actions)] = executed_actions
    chunk = chunk[:max_plan_rows]
    if state_padding is not None:
        chunk = chunk[:, :len(state)]
    ik_seconds = 0.0
    if pose_actions:
        from tool_pose_actions import UnreachableToolPose

        # The queued joints already select an IK branch. Solve only the new rows,
        # starting at the last queued target so the step bound holds across the handoff.
        kept = (queued_joints[:len(chunk)] if queued_joints is not None
                else np.empty((0, 7), dtype=np.float32))
        suffix = chunk[len(kept):]
        chunk = kept
        if len(suffix):
            ik_started = time.monotonic()
            try:
                solved = pose_actions.actions(suffix, kept[-1] if len(kept) else state, matrix_policy=matrix_policy)
                chunk = np.concatenate((kept, solved))
                if len(solved) < len(suffix):
                    print(f"[{model_label.lower()}] Tool IK kept {len(chunk)} of {len(kept) + len(suffix)} rows.",
                          flush=True)
            except UnreachableToolPose as error:
                print(f"[{model_label.lower()}] {error}. Replanning after the queued targets.", flush=True)
            finally:
                ik_seconds = time.monotonic() - ik_started
    chunk = chunk.copy()
    chunk[:, 6] = clip_predicted_aperture(chunk[:, 6], model_label, tolerance=gripper_tolerance)
    ready = time.monotonic()
    if timings is not None:
        timings.update(started=start, requested=requested, responded=responded, ready=ready)
        if pose_actions:
            timings["ik_seconds"] = ik_seconds
    return chunk, ready - start


def _ms(seconds):
    return 1000.0 * seconds


def prediction_latency(timings, submitted_at):
    """``PREDICTION_STAGES`` from ``checked_chunk`` timestamps and the time the request was submitted."""
    ik_seconds = timings.get("ik_seconds")
    stages = {
        "queue_ms": _ms(timings["started"] - submitted_at),
        "prepare_ms": _ms(timings["requested"] - timings["started"]),
        "predict_ms": _ms(timings["responded"] - timings["requested"]),
        "postprocess_ms": _ms(timings["ready"] - timings["responded"] - (ik_seconds or 0.0)),
    }
    if ik_seconds is not None:
        stages["ik_ms"] = _ms(ik_seconds)
    return stages


def command_latency(timings, observed_at, accepted_at, commanded_at):
    """``COMMAND_STAGES`` once the first row of an accepted prediction's plan is sent."""
    return {
        "ready_to_accept_ms": _ms(accepted_at - timings["ready"]),
        "accept_to_command_ms": _ms(commanded_at - accepted_at),
        "response_to_command_ms": _ms(commanded_at - timings["responded"]),
        "observation_to_command_ms": _ms(commanded_at - observed_at),
    }


@dataclass
class AcceptedPrediction:
    """A prediction whose plan the control loop took, until its first row is sent."""

    prediction_id: int
    timings: dict
    stages: dict
    observed_at: float
    accepted_at: float


class LatencySpans:
    """Send rollout latency to Grafana as spans, next to the Orchestral SDK's own.

    - `yam.rollout.ik`: every prediction that ran IK (pose workers), whatever became of its chunk;
      the duration is the IK solve.
    - `yam.rollout.response_to_command`: every accepted prediction; the duration is the worker
      response to the first new row sent.

    Each span carries its prediction's stages as `yam.rollout.<stage>` attributes, its
    `yam.rollout.outcome` (one of ``PREDICTION_OUTCOMES``), the family, `yam.rollout.shadow` and
    `yam.rollout.chunk_fusion`, and the Session's own `orc.*` attributes. The SDK relays them with its
    own spans, so they are sent only when it has telemetry credentials (`ORC_TELEMETRY_BASE_URL` and
    `ORC_TELEMETRY_API_KEY` for a grant-only Session). Recording never stops the control loop.
    """

    ik_name = "yam.rollout.ik"
    command_name = "yam.rollout.response_to_command"

    def __init__(self, session, family, *, shadow, chunk_fusion):
        try:
            from orchestral.core import tracing
        except ImportError:
            tracing = None
        self._tracing = tracing if callable(getattr(tracing, "record_span", None)) else None
        if self._tracing is None:
            print("[telemetry] This Orchestral SDK cannot record spans; rollout latency "
                  "stays in the Rerun trace only.", flush=True)
        attributes = getattr(session, "_trace_attributes", None)
        self._attributes = attributes if callable(attributes) else dict
        self._labels = {"yam.rollout.family": family, "yam.rollout.shadow": bool(shadow),
                        "yam.rollout.chunk_fusion": chunk_fusion}
        self._failed = False

    def prediction(self, prediction_id, outcome, stages):
        if "ik_ms" in stages:
            self._record(self.ik_name, stages["ik_ms"], prediction_id, outcome, stages)

    def command(self, prediction_id, stages):
        self._record(self.command_name, stages["response_to_command_ms"], prediction_id, "accepted", stages)

    def _record(self, name, duration_ms, prediction_id, outcome, stages):
        if self._tracing is None or self._failed:
            return
        try:
            end_ns = time.time_ns()
            attributes = {**self._attributes(), **self._labels, "yam.rollout.prediction_id": prediction_id,
                          "yam.rollout.outcome": outcome,
                          **{f"yam.rollout.{stage}": round(value, 3) for stage, value in stages.items()}}
            self._tracing.record_span(name, end_ns - round(duration_ms * 1e6), end_ns, attributes=attributes)
        except Exception as error:  # telemetry must not interrupt the robot
            self._failed = True
            print(f"[telemetry] Stopped recording latency spans: {error}", flush=True)


class SdkLoopMetrics:
    """Feed the SDK's `ControlLoopTimings` from this loop, which keeps its own chunk plan.

    `ActionQueue` (used by `rtc_rollout.py`) feeds `ControlLoopTimings` itself. This loop does not use the
    queue, so without this the Grafana panels built on it (`orchestral_ingest_loop_*`: observation to
    action, reaction, action age, held ticks, tick jitter) stay empty for every non-RTC family. The calls
    mirror the queue's: `delivered` when a chunk is accepted, `tick` once per control tick (a tick with
    no row is a held tick), `finish` at the end. It uses the SDK's underscore methods, so any failure
    turns it off with one warning: telemetry must never interrupt the robot.
    """

    def __init__(self, session, started_at, period):
        self._timings = None
        self._failed = False
        clock = getattr(session, "clock", None)
        try:
            from orchestral.client.session_queues import ControlLoopTimings
            self._clock = clock
            self._period = period
            self._t0 = clock.from_source(started_at)
            self._timings = ControlLoopTimings(getattr(session, "_trace_attributes", None))
        except Exception as error:  # an older SDK, or a session without a clock
            self._failed = True
            print(f"[telemetry] This Orchestral SDK cannot record loop metrics: {error}", flush=True)
        self._chunks = {}

    def delivered(self, chunk_id, observed_at, sent_at, captured_at=None):
        """A chunk was accepted into the plan. `observed_at` and `sent_at` are `time.monotonic()` readings;
        `captured_at` is the frames' capture time on the Session clock (None: measure from `observed_at`)."""
        def call():
            chunk = SimpleNamespace(seq=chunk_id, t_observed=self._clock.from_source(observed_at),
                                    t_captured=captured_at, t_sent=self._clock.from_source(sent_at))
            self._chunks[chunk_id] = chunk
            self._timings._delivered(chunk, self._clock.now())
        self._guard(call)

    def tick(self, tick, chunk_id):
        """One control tick; `chunk_id` is the chunk the row came from, or None when no row ran."""
        def call():
            now = self._clock.now()
            grid = max(tick, math.floor((now - self._t0) / self._period))
            chunk = None if chunk_id is None else self._chunks.get(chunk_id)
            self._timings._tick(grid=grid, now=now, due=self._t0 + tick * self._period, chunk=chunk)
            for stale in [seq for seq in self._chunks if chunk_id is not None and seq < chunk_id - 2]:
                del self._chunks[stale]
        self._guard(call)

    def finish(self):
        self._guard(lambda: self._timings._finish())

    def _guard(self, call):
        if self._failed:
            return
        try:
            call()
        except Exception as error:
            self._failed = True
            print(f"[telemetry] Stopped recording loop metrics: {error}", flush=True)


def latency_summary(prediction_samples, command_samples, tick_ms, outcomes):
    """Lines of p50 / p95 / max per stage and the prediction outcomes, for the end-of-run printout."""
    def row(name, values):
        return (f"  {name:<27} {np.percentile(values, 50):8.1f} {np.percentile(values, 95):8.1f} "
                f"{np.max(values):8.1f}")

    lines = [f"Latency in ms over {len(prediction_samples)} predictions, "
             f"{len(command_samples)} commanded:      p50      p95      max"]
    for samples, stages in ((prediction_samples, PREDICTION_STAGES), (command_samples, COMMAND_STAGES)):
        for stage in stages:
            values = [sample[stage] for sample in samples if stage in sample]
            if values:
                lines.append(row(stage, values))
    if tick_ms:
        lines.append(row(TICK_STAGE, tick_ms))
    lines.append("  outcomes: " + ", ".join(f"{name} {outcomes[name]}" for name in PREDICTION_OUTCOMES
                                            if outcomes[name]))
    return lines


def observe(session, frames, joints, instruction, *, pose_actions=None, matrix_policy=False, predict_timeout=None):
    """Stream one observation; a pose worker's forward kinematics runs here, off the control tick."""
    redial_session_transport(session, timeout_s=predict_timeout)
    if predict_timeout is not None:
        transport = getattr(session, "_transport", None)
        if isinstance(getattr(transport, "_timeout_s", None), (int, float)) and transport._timeout_s > predict_timeout:
            transport._timeout_s = float(predict_timeout)
    state = pose_actions.state(joints, matrix_policy=matrix_policy) if pose_actions else joints
    session.observe(cameras=frames, state=state, instruction=instruction)


def warm_session(session, instruction):
    with np.load(Path(__file__).with_name("warmup-inputs.npz")) as sample:
        cameras = {name: sample[name] for name in CAMERAS}
        state = sample["state"]
    for i in range(3):
        _, duration = checked_chunk(session, cameras, state, instruction, episode_start=(i == 0))
        print(f"[gr00t] Warmup {i + 1}/3 finished in {duration * 1000:.0f} ms.", flush=True)


def latency_line(label, call_ms, stage_rows, final=False):
    """One line of prediction latency: the whole call (predict + IK) and the worker-reported stages."""
    ms = np.asarray(call_ms)
    p50, p95, p99 = np.percentile(ms, [50, 95, 99])
    line = (f"[{label.lower()}] {'Final' if final else 'Running'} prediction latency over {len(ms)} calls "
            f"(predict + IK): p50 {p50:.0f} ms, p95 {p95:.0f} ms, p99 {p99:.0f} ms, max {ms.max():.0f} ms")
    if stage_rows:
        def p(key, q):
            values = [row[key] for row in stage_rows if row.get(key) is not None]
            return float(np.percentile(values, q)) if values else float("nan")
        line += ("; worker p50/p95: model {:.0f}/{:.0f} ms, network {:.0f}/{:.0f} ms, encode {:.0f}/{:.0f} ms"
                 .format(p("model_ms", 50), p("model_ms", 95), p("network_ms", 50), p("network_ms", 95),
                         p("encode_ms", 50), p("encode_ms", 95)))
    return line


def run(env, session, mask, *, seconds, instruction, shadow, lead_steps=8,
        camera_bindings=CAMERAS, history_offsets=(-3, 0), control_hz=CONTROL_HZ,
        action_shape=ACTION_SHAPE, model_label="GR00T", gripper_tolerance=None, pose_actions=None, matrix_policy=False,
        timestamp_observations=False, max_plan_rows=None, temporal_history=None, trace=None,
        state_padding=None, chunk_fusion="splice", fusion_rows=4, committed_actions=False,
        stream_observations=False, noise_seed=None, max_observed_rows=None, hold_gripper_rows=0,
        predict_timeout=None, max_prediction_failures=0):
    if committed_actions and (chunk_fusion != "splice" or lead_steps < 1):
        raise ValueError("Committed actions require splice and at least one prefetch lead step")
    period = 1 / control_hz
    history_size = 1 - min(history_offsets, default=0)
    history = deque(maxlen=history_size)
    pending = ChunkPlan(chunk_fusion, blend_rows=fusion_rows, hold_gripper_rows=hold_gripper_rows)
    request_slot = 0
    future = None
    active_prediction = 0
    submitted_with = 0
    submitted_at = 0.0
    request_hold_ticks = 0
    request_prefix_changed = False
    observations = deque()
    request_timings = {}
    request_observed_at = request_submitted_at = request_captured_at = None
    awaiting_command = None  # an AcceptedPrediction until its first row is sent
    prediction_samples, command_samples, tick_latencies_ms = [], [], []
    outcomes = Counter()
    latency_spans = LatencySpans(session, model_label, shadow=shadow, chunk_fusion=chunk_fusion)
    rejected_chunks = 0
    failed_predictions = 0
    recovered_failures = 0
    hold_warned_at = 0.0
    rows = predictions = handoffs = misses = holds = 0
    recorder = run_recording.current()
    tick = -1
    call_ms, stage_rows = [], []
    observed_since_prediction = 0
    skipped_observations = 0
    next_tick = time.monotonic()
    started_at = next_tick
    deadline = next_tick + seconds
    loop_metrics = SdkLoopMetrics(session, started_at, period)
    policy_camera_names = tuple(rig.removesuffix("_rgb") for rig in camera_bindings.values())
    warned_no_capture = False
    predictor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{model_label.lower()}-predict")
    try:
        while (tick_started := time.monotonic()) < deadline:
            while observations and observations[0].done():
                try:
                    observations.popleft().result()
                except Exception as error:
                    failed_predictions += 1
                    if failed_predictions > max_prediction_failures:
                        raise
                    recovered_failures += 1
                    print(f"[{model_label.lower()}] Streamed observation failed "
                          f"({failed_predictions}/{max_prediction_failures} allowed): "
                          f"{type(error).__name__}: {error}. Continuing.", flush=True)
            tick += 1
            with recorder.span("get_obs"):
                obs = env.get_obs()
            observed_at = time.monotonic()
            session_clock = getattr(session, "clock", None)
            captured_at = (session_capture_time(env, session_clock, camera_names=policy_camera_names)
                           if session_clock is not None else None)
            if captured_at is None and not warned_no_capture:
                warned_no_capture = True
                print("[telemetry] The env reported no capture time for the policy's cameras "
                      f"{policy_camera_names}: loop latencies are measured from the observation, "
                      f"not from capture. env timestamps: {getattr(env, 'last_frame_timestamps', None)}", flush=True)
            observation_time = observed_at - started_at if timestamp_observations else None
            frames = frames_for_policy(obs, session.grant.io_spec.cameras, camera_bindings)
            history.append(frames)
            measured = mask.validate_state(obs["joint_positions"], name="policy input state")
            if temporal_history is not None:
                temporal_history.observe(frames, measured[:7])
            # A fusing plan takes each chunk as it arrives and blends it into the queued rows.
            # A splicing plan waits until the queued rows have run out.
            if future is not None and future.done() and (pending.fuses or not pending):
                try:
                    chunk, elapsed = future.result()
                except Exception as error:
                    future = None
                    failed_predictions += 1
                    if trace is not None:
                        trace.event("Prediction %d failed: %s", predictions, error)
                    if failed_predictions > max_prediction_failures:
                        print(f"[{model_label.lower()}] Prediction {predictions} failed: "
                              f"{type(error).__name__}: {error}", flush=True)
                        raise
                    recovered_failures += 1
                    print(f"[{model_label.lower()}] Prediction {predictions} failed "
                          f"({failed_predictions}/{max_prediction_failures} allowed): "
                          f"{type(error).__name__}: {error}. Replanning.", flush=True)
                    if type(error).__name__ == "EpisodeResetRequired" and callable(getattr(session, "new_episode", None)):
                        session.new_episode()
                else:
                    future = None
                    failed_predictions = 0
                    call_ms.append(elapsed * 1000)
                    try:
                        stage_rows.append(session.step_timings[-1].stages_ms())
                    except (AttributeError, IndexError, TypeError):
                        pass
                    if len(call_ms) % 20 == 0:
                        print(latency_line(model_label, call_ms, stage_rows), flush=True)
                    # Each command sent since the request corresponds to one row of the new
                    # plan, which starts at the observation taken for this prediction.
                    # A missed deadline invalidates a prefetched plan. Wait for a fresh
                    # observation instead of applying a plan computed before the hold.
                    missed = bool(submitted_with and request_hold_ticks)
                    first_row = None if missed or request_prefix_changed else pending.add(chunk, request_slot)
                    outcome = ("unreachable" if not len(chunk) else "missed_boundary" if missed
                               else "prefix_changed" if request_prefix_changed
                               else "stale" if first_row is None else "accepted")
                    # Recorded for every chunk: IK is most likely to matter in the ones that came too late.
                    prediction_stages = prediction_latency(request_timings, request_submitted_at)
                    prediction_samples.append(prediction_stages)
                    outcomes[outcome] += 1
                    latency_spans.prediction(predictions, outcome, prediction_stages)
                    recorder.chunk(predictions, outcome="accepted" if outcome == "accepted" else "dropped",
                                   reason=None if outcome == "accepted" else outcome, first_row=first_row,
                                   hold_ticks_while_waiting=request_hold_ticks, queued_rows_at_request=submitted_with,
                                   fusion=chunk_fusion)
                    if trace is not None:
                        trace.latency(prediction_stages)
                        if outcome == "unreachable":
                            trace.event("Prediction %d rejected: unreachable plan", predictions)
                        elif outcome == "missed_boundary":
                            trace.event("Prediction %d dropped after %d hold ticks", predictions, request_hold_ticks)
                        elif outcome == "prefix_changed":
                            trace.event("Prediction %d dropped after the safety mask changed a committed target", predictions)
                        elif outcome == "stale":
                            trace.event("Prediction %d dropped: every row was stale", predictions)
                        else:
                            trace.event("Prediction %d accepted starting at row %d", predictions, first_row)
                    if len(chunk):
                        rejected_chunks = 0
                    else:
                        rejected_chunks += 1
                        if rejected_chunks == 3:
                            raise SystemExit(f"{model_label} returned three consecutive unreachable plans; rollout stopped")
                    if missed:
                        misses += 1
                        # elapsed is predict + IK only; since-request also counts time queued behind observe() calls.
                        print(f"[{model_label.lower()}] Prefetch missed the boundary ({elapsed * 1000:.0f} ms; "
                              f"{(time.monotonic() - submitted_at) * 1000:.0f} ms since request, requested with "
                              f"{submitted_with} queued rows, {request_hold_ticks} hold ticks, prediction {predictions}, "
                              f"{time.monotonic() - started_at:.1f} s into the run). Replanning.", flush=True)
                    elif outcome == "prefix_changed":
                        print(f"[{model_label.lower()}] Prefetch dropped after the safety mask clipped a committed "
                              f"target (prediction {predictions}, {time.monotonic() - started_at:.1f} s into the run). "
                              "Replanning.", flush=True)
                    elif first_row is not None:
                        active_prediction = predictions
                        loop_metrics.delivered(predictions, request_observed_at, request_submitted_at, request_captured_at)
                        handoffs += bool(submitted_with)
                        awaiting_command = AcceptedPrediction(predictions, request_timings, prediction_stages,
                                                              request_observed_at, time.monotonic())
            short_truncated_plan = (
                bool(pending) and len(pending) < lead_steps and bool(call_ms)
                and len(pending) * period < min(call_ms[-3:]) / 1000.0
            )
            if len(history) == history_size and future is None and len(pending) <= lead_steps and not short_truncated_plan:
                cameras = (
                    {name: np.stack([history[offset - 1][name] for offset in history_offsets])
                     for name in camera_bindings}
                    if history_offsets else history[-1]
                )
                history_inputs = None
                if temporal_history is not None:
                    cameras, history_inputs = temporal_history.inputs()
                submitted_with = len(pending)
                submitted_at = time.monotonic()
                request_slot = pending.slot
                request_hold_ticks = 0
                hold_warned_at = 0.0
                request_prefix_changed = False
                prefix = pending.committed_actions() if committed_actions and submitted_with else None
                handle = recorder.observation(prediction=predictions + 1, queued_rows=submitted_with,
                                              committed_rows=0 if prefix is None else len(prefix),
                                              lead_steps=lead_steps)
                request_timings = {}
                request_observed_at = observed_at
                request_captured_at = captured_at
                if trace is not None:
                    trace.event("Prediction %d requested with %d queued rows", predictions + 1, submitted_with)
                request_submitted_at = time.monotonic()
                future = predictor.submit(checked_chunk, session, cameras, measured[:7].copy(),
                                          instruction, episode_start=(predictions == 0),
                                          action_shape=action_shape, model_label=model_label,
                                          gripper_tolerance=gripper_tolerance, pose_actions=pose_actions,
                                          matrix_policy=matrix_policy, observation_time=observation_time,
                                          max_plan_rows=max_plan_rows, history_inputs=history_inputs,
                                          trace=trace, prediction_id=predictions + 1,
                                          state_padding=state_padding, committed_actions=prefix, noise_seed=noise_seed,
                                          timings=request_timings,
                                          observation=handle,
                                          predict_timeout=predict_timeout)
                predictions += 1
                observed_since_prediction = 0
            elif stream_observations:
                # Use the prediction executor to preserve Session order without blocking the control tick.
                # It also runs a pose worker's forward kinematics, so the command below does not wait for it;
                # the single worker thread never uses pose_actions from two threads at once.
                # `max_observed_rows` bounds the frames sent between two predictions: see managed_rollout.
                # At most one observe in flight: each is a full round trip (~60-70 ms to a remote Worker), about one
                # 15 Hz tick, on the same thread as predict. Queuing one per tick backed up to 19 of them and the next
                # predict waited ~1.3 s behind them (queue_ms), so every prefetch missed its boundary. Skip the frame.
                if observations:
                    skipped_observations += 1
                elif max_observed_rows is None or observed_since_prediction < max_observed_rows:
                    observed_since_prediction += 1
                    observations.append(predictor.submit(
                        observe, session, frames, measured[:7].copy(), instruction,
                        pose_actions=pose_actions, matrix_policy=matrix_policy,
                        predict_timeout=predict_timeout,
                    ))

            next_tick += period
            row = command = row_index = tick_ms = None
            if pending and time.monotonic() < deadline:
                row_index, row = pending.pop()
                if not shadow:
                    measured = mask.validate_state(env.get_robot_state()["joint_positions"], name="measured state")
                    target = measured.copy()
                    target[:7] = row
                    command = mask.command_target(target, measured)
                    env.step_command_only(command, wait=False)
                    recorder.commanded(tick, command, chunk=active_prediction, row=row_index,
                                       clipped=not np.allclose(command[:7], row, rtol=0, atol=1e-6))
                    if committed_actions and future is not None and submitted_with:
                        # The hardware limit takes precedence. Replan from fresh feedback if a conditioned
                        # target had to change, instead of applying a suffix based on a different command.
                        request_prefix_changed |= not np.allclose(command[:7], row, rtol=0, atol=1e-6)
                    if temporal_history is not None:
                        temporal_history.commanded(command[:7])
                rows += 1
                commanded_at = time.monotonic()
                next_tick = max(next_tick, commanded_at)
                tick_ms = 1000.0 * (commanded_at - tick_started)
                tick_latencies_ms.append(tick_ms)
                if awaiting_command is not None:
                    accepted, awaiting_command = awaiting_command, None
                    stages = command_latency(accepted.timings, accepted.observed_at, accepted.accepted_at,
                                             commanded_at)
                    command_samples.append(stages)
                    latency_spans.command(accepted.prediction_id, {**accepted.stages, **stages})
                    if trace is not None:
                        trace.latency(stages, event=f"Prediction {accepted.prediction_id} first row commanded "
                                                    f"{stages['response_to_command_ms']:.0f} ms after the response")
            elif future is not None:
                request_hold_ticks += 1
                holds += 1
                hold_seconds = request_hold_ticks * period
                if hold_seconds >= 1.0 and tick_started - hold_warned_at >= (5.0 if hold_warned_at else 0.0):
                    hold_warned_at = tick_started
                    print(f"[{model_label.lower()}] Holding while waiting on prediction {predictions} "
                          f"({hold_seconds:.1f} s, {request_hold_ticks} hold ticks)...", flush=True)
            loop_metrics.tick(tick, active_prediction if row is not None else None)
            recorder.tick(tick, chunk=active_prediction if row is not None else None, row=row_index,
                          held=row is None, reason=None if row is not None else "waiting" if future is not None
                          else "idle", tick_elapsed_ms=1000 * (time.monotonic() - tick_started))
            if trace is not None:
                trace.tick(measured, row, command, active_prediction, row_index,
                           holding=row is None and future is not None, tick_ms=tick_ms)
            now = time.monotonic()
            if now > next_tick:
                next_tick = now
            time.sleep(max(0, next_tick - now))
    finally:
        predictor.shutdown(wait=False, cancel_futures=True)
        loop_metrics.finish()
        # Printed on Ctrl+C too: that is how an operator ends most rollouts.
        if call_ms:
            print(latency_line(model_label, call_ms, stage_rows, final=True), flush=True)
        if skipped_observations:
            print(f"[{model_label.lower()}] Skipped {skipped_observations} streamed frames while an observe was in flight.",
                  flush=True)
        if prediction_samples:
            print("\n".join(latency_summary(prediction_samples, command_samples, tick_latencies_ms, outcomes)),
                  flush=True)
    print(f"{model_label} rollout finished. {predictions} predictions, {rows} {'shadow rows' if shadow else 'commands'}, "
          f"{handoffs} prefetched handoffs, {misses} missed boundaries, {holds} hold ticks"
          + (f", {recovered_failures} recovered failures." if recovered_failures else "."))


def default_rrd_path(config_path):
    config = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    base_dir = Path((config.get("storage") or {}).get("base_dir", "./yam_eval_runs"))
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    return (base_dir / "rerun" / f"gr00t_{stamp}.rrd").expanduser().resolve()


def record_camera_frame(recording, camera_paths, frame_counts, count_lock, start_time,
                        rig_camera, frame, captured_at):
    model_camera = camera_paths.get(rig_camera)
    if model_camera is None:
        return
    image = np.asarray(frame)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3:
        raise ValueError(f"Invalid RGB frame from {rig_camera}: {image.shape} {image.dtype}")
    recording.set_time("elapsed", duration=max(0.0, captured_at - start_time))
    recording.log(
        f"/cameras/{model_camera}",
        rr.Image(np.ascontiguousarray(image)).compress(jpeg_quality=85),
    )
    with count_lock:
        frame_counts[model_camera] += 1


def record_blueprint(camera_bindings, *, control_trace=False):
    views = [
        rrb.Spatial2DView(origin=f"/cameras/{name}", name=f"{name} camera")
        for name in camera_bindings
    ]
    cameras = rrb.Horizontal(*views, column_shares=[1] * len(views))
    if not control_trace:
        return rrb.Blueprint(cameras)
    telemetry = rrb.Tabs(
        rrb.TimeSeriesView(origin="/robot", name="Measured and commanded joints"),
        rrb.TimeSeriesView(origin="/policy/latency", name="Inference latency (ms)"),
        rrb.TimeSeriesView(origin="/policy/upload", name="Request size (bytes)"),
        rrb.TimeSeriesView(origin="/policy/holding", name="Hold ticks"),
        rrb.TensorView(origin="/policy/raw_actions", name="Raw action chunk"),
        rrb.TextLogView(origin="/policy/events", name="Chunk events"),
    )
    return rrb.Blueprint(rrb.Vertical(cameras, telemetry, row_shares=[3, 2]))


def main():
    from launch_yaml_eval_molmoact import _build_env, resolve_bimanual_execution_mask, validate_bimanual_model_arm_order

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestral-grant", type=Path, required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--right-config-path", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--instruction", default="Pick up the red lid from the conveyor belt.")
    parser.add_argument("--prefetch-lead-steps", type=int, default=8)
    parser.add_argument("--chunk-fusion", choices=CHUNK_FUSION_MODES, default="splice",
                        help="How successive action chunks are joined (see chunk_fusion.py)")
    parser.add_argument("--fusion-rows", type=int, default=4,
                        help="Ticks over which crossfade and rail move onto a new chunk")
    parser.add_argument("--rrd", type=Path, help="Rerun recording path (default: yam_eval_runs/rerun/gr00t_<timestamp>.rrd)")
    parser.add_argument("--shadow", action="store_true", help="Read cameras and feedback without sending policy commands.")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if not 0 <= args.prefetch_lead_steps < ACTION_SHAPE[0]:
        parser.error("--prefetch-lead-steps must be between 0 and 15")
    if args.fusion_rows < 1:
        parser.error("--fusion-rows must be at least 1")
    if not args.orchestral_grant.is_file():
        parser.error(f"local GR00T Worker Grant is missing at {args.orchestral_grant}")
    if args.orchestral_grant.stat().st_mode & 0o077:
        parser.error("the Grant file must have mode 600")
    grant = ConnectGrant.from_token(args.orchestral_grant.read_text().strip())
    spec = grant.io_spec
    if (spec.state_dim != 7 or spec.action_shape != ACTION_SHAPE or set(spec.cameras) != set(CAMERAS)
            or spec.camera_history != dict.fromkeys(CAMERAS, (-3, 0))):
        parser.error("the Grant does not match the conveyor GR00T checkpoint")
    left_cfg = OmegaConf.to_container(OmegaConf.load(args.config_path), resolve=True)
    right_cfg = OmegaConf.to_container(OmegaConf.load(args.right_config_path), resolve=True)
    validate_bimanual_model_arm_order(left_cfg, right_cfg)
    limits = ((left_cfg.get("eval") or {}).get("bimanual") or {}).get("both_arm_max_delta")
    if limits is None:
        limits = ([0.05] * 6 + [0.025]) * 2
    mask = resolve_bimanual_execution_mask(bimanual=True, active_arm_side="left",
                                          execution_mode="shadow" if args.shadow else "active_arm_hold",
                                          both_arm_max_delta=limits)
    print(f"[gr00t] Opening local Worker {grant.deployment_id}.", flush=True)
    session = Session(grant, encoder="h264")
    try:
        warm_session(session, args.instruction)
        input(f"GR00T left-arm {'shadow' if args.shadow else 'rollout'} at {CONTROL_HZ} Hz. "
              "Press Enter to start (Ctrl-C stops)... ")
        output_path = (args.rrd.expanduser().resolve() if args.rrd else default_rrd_path(args.config_path))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"[rerun] Recording every captured camera frame to {output_path}", flush=True)
        recording = rr.RecordingStream("yam_gr00t_rollout")
        sensor_roles = {
            "top": "front_camera_rgb",
            "left": "left_camera_rgb",
            "right": "right_camera_rgb",
        }
        camera_paths = {
            sensor_role: model_camera
            for model_camera, rig_camera in CAMERAS.items()
            for sensor_role, configured_rig_camera in sensor_roles.items()
            if rig_camera == configured_rig_camera
        }
        frame_counts = {name: 0 for name in CAMERAS}
        count_lock = threading.Lock()
        recording_start = time.monotonic()
        try:
            with recording:
                recording.save(output_path, default_blueprint=record_blueprint(CAMERAS, control_trace=True))
                recording.log(
                    "/task/instruction",
                    rr.TextDocument(f"# Task instruction\n\n{args.instruction}", media_type="text/markdown"),
                    static=True,
                )
                env, *_ = _build_env(
                    SimpleNamespace(config_path=args.config_path, right_config_path=args.right_config_path,
                                    control_hz=CONTROL_HZ, max_steps=None),
                    camera_sources={camera: {"capture_rate_hz": CONTROL_HZ} for camera in ("top", "left")},
                    camera_frame_callback=lambda camera, frame, timestamp: record_camera_frame(
                        recording, camera_paths, frame_counts, count_lock, recording_start,
                        camera, frame, timestamp,
                    ),
                )
                trace = None
                try:
                    trace = RolloutTrace(recording, recording_start, session)
                    run(env, session, mask, seconds=args.seconds, instruction=args.instruction, shadow=args.shadow,
                        lead_steps=args.prefetch_lead_steps,
                        chunk_fusion=args.chunk_fusion, fusion_rows=args.fusion_rows,
                        trace=trace)
                finally:
                    try:
                        env.close()
                    finally:
                        if trace is not None:
                            trace.stop()
        finally:
            recording.disconnect()
        print(f"[rerun] Saved {', '.join(f'{name}={count}' for name, count in frame_counts.items())} frames to {output_path}",
              flush=True)
    finally:
        session.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
