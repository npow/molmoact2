"""Shared YAM rollout loop (prediction, IK, prefetch, execution) for every policy family.

Run directly it drives the left-arm conveyor GR00T checkpoint through a local Orchestral Worker;
``managed_rollout.py`` loads this file for all other families.
"""
import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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
from gripper_aperture import clip_predicted_aperture
from rollout_trace import RolloutTrace

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
                  state_padding=None, committed_actions=None, noise_seed=None):
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
        sensors["committed_actions"] = committed_actions
    if noise_seed is not None:
        sensors["noise_seed"] = noise_seed
    response = session.predict(cameras=cameras, state=query_state, instruction=instruction,
                               episode_start=episode_start, **sensors)
    chunk = np.asarray(response.actions, dtype=np.float32)
    if trace is not None:
        trace.prediction(prediction_id, response, chunk, session, time.monotonic() - start)
    if chunk.shape != action_shape or not np.isfinite(chunk).all():
        raise ValueError(f"{model_label} returned an invalid action chunk")
    if committed_actions is not None and not np.array_equal(chunk[:len(committed_actions)], committed_actions):
        raise ValueError(f"{model_label} did not preserve the committed action prefix")
    chunk = chunk[:max_plan_rows]
    if state_padding is not None:
        chunk = chunk[:, :len(state)]
    if pose_actions:
        from tool_pose_actions import UnreachableToolPose

        # The queued joints already select an IK branch. Solve only the new rows,
        # starting at the last queued target so the step bound holds across the handoff.
        kept = (queued_joints[:len(chunk)] if queued_joints is not None
                else np.empty((0, 7), dtype=np.float32))
        suffix = chunk[len(kept):]
        chunk = kept
        if len(suffix):
            try:
                solved = pose_actions.actions(suffix, kept[-1] if len(kept) else state, matrix_policy=matrix_policy)
                chunk = np.concatenate((kept, solved))
            except UnreachableToolPose as error:
                print(f"[{model_label.lower()}] {error}. Replanning after the queued targets.", flush=True)
    chunk = chunk.copy()
    chunk[:, 6] = clip_predicted_aperture(chunk[:, 6], model_label, tolerance=gripper_tolerance)
    return chunk, time.monotonic() - start


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
        stream_observations=False, noise_seed=None):
    if committed_actions and (chunk_fusion != "splice" or lead_steps < 1):
        raise ValueError("Committed actions require splice and at least one prefetch lead step")
    period = 1 / control_hz
    history_size = 1 - min(history_offsets, default=0)
    history = deque(maxlen=history_size)
    pending = ChunkPlan(chunk_fusion, blend_rows=fusion_rows)
    request_slot = 0
    future = None
    active_prediction = 0
    submitted_with = 0
    submitted_at = 0.0
    request_hold_ticks = 0
    request_prefix_changed = False
    observations = deque()
    rejected_chunks = 0
    rows = predictions = handoffs = misses = holds = 0
    call_ms, stage_rows = [], []
    next_tick = time.monotonic()
    started_at = next_tick
    deadline = next_tick + seconds
    predictor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{model_label.lower()}-predict")
    try:
        while time.monotonic() < deadline:
            while observations and observations[0].done():
                observations.popleft().result()
            obs = env.get_obs()
            observation_time = time.monotonic() - started_at if timestamp_observations else None
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
                    if trace is not None:
                        trace.event("Prediction %d failed: %s", predictions, error)
                    raise
                future = None
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
                if trace is not None:
                    if not len(chunk):
                        trace.event("Prediction %d rejected: unreachable plan", predictions)
                    elif missed:
                        trace.event("Prediction %d dropped after %d hold ticks", predictions, request_hold_ticks)
                    elif request_prefix_changed:
                        trace.event("Prediction %d dropped after the safety mask changed a committed target", predictions)
                    elif first_row is None:
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
                          f"{submitted_with} queued rows, {request_hold_ticks} hold ticks). Replanning.", flush=True)
                elif first_row is not None:
                    active_prediction = predictions
                    handoffs += bool(submitted_with)
            if len(history) == history_size and future is None and len(pending) <= lead_steps:
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
                request_prefix_changed = False
                prefix = pending.committed_actions() if committed_actions and submitted_with else None
                if trace is not None:
                    trace.event("Prediction %d requested with %d queued rows", predictions + 1, submitted_with)
                future = predictor.submit(checked_chunk, session, cameras, measured[:7].copy(),
                                          instruction, episode_start=(predictions == 0),
                                          action_shape=action_shape, model_label=model_label,
                                          gripper_tolerance=gripper_tolerance, pose_actions=pose_actions,
                                          matrix_policy=matrix_policy, observation_time=observation_time,
                                          max_plan_rows=max_plan_rows, history_inputs=history_inputs,
                                          trace=trace, prediction_id=predictions + 1,
                                          state_padding=state_padding, committed_actions=prefix, noise_seed=noise_seed)
                predictions += 1
            elif stream_observations:
                # Use the prediction executor to preserve Session order without blocking the control tick.
                observed_state = (
                    pose_actions.state(measured[:7].copy(), matrix_policy=matrix_policy)
                    if pose_actions else measured[:7].copy()
                )
                observations.append(predictor.submit(
                    session.observe, cameras=frames, state=observed_state, instruction=instruction,
                ))

            next_tick += period
            row = command = row_index = None
            if pending and time.monotonic() < deadline:
                row_index, row = pending.pop()
                if not shadow:
                    measured = mask.validate_state(env.get_robot_state()["joint_positions"], name="measured state")
                    target = measured.copy()
                    target[:7] = row
                    command = mask.command_target(target, measured)
                    env.step_command_only(command, wait=False)
                    if committed_actions and future is not None and submitted_with:
                        # The hardware limit takes precedence. Replan from fresh feedback if a conditioned
                        # target had to change, instead of applying a suffix based on a different command.
                        request_prefix_changed |= not np.allclose(command[:7], row, rtol=0, atol=1e-6)
                    if temporal_history is not None:
                        temporal_history.commanded(command[:7])
                rows += 1
                next_tick = max(next_tick, time.monotonic() + period)
            elif future is not None:
                request_hold_ticks += 1
                holds += 1
            if trace is not None:
                trace.tick(measured, row, command, active_prediction, row_index,
                           holding=row is None and future is not None)
            time.sleep(max(0, next_tick - time.monotonic()))
    finally:
        predictor.shutdown(wait=False, cancel_futures=True)
        if call_ms:
            print(latency_line(model_label, call_ms, stage_rows, final=True), flush=True)
    print(f"{model_label} rollout finished. {predictions} predictions, {rows} {'shadow rows' if shadow else 'commands'}, "
          f"{handoffs} prefetched handoffs, {misses} missed boundaries, {holds} hold ticks.")


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
