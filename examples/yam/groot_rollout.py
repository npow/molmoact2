"""Run the left-arm conveyor GR00T checkpoint through a local Orchestral Worker."""
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

from launch_yaml_eval_molmoact import _build_env, resolve_bimanual_execution_mask, validate_bimanual_model_arm_order

CAMERAS = {"belt": "front_camera_rgb", "wrist": "left_camera_rgb"}
CONTROL_HZ = 30
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
                  model_label="GR00T", gripper_tolerance=0.05, pose_actions=None, matrix_policy=False,
                  observation_time=None, max_plan_rows=None):
    start = time.monotonic()
    query_state = pose_actions.state(state, matrix_policy=matrix_policy) if pose_actions else state
    sensors = {} if observation_time is None else {"timestamp": np.array([observation_time], dtype=np.float64)}
    response = session.predict(cameras=cameras, state=query_state, instruction=instruction,
                               episode_start=episode_start, **sensors)
    chunk = np.asarray(response.actions, dtype=np.float32)
    if chunk.shape != action_shape or not np.isfinite(chunk).all():
        raise ValueError(f"{model_label} returned an invalid action chunk")
    chunk = chunk[:max_plan_rows]
    if pose_actions:
        from tool_pose_actions import UnreachableToolPose

        try:
            chunk = pose_actions.actions(chunk, state, matrix_policy=matrix_policy)
        except UnreachableToolPose as error:
            print(f"[{model_label.lower()}] {error}. Holding and replanning.", flush=True)
            return np.empty((0, 7), dtype=np.float32), time.monotonic() - start
    gripper = chunk[:, 6]
    lower, upper = -gripper_tolerance, 1.0 + gripper_tolerance
    if np.any((gripper < lower) | (gripper > upper)):
        raise ValueError(
            f"{model_label} returned gripper targets ranging from {gripper.min():.4f} to "
            f"{gripper.max():.4f}, outside the accepted aperture range [{lower:.2f}, {upper:.2f}]"
        )
    chunk = chunk.copy()
    chunk[:, 6] = np.clip(chunk[:, 6], 0, 1)
    return chunk, time.monotonic() - start


def warm_session(session, instruction):
    with np.load(Path(__file__).with_name("warmup-inputs.npz")) as sample:
        cameras = {name: sample[name] for name in CAMERAS}
        state = sample["state"]
    for i in range(3):
        _, duration = checked_chunk(session, cameras, state, instruction, episode_start=(i == 0))
        print(f"[gr00t] Warmup {i + 1}/3 finished in {duration * 1000:.0f} ms.", flush=True)


def run(env, session, mask, *, seconds, instruction, shadow, lead_steps=8,
        camera_bindings=CAMERAS, history_offsets=(-3, 0), control_hz=CONTROL_HZ,
        action_shape=ACTION_SHAPE, model_label="GR00T", gripper_tolerance=0.05, pose_actions=None, matrix_policy=False,
        timestamp_observations=False, max_plan_rows=None):
    period = 1 / control_hz
    history_size = 1 - min(history_offsets, default=0)
    history = deque(maxlen=history_size)
    pending = deque()
    future = None
    submitted_with = 0
    request_hold_ticks = 0
    rejected_chunks = 0
    rows = predictions = handoffs = misses = holds = 0
    next_tick = time.monotonic()
    started_at = next_tick
    deadline = next_tick + seconds
    predictor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"{model_label.lower()}-predict")
    try:
        while time.monotonic() < deadline:
            obs = env.get_obs()
            observation_time = time.monotonic() - started_at if timestamp_observations else None
            history.append(frames_for_policy(obs, session.grant.io_spec.cameras, camera_bindings))
            measured = mask.validate_state(obs["joint_positions"], name="policy input state")
            if not pending and future is not None and future.done():
                chunk, elapsed = future.result()
                future = None
                if len(chunk):
                    rejected_chunks = 0
                else:
                    rejected_chunks += 1
                    if rejected_chunks == 3:
                        raise SystemExit(f"{model_label} returned three consecutive unreachable plans; rollout stopped")
                # Each elapsed command in the active plan corresponds to one row in the
                # new plan, which starts at the observation taken for this prediction.
                # A missed deadline invalidates a prefetched plan. Wait for a fresh
                # observation instead of applying a plan computed before the hold.
                if submitted_with and request_hold_ticks:
                    misses += 1
                    print(f"[{model_label.lower()}] Prefetch missed the boundary ({elapsed * 1000:.0f} ms). "
                          "Replanning.", flush=True)
                else:
                    pending.extend(chunk[submitted_with:])
                    handoffs += bool(submitted_with)
            if len(history) == history_size and future is None and len(pending) <= lead_steps:
                cameras = (
                    {name: np.stack([history[offset - 1][name] for offset in history_offsets])
                     for name in camera_bindings}
                    if history_offsets else history[-1]
                )
                submitted_with = len(pending)
                request_hold_ticks = 0
                future = predictor.submit(checked_chunk, session, cameras, measured[:7].copy(),
                                          instruction, episode_start=(predictions == 0),
                                          action_shape=action_shape, model_label=model_label,
                                          gripper_tolerance=gripper_tolerance, pose_actions=pose_actions,
                                          matrix_policy=matrix_policy, observation_time=observation_time,
                                          max_plan_rows=max_plan_rows)
                predictions += 1

            next_tick += period
            if pending and time.monotonic() < deadline:
                row = pending.popleft()
                if not shadow:
                    measured = mask.validate_state(env.get_robot_state()["joint_positions"], name="measured state")
                    target = measured.copy()
                    target[:7] = row
                    env.step_command_only(mask.command_target(target, measured), wait=False)
                rows += 1
                next_tick = max(next_tick, time.monotonic() + period)
            elif future is not None:
                request_hold_ticks += 1
                holds += 1
            time.sleep(max(0, next_tick - time.monotonic()))
    finally:
        predictor.shutdown(wait=False, cancel_futures=True)
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


def record_blueprint(camera_bindings):
    views = [
        rrb.Spatial2DView(origin=f"/cameras/{name}", name=f"{name} camera")
        for name in camera_bindings
    ]
    return rrb.Blueprint(rrb.Horizontal(*views, column_shares=[1] * len(views)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--orchestral-grant", type=Path, required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--right-config-path", required=True)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--instruction", default="Pick up the red lid from the conveyor belt.")
    parser.add_argument("--prefetch-lead-steps", type=int, default=8)
    parser.add_argument("--rrd", type=Path, help="Rerun recording path (default: yam_eval_runs/rerun/gr00t_<timestamp>.rrd)")
    parser.add_argument("--shadow", action="store_true", help="Read cameras and feedback without sending policy commands.")
    args = parser.parse_args()
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("--seconds must be positive and finite")
    if not 0 <= args.prefetch_lead_steps < ACTION_SHAPE[0]:
        parser.error("--prefetch-lead-steps must be between 0 and 15")
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
                recording.save(output_path, default_blueprint=record_blueprint(CAMERAS))
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
                try:
                    run(env, session, mask, seconds=args.seconds, instruction=args.instruction, shadow=args.shadow,
                        lead_steps=args.prefetch_lead_steps)
                finally:
                    env.close()
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
