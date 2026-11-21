"""Rollout and data-collection script for the predictive YAM conveyor controller.

Runs the classical vision-guided predictive catching controller on the real
bimanual YAM rig (using ``pi05_red_lid_conveyor_left.yaml`` + right hold arm),
with optional episode recording (``--save-data``) and automatic export to a
7-DoF ``i2rt_yam`` LeRobot dataset (``--export-lerobot``) matching the schema of
``npow/red-lid-conveyor-speedaug`` for fine-tuning pi0.5.

Example usage:
    # Dry-run camera + velocity estimation check (no motor commands)
    ./examples/yam/run-rollout.sh --predictive --dry-run

    # Collect 20 catching demos and export a 7-DoF LeRobot dataset at 15 Hz
    ./examples/yam/run-rollout.sh --predictive --save-data --export-lerobot -n 20
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import av
import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from omegaconf import OmegaConf
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_utils import EvalRolloutSaver, move_rollout, prompt_label  # noqa: E402
from launch_yaml_eval_molmoact import (  # noqa: E402
    _build_env,
    resolve_bimanual_execution_mask,
    return_to_captured_rest_pose,
    validate_bimanual_model_arm_order,
)
from predictive_conveyor_controller import (  # noqa: E402
    BeltCalib,
    RealYamPredictiveController,
    YAM_JOINT_MAX,
    YAM_JOINT_MIN,
    run_predictive_calibration,
)

logger = logging.getLogger("predictive_conveyor_rollout")
ARM_DIM = 7

JOINT_NAMES_7D = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
CAMERA_MAP_TO_LEROBOT = (
    ("left_rgb", "observation.images.left"),
    ("front_rgb", "observation.images.middle"),
    ("right_rgb", "observation.images.right"),
)


def _write_h264_mp4(png_paths: Sequence[Path], out_path: Path, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(png_paths[0]) as first_im:
        w, h = first_im.size
    container = av.open(str(out_path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = w
    stream.height = h
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "fast"}
    for p in png_paths:
        with Image.open(p) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        for pkt in stream.encode(frame):
            container.mux(pkt)
    for pkt in stream.encode():
        container.mux(pkt)
    container.close()


def export_7d_lerobot_dataset(
    rollout_dirs: Sequence[Path],
    output_dir: Path,
    *,
    active_arm_slice: slice,
    fps: int = 15,
    instruction: str = "Pick up the red lid from the conveyor belt",
    robot_type: str = "i2rt_yam",
) -> Optional[Path]:
    """Exports saved EvalRolloutSaver dirs to a 7-DoF LeRobot v2.1 dataset.

    Matches the exact column schema and video keys of
    ``npow/red-lid-conveyor-speedaug`` so pi0.5 can fine-tune directly on it.
    """
    valid_dirs = [Path(d) for d in rollout_dirs if (Path(d) / "episode.h5").is_file()]
    if not valid_dirs:
        logger.warning("No valid episode.h5 files to export to LeRobot.")
        return None

    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        output_dir = output_dir.parent / f"{output_dir.name}_{datetime.now().strftime('%H%M%S')}"
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)
    (output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)

    global_index = 0
    episodes_meta: List[Dict[str, Any]] = []
    img_shape: Optional[List[int]] = None

    for ep_idx, ep_dir in enumerate(valid_dirs):
        with h5py.File(ep_dir / "episode.h5", "r") as f:
            states_all = np.asarray(f["state"], dtype=np.float32)
            actions_all = (
                np.asarray(f["action"], dtype=np.float32)
                if "action" in f
                else np.asarray(f["next_state"], dtype=np.float32)
            )
        states_7d = states_all[:, active_arm_slice]
        actions_7d = actions_all[:, active_arm_slice]

        cam_lists: Dict[str, List[Path]] = {}
        for src_cam, _ in CAMERA_MAP_TO_LEROBOT:
            cdir = ep_dir / src_cam
            if cdir.is_dir():
                files = sorted(cdir.glob("*.png"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
                if files:
                    cam_lists[src_cam] = files

        if len(cam_lists) < len(CAMERA_MAP_TO_LEROBOT):
            logger.warning("Skipping %s (missing camera streams: found %s)", ep_dir, list(cam_lists))
            continue

        n_frames = min(len(states_7d), len(actions_7d), *(len(v) for v in cam_lists.values()))
        if n_frames < 2:
            logger.warning("Skipping %s (too short: %d frames)", ep_dir, n_frames)
            continue

        if img_shape is None:
            with Image.open(cam_lists["front_rgb"][0]) as im0:
                w0, h0 = im0.size
                img_shape = [h0, w0, 3]

        states_7d = states_7d[:n_frames]
        actions_7d = actions_7d[:n_frames]

        for src_cam, dst_key in CAMERA_MAP_TO_LEROBOT:
            vid_path = output_dir / "videos" / "chunk-000" / dst_key / f"episode_{ep_idx:06d}.mp4"
            _write_h264_mp4(cam_lists[src_cam][:n_frames], vid_path, fps=fps)

        timestamps = (np.arange(n_frames, dtype=np.float32) / float(fps)).tolist()
        frame_indices = list(range(n_frames))
        ep_indices = [ep_idx] * n_frames
        indices = list(range(global_index, global_index + n_frames))
        task_indices = [0] * n_frames
        global_index += n_frames

        state_arr = pa.FixedSizeListArray.from_arrays(pa.array(states_7d.reshape(-1), type=pa.float32()), 7)
        action_arr = pa.FixedSizeListArray.from_arrays(pa.array(actions_7d.reshape(-1), type=pa.float32()), 7)
        table = pa.table(
            {
                "observation.state": state_arr,
                "action": action_arr,
                "timestamp": pa.array(timestamps, type=pa.float32()),
                "frame_index": pa.array(frame_indices, type=pa.int64()),
                "episode_index": pa.array(ep_indices, type=pa.int64()),
                "index": pa.array(indices, type=pa.int64()),
                "task_index": pa.array(task_indices, type=pa.int64()),
            }
        )
        pq.write_table(table, output_dir / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet")
        episodes_meta.append({"episode_index": ep_idx, "tasks": [instruction], "length": n_frames})

    if not episodes_meta:
        logger.warning("No episodes exported to %s.", output_dir)
        return None

    h_im, w_im, c_im = img_shape or [480, 640, 3]
    features: Dict[str, Any] = {
        "observation.state": {"dtype": "float32", "shape": [7], "names": [JOINT_NAMES_7D]},
        "action": {"dtype": "float32", "shape": [7], "names": [JOINT_NAMES_7D]},
    }
    for _, dst_key in CAMERA_MAP_TO_LEROBOT:
        features[dst_key] = {
            "dtype": "video",
            "shape": [h_im, w_im, c_im],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": h_im,
                "video.width": w_im,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": c_im,
                "has_audio": False,
            },
        }
    for name, dtype in (("timestamp", "float32"), ("frame_index", "int64"),
                        ("episode_index", "int64"), ("index", "int64"), ("task_index", "int64")):
        features[name] = {"dtype": dtype, "shape": [1], "names": None}

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": len(episodes_meta),
        "total_frames": global_index,
        "total_tasks": 1,
        "total_videos": len(episodes_meta) * len(CAMERA_MAP_TO_LEROBOT),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes_meta)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    (output_dir / "meta" / "info.json").write_text(json.dumps(info, indent=4) + "\n", encoding="utf-8")
    (output_dir / "meta" / "tasks.jsonl").write_text(
        json.dumps({"task_index": 0, "task": instruction}) + "\n", encoding="utf-8"
    )
    (output_dir / "meta" / "episodes.jsonl").write_text(
        "\n".join(json.dumps(ep) for ep in episodes_meta) + "\n", encoding="utf-8"
    )
    logger.info(
        "Exported 7-DoF LeRobot dataset (%d episodes, %d frames @ %d Hz) to %s",
        len(episodes_meta),
        global_index,
        fps,
        output_dir,
    )
    return output_dir


def _move_to_ready(
    env: Any,
    mask: Any,
    ep_slice: slice,
    ready_q: np.ndarray,
    *,
    dt: float,
    max_joint_step: float = 0.04,
) -> None:
    """Smoothly interpolate the active arm from its current pose to ready_q."""
    cur = mask.validate_state(env.get_robot_state()["joint_positions"], name="rearm_state")
    cur_arm = cur[ep_slice]
    max_delta = float(np.max(np.abs(ready_q[:6] - cur_arm[:6])))
    steps = max(20, int(np.ceil(max_delta / max(max_joint_step, 1e-3))))
    for k in range(1, steps + 1):
        alpha = 0.5 * (1.0 - np.cos(np.pi * (k / steps)))
        tgt = cur.copy()
        tgt[ep_slice] = cur_arm + alpha * (ready_q - cur_arm)
        env.step_command_only(mask.command_target(tgt, cur), wait=False)
        time.sleep(dt)


def run_collection_session(
    env: Any,
    mask: Any,
    *,
    calib: BeltCalib,
    control_hz: float,
    record_hz: float,
    num_episodes: int,
    episode_timeout_s: float,
    instruction: str,
    save_data: bool,
    base_save_dir: Path,
    auto_label: bool,
    export_lerobot: bool,
    lerobot_dir: Optional[Path],
    use_color_mask: bool,
    dry_run: bool,
    max_joint_step: float = 0.12,
    saver_max_workers: int = 2,
    png_compress_level: int = 1,
) -> Dict[str, Any]:
    """Runs episode-by-episode predictive catches with optional disk + LeRobot capture."""
    ep_slice = slice(0, ARM_DIM) if mask.active_arm_side == "left" else slice(ARM_DIM, 2 * ARM_DIM)
    ctrl = RealYamPredictiveController(calib, use_color_mask=use_color_mask, auto_reset=False)
    dt = 1.0 / max(control_hz, 1.0)

    if not dry_run:
        start_state = mask.validate_state(env.get_robot_state()["joint_positions"], name="startup_state")
        ready_q = ctrl.ready_joints(start_state[ep_slice][:6])
        logger.info("Moving active arm smoothly to hover ready pose...")
        _move_to_ready(env, mask, ep_slice, ready_q, dt=dt)

    print("Capturing empty belt background (keep lid off the belt for ~0.5s)...")
    bg_frames = []
    for _ in range(12):
        obs = env.get_obs()
        bg_frames.append(obs["front_camera_rgb"])
        time.sleep(0.03)
    ctrl.capture_background(bg_frames)
    record_stride = max(1, int(round(control_hz / max(record_hz, 1.0))))
    effective_record_fps = max(1, int(round(control_hz / record_stride)))

    saved_success_dirs: List[Path] = []
    saved_all_dirs: List[Path] = []
    catches = 0

    for ep_idx in range(num_episodes):
        measured0 = mask.validate_state(env.get_robot_state()["joint_positions"], name="start_state")
        ctrl.reset_episode(measured0[ep_slice])

        saver: Optional[EvalRolloutSaver] = None
        rollout_dir: Optional[Path] = None
        if save_data:
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            rollout_dir = base_save_dir / "eval" / f"pred_{ts_tag}"
            manifest = {
                "schema_version": 1,
                "controller": "predictive_conveyor",
                "active_arm_side": mask.active_arm_side,
                "control_hz": control_hz,
                "record_fps": effective_record_fps,
                "dry_run": dry_run,
                "reproducibility": {"rollout_index": ep_idx},
            }
            saver = EvalRolloutSaver(
                rollout_dir,
                instruction=instruction,
                max_workers=saver_max_workers,
                png_compress_level=png_compress_level,
                rollout_manifest=manifest,
            )

        logger.info("Episode %d/%d armed — send red lid down the conveyor...", ep_idx + 1, num_episodes)
        t_start = time.monotonic()
        deadline = t_start + episode_timeout_s
        tick = 0
        hold_started_t: Optional[float] = None
        caught = False

        try:
            while time.monotonic() < deadline:
                t_tick = time.monotonic()
                t_rel = t_tick - t_start
                obs_pre = env.get_obs()
                measured = mask.validate_state(obs_pre["joint_positions"], name="measured_state")
                q7_act = measured[ep_slice]

                q7_cmd, info = ctrl.step(t_rel, obs_pre.get("front_camera_rgb"), q7_act)
                delta = np.clip(q7_cmd[:6] - q7_act[:6], -max_joint_step, max_joint_step)
                q7_safe = q7_cmd.copy()
                q7_safe[:6] = np.clip(q7_act[:6] + delta, YAM_JOINT_MIN, YAM_JOINT_MAX)
                q7_safe[6] = float(np.clip(q7_safe[6], 0.0, 1.0))

                full_target = measured.copy()
                full_target[ep_slice] = q7_safe
                applied_target = mask.command_target(full_target, measured)

                if not dry_run:
                    env.step_command_only(applied_target, wait=False)

                sleep_s = (t_tick + dt) - time.monotonic()
                if sleep_s > 0.0:
                    time.sleep(sleep_s)

                if saver is not None and (tick % record_stride == 0):
                    obs_post = env.get_robot_state()
                    saver.add_step(obs_pre, obs_post, action=applied_target)

                if info["phase"] == "HOLD":
                    if hold_started_t is None:
                        hold_started_t = t_rel
                        caught = True
                    elif (t_rel - hold_started_t) >= 0.8:
                        break
                tick += 1
        finally:
            if saver is not None:
                saver.flush()

        if caught:
            catches += 1

        if saver is not None and rollout_dir is not None:
            if auto_label:
                label = "success" if caught else "failure"
            else:
                label = prompt_label()
            if label in ("success", "failure"):
                dest = move_rollout(rollout_dir, label, base_save_dir)
                saved_all_dirs.append(dest)
                if label == "success":
                    saved_success_dirs.append(dest)
                logger.info("Saved episode %d/%d -> %s (%s)", ep_idx + 1, num_episodes, label, dest)
            else:
                saved_all_dirs.append(rollout_dir)

        if ep_idx + 1 < num_episodes and not dry_run:
            # Smoothly reopen/return to ready observation pose for the next lid.
            cur = mask.validate_state(env.get_robot_state()["joint_positions"], name="rearm_state")
            ready_q = ctrl.ready_joints(cur[ep_slice][:6])
            _move_to_ready(env, mask, ep_slice, ready_q, dt=dt)

    exported_path: Optional[Path] = None
    if save_data and export_lerobot and saved_success_dirs:
        out_lerobot = lerobot_dir or (
            base_save_dir / "lerobot_7d" / datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        exported_path = export_7d_lerobot_dataset(
            saved_success_dirs,
            out_lerobot,
            active_arm_slice=ep_slice,
            fps=effective_record_fps,
            instruction=instruction,
        )

    return {
        "episodes": num_episodes,
        "catches": catches,
        "saved_success": len(saved_success_dirs),
        "saved_total": len(saved_all_dirs),
        "lerobot_dataset": str(exported_path) if exported_path else None,
        "dry_run": dry_run,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config-path", required=True, help="primary (left-arm) rig config")
    parser.add_argument("--right-config-path", required=True, help="secondary (right-arm) rig config")
    parser.add_argument("--active-arm-side", choices=["left", "right"], default=None)
    parser.add_argument("--execution-mode", choices=["active_arm_hold", "shadow"], default=None)
    parser.add_argument("--instruction", default=None, help="task instruction for dataset metadata")
    parser.add_argument("--control-hz", type=float, default=60.0, help="controller loop rate (Hz)")
    parser.add_argument("--record-hz", type=float, default=15.0, help="dataset recording rate (Hz)")
    parser.add_argument("-n", "--num-rollouts", type=int, default=1, help="number of catch episodes")
    parser.add_argument("--seconds", type=float, default=15.0, help="per-episode timeout (s)")
    parser.add_argument("--save-data", action="store_true", help="record episode.h5 and camera PNGs")
    parser.add_argument("--save-dir", type=Path, default=None, help="override storage base_dir/task_directory")
    parser.add_argument("--no-auto-label", action="store_true", help="prompt y/n after each episode")
    parser.add_argument("--export-lerobot", action="store_true",
                        help="export successful episodes to a 7-DoF LeRobot dataset at session end")
    parser.add_argument("--lerobot-dir", type=Path, default=None, help="output dir for exported LeRobot dataset")
    parser.add_argument("--dry-run", action="store_true", help="track and record without commanding motors")
    parser.add_argument("--calibrate", action="store_true", help="run 4-point belt calibration first")
    parser.add_argument("--calib-file", type=Path,
                        default=Path(__file__).resolve().parent / "conveyor_calib.json")
    parser.add_argument("--use-color-mask", action="store_true",
                        help="segment red lid via RGB instead of IR background subtraction")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    left_cfg = OmegaConf.to_container(OmegaConf.load(args.config_path), resolve=True)
    right_cfg = OmegaConf.to_container(OmegaConf.load(args.right_config_path), resolve=True)
    validate_bimanual_model_arm_order(left_cfg, right_cfg)
    eval_cfg = left_cfg.get("eval") or {}
    bimanual_cfg = eval_cfg.get("bimanual") or {}
    mask = resolve_bimanual_execution_mask(
        bimanual=True,
        active_arm_side=args.active_arm_side or bimanual_cfg.get("active_arm_side"),
        execution_mode=args.execution_mode or bimanual_cfg.get("execution_mode"),
        both_arm_max_delta=bimanual_cfg.get("both_arm_max_delta"),
    )
    if mask.active_arm_side == "both":
        raise SystemExit("this script drives one arm; use --active-arm-side left or right")

    storage = left_cfg.get("storage") or {}
    base_save_dir = args.save_dir or (
        Path(storage.get("base_dir", "./yam_eval_runs")).expanduser()
        / str(storage.get("task_directory", "predictive_conveyor"))
    )
    instruction = args.instruction or storage.get("language_instruction") or "Pick up the red lid from the conveyor belt"

    env, *_ = _build_env(
        SimpleNamespace(
            config_path=args.config_path,
            right_config_path=args.right_config_path,
            control_hz=args.control_hz,
            max_steps=None,
        )
    )
    try:
        rest = np.asarray(env.get_robot_state()["joint_positions"], dtype=np.float32).copy()
        if args.calibrate:
            calib = run_predictive_calibration(env, mask, args.calib_file)
        else:
            calib = BeltCalib.load(args.calib_file)

        input(
            f"Predictive conveyor rollout ({args.control_hz:.0f} Hz, save_data={args.save_data}, "
            f"export_lerobot={args.export_lerobot}, dry_run={args.dry_run}). "
            "Clear workspace, then press Enter to start (Ctrl-C ends)... "
        )
        completed = True
        try:
            summary = run_collection_session(
                env,
                mask,
                calib=calib,
                control_hz=args.control_hz,
                record_hz=args.record_hz,
                num_episodes=args.num_rollouts,
                episode_timeout_s=args.seconds,
                instruction=instruction,
                save_data=args.save_data,
                base_save_dir=base_save_dir,
                auto_label=not args.no_auto_label,
                export_lerobot=args.export_lerobot,
                lerobot_dir=args.lerobot_dir,
                use_color_mask=args.use_color_mask,
                dry_run=args.dry_run,
                saver_max_workers=int(storage.get("saver_max_workers", 2)),
                png_compress_level=int(storage.get("png_compress_level", 1)),
            )
            logger.info("Session summary: %s", summary)
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        except Exception:
            completed = False
            raise
        finally:
            if completed and not args.dry_run and eval_cfg.get("return_to_rest_on_exit", True):
                return_to_captured_rest_pose(
                    env, rest, max_joint_step=float(eval_cfg.get("return_to_rest_max_joint_step", 0.015))
                )
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
