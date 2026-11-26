"""Rollout and offline-RRD replay runner for the YAM holesaw CV conveyor controller.

Live robot usage (left arm active, right arm held):
    PYTHONPATH=examples/yam:../i2rt .venv/bin/python examples/yam/holesaw_conveyor_rollout.py \\
        --config-path examples/yam/configs/pi05_holesaw_conveyor_left.yaml \\
        --right-config-path examples/yam/configs/yam_deployment_right.yaml \\
        --target-count 3 --dry-run

Offline verification against a cached RRD pickle (zero robot hardware):
    PYTHONPATH=examples/yam .venv/bin/python examples/yam/holesaw_conveyor_rollout.py \\
        --replay-pickle /path/to/cached_run.pkl \\
        --out-video /tmp/holesaw_replay.mp4
"""

from __future__ import annotations

import argparse
import io
import logging
import pickle
import subprocess
import sys
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from holesaw_conveyor_controller import (  # noqa: E402
    DEFAULT_URDF_PATH,
    HolesawBeltCalib,
    HolesawConveyorController,
    HolesawYamKinematics,
    YAM_JOINT_MAX,
    YAM_JOINT_MIN,
    draw_debug_overlay,
)

logger = logging.getLogger("holesaw_conveyor_rollout")
ARM_DIM = 7


def _move_smooth(
    env: Any,
    mask: Any,
    ep_slice: slice,
    target_q7: np.ndarray,
    *,
    dt: float,
    max_joint_step: float = 0.04,
) -> None:
    cur = mask.validate_state(env.get_robot_state()["joint_positions"], name="rearm_state")
    cur_arm = cur[ep_slice]
    max_delta = float(np.max(np.abs(target_q7[:6] - cur_arm[:6])))
    steps = max(20, int(np.ceil(max_delta / max(max_joint_step, 1e-3))))
    for k in range(1, steps + 1):
        alpha = 0.5 * (1.0 - np.cos(np.pi * (k / steps)))
        tgt = cur.copy()
        tgt[ep_slice] = cur_arm + alpha * (target_q7 - cur_arm)
        env.step_command_only(mask.command_target(tgt, cur), wait=False)
        time.sleep(dt)


def run_interactive_calibration(env: Any, mask: Any, calib_path: Path) -> HolesawBeltCalib:
    """4-point belt-to-camera homography calibration on the real holesaw rig."""
    kin = HolesawYamKinematics(DEFAULT_URDF_PATH)
    ep_slice = slice(0, ARM_DIM) if mask.active_arm_side == "left" else slice(ARM_DIM, 2 * ARM_DIM)
    pts_uv: List[List[float]] = []
    pts_xy: List[List[float]] = []
    zs: List[float] = []

    print("\n=== Holesaw Conveyor Calibration (4 points on belt surface) ===")
    for idx in range(4):
        input(f"  Point {idx + 1}/4: place gripper fingertips at belt point and press Enter...")
        obs = env.get_obs()
        q7 = mask.validate_state(obs["joint_positions"], name="calib_state")[ep_slice]
        p, _ = kin.fk(q7[:6])
        zs.append(float(p[2]))
        pts_xy.append([float(p[0]), float(p[1])])
        raw = input(f"    FK=({p[0]:+.3f}, {p[1]:+.3f}, {p[2]:+.3f}). Enter 'u,v' in 640x480 top cam: ").strip()
        u_s, v_s = raw.replace(" ", "").split(",")
        pts_uv.append([float(u_s), float(v_s)])

    H, _ = cv2.findHomography(np.asarray(pts_uv, dtype=np.float64), np.asarray(pts_xy, dtype=np.float64))
    if H is None:
        raise RuntimeError("Failed to compute belt homography from 4 points")
    calib = HolesawBeltCalib(homography=H, z_grasp=float(np.mean(zs)))
    calib.save(calib_path)
    print(f"Saved holesaw calibration to {calib_path}")
    return calib


def run_live_rollout(args: argparse.Namespace) -> Dict[str, Any]:
    from launch_yaml_eval_molmoact import (
        _build_env,
        resolve_bimanual_execution_mask,
        return_to_captured_rest_pose,
        validate_bimanual_model_arm_order,
    )

    env, cfg, right_cfg, bimanual = _build_env(
        SimpleNamespace(
            config_path=args.config_path,
            right_config_path=args.right_config_path,
            control_hz=args.control_hz,
            max_steps=None,
        )
    )
    if bimanual and right_cfg is not None:
        validate_bimanual_model_arm_order(cfg, right_cfg)
    eval_cfg = cfg.get("eval") or {}
    bimanual_cfg = eval_cfg.get("bimanual") or {}
    mask = resolve_bimanual_execution_mask(
        bimanual=bimanual,
        active_arm_side=args.active_arm_side or bimanual_cfg.get("active_arm_side"),
        execution_mode=bimanual_cfg.get("execution_mode"),
        both_arm_max_delta=bimanual_cfg.get("both_arm_max_delta"),
    )
    if mask.active_arm_side == "both":
        raise SystemExit("this script drives one arm; use --active-arm-side left or right")
    ep_slice = slice(0, ARM_DIM) if mask.active_arm_side == "left" else slice(ARM_DIM, 2 * ARM_DIM)

    rest_pose = mask.validate_state(env.get_robot_state()["joint_positions"], name="initial_rest")
    calib_path = Path(args.calib_path) if args.calib_path else None
    if args.calibrate:
        if calib_path is None:
            raise ValueError("--calibrate requires --calib-path")
        run_interactive_calibration(env, mask, calib_path)
        return {"calibrated": True, "calib_path": str(calib_path)}

    calib = HolesawBeltCalib.load(calib_path)
    ctrl = HolesawConveyorController(
        calib,
        target_count=args.target_count,
        pd_lag_s=args.pd_lag_sec,
        verify_grasp=not args.no_verify_grasp,
    )
    dt = 1.0 / max(args.control_hz, 1.0)

    try:
        if not args.dry_run:
            ready_q7 = ctrl.ready_joints(rest_pose[ep_slice][:6])
            logger.info("Moving active arm smoothly to hover ready pose...")
            _move_smooth(env, mask, ep_slice, ready_q7, dt=dt)

        t0 = time.monotonic()
        deadline = t0 + float(args.seconds)
        ticks = 0
        last_log = 0.0
        max_step = float(args.max_joint_step)

        while time.monotonic() < deadline:
            t_tick = time.monotonic()
            t_rel = t_tick - t0
            obs = env.get_obs()
            capture_t = env.last_frame_monotonic.get("front_camera")
            frame_t = None if capture_t is None else float(capture_t - t0)
            measured = mask.validate_state(obs["joint_positions"], name="measured_state")
            q7_act = measured[ep_slice]

            q7_cmd, info = ctrl.step(
                t_rel,
                obs.get("front_camera_rgb"),
                q7_act,
                frame_t=frame_t,
            )
            requested_delta = q7_cmd[:6] - q7_act[:6]
            clipped_joints = int(np.count_nonzero(np.abs(requested_delta) > max_step))
            delta = np.clip(requested_delta, -max_step, max_step)
            q7_safe = q7_cmd.copy()
            q7_safe[:6] = np.clip(q7_act[:6] + delta, YAM_JOINT_MIN, YAM_JOINT_MAX)
            q7_safe[6] = float(np.clip(q7_safe[6], 0.0, 1.0))

            actual_ee, _ = ctrl.kin.fk(q7_act[:6])
            ee_error_mm = 1000.0 * float(np.linalg.norm(actual_ee - np.asarray(info["p_des"])))

            if not args.dry_run:
                target = measured.copy()
                target[ep_slice] = q7_safe
                env.step_command_only(mask.command_target(target, measured), wait=False)

            log_period = 0.1 if info["phase"] in {"PRE_POSITION", "SYNC_ACCEL", "DESCEND", "GRASP"} else 0.5
            if t_rel - last_log >= log_period:
                last_log = t_rel
                logger.info(
                    "t=%.2fs phase=%s catches=%d misses=%d tracks=%d(lock=%d) vy=%.3f "
                    "ee_y=%.3f cmd_y=%.3f ee_err=%.1fmm ik=%.1fmm clipped=%d/6 lead=%.0fms dry=%s",
                    t_rel,
                    info["phase"],
                    info["catches"],
                    info["misses"],
                    info["num_tracks"],
                    info["num_locked"],
                    info["est_vy_mps"],
                    float(actual_ee[1]),
                    float(info["p_des"][1]),
                    ee_error_mm,
                    info["ik_pos_err_mm"],
                    clipped_joints,
                    info["prediction_lead_ms"],
                    args.dry_run,
                )

            ticks += 1
            if ctrl.phase == "DONE":
                logger.info("Reached target_count=%d catches cleanly!", args.target_count)
                break

            sleep_s = (t_tick + dt) - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)

        return {
            "ticks": ticks,
            "catches": ctrl.catches,
            "misses": ctrl.misses,
            "final_phase": ctrl.phase,
            "dry_run": args.dry_run,
        }
    finally:
        try:
            if not args.dry_run:
                return_to_captured_rest_pose(
                    env,
                    rest_pose,
                    max_joint_step=float(eval_cfg.get("return_to_rest_max_joint_step", 0.015)),
                )
        finally:
            env.close()


def run_offline_replay(
    pickle_path: Path,
    *,
    calib_path: Optional[Path] = None,
    out_video: Optional[Path] = None,
    target_count: int = 0,
    stride: int = 2,
) -> Dict[str, Any]:
    """Replay a cached RRD pickle through the full CV detector, tracker, state machine, and IK."""
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    calib = HolesawBeltCalib.load(calib_path)
    # In offline replay the simulated arm follows its own IK commands (assume successful grasp
    # on belt so we exercise the full OBSERVE -> CATCH -> TRANSIT -> DROP -> RETURN cycle).
    ctrl = HolesawConveyorController(calib, target_count=target_count, verify_grasp=False)
    q_sim = ctrl.ready_joints().astype(np.float64)

    imgs = data["imgs"]
    frames_out: List[np.ndarray] = []
    max_ik_err_mm = 0.0
    ik_errs: List[float] = []
    locked_tracks_seen = set()
    phase_counts: Dict[str, int] = {}
    catch_events: List[Dict[str, Any]] = []
    prev_phase = ctrl.phase

    for idx in range(0, len(imgs), max(1, stride)):
        t_s, blob = imgs[idx]
        img_rgb = np.asarray(Image.open(io.BytesIO(blob)).convert("RGB"), dtype=np.uint8)
        q7_cmd, info = ctrl.step(t_s, img_rgb, q_sim)
        # Advance simulated robot smoothly toward commanded pose.
        q_sim[:6] = q7_cmd[:6]
        q_sim[6] = 0.22 if q7_cmd[6] < 0.2 else q7_cmd[6]

        err_mm = float(info["ik_pos_err_mm"])
        ik_errs.append(err_mm)
        max_ik_err_mm = max(max_ik_err_mm, err_mm)
        phase_counts[info["phase"]] = phase_counts.get(info["phase"], 0) + 1

        for tr in ctrl.tracker.tracks:
            if tr.locked:
                locked_tracks_seen.add(tr.track_id)

        if prev_phase != "GRASP" and info["phase"] == "GRASP":
            catch_events.append(
                {
                    "t": round(t_s, 2),
                    "track_id": info["active_track_id"],
                    "vy_mps": round(info["est_vy_mps"], 3),
                    "p_des": [round(v, 3) for v in info["p_des"]],
                }
            )
        prev_phase = info["phase"]

        if out_video is not None and (idx // max(1, stride)) % 2 == 0:
            frames_out.append(draw_debug_overlay(img_rgb, ctrl, t_s, info))

    if out_video is not None and frames_out:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames_out[0].shape[:2]
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{w}x{h}",
            "-pix_fmt",
            "rgb24",
            "-r",
            "15",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "24",
            str(out_video),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert proc.stdin is not None
        for fr in frames_out:
            proc.stdin.write(fr.tobytes())
        proc.stdin.close()
        proc.wait()

    return {
        "run": data.get("name", pickle_path.stem),
        "frames_processed": len(ik_errs),
        "locked_tracks_total": len(locked_tracks_seen),
        "simulated_catches": ctrl.catches,
        "catch_events": catch_events,
        "median_ik_err_mm": float(np.median(ik_errs)) if ik_errs else 0.0,
        "max_ik_err_mm": max_ik_err_mm,
        "phase_counts": phase_counts,
        "out_video": str(out_video) if out_video else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YAM holesaw CV conveyor controller rollout & offline replay")
    p.add_argument("--replay-pickle", type=Path, default=None, help="Offline replay from cached RRD pickle")
    p.add_argument("--out-video", type=Path, default=None, help="Output annotated MP4 for offline replay")
    p.add_argument("--stride", type=int, default=2, help="Frame stride during offline replay (default 2 -> ~15 Hz)")
    p.add_argument("--config-path", type=str, default="examples/yam/configs/pi05_holesaw_conveyor_left.yaml")
    p.add_argument("--right-config-path", type=str, default="examples/yam/configs/yam_deployment_right.yaml")
    p.add_argument("--active-arm-side", type=str, default="left", choices=("left", "right"))
    p.add_argument("--calib-path", type=str, default=None)
    p.add_argument("--calibrate", action="store_true")
    p.add_argument("--target-count", type=int, default=3, help="Stop after N catches (0 = continuous)")
    p.add_argument("--control-hz", type=float, default=60.0)
    p.add_argument("--seconds", type=float, default=120.0)
    p.add_argument("--max-joint-step", type=float, default=0.12)
    p.add_argument(
        "--pd-lag-sec",
        type=float,
        default=0.035,
        help="Extra prediction lead for physical arm tracking lag (seconds; default 0.035)",
    )
    p.add_argument("--no-verify-grasp", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    if args.replay_pickle is not None:
        summary = run_offline_replay(
            args.replay_pickle,
            calib_path=Path(args.calib_path) if args.calib_path else None,
            out_video=args.out_video,
            target_count=args.target_count,
            stride=args.stride,
        )
    else:
        summary = run_live_rollout(args)
    print(summary)


if __name__ == "__main__":
    main()
