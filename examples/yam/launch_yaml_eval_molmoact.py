"""MolmoAct eval launcher.

Runs N rollouts, prompting for an instruction each time. Saves all three
cameras frame-by-frame (PNG) plus the joint trajectory (``episode.h5``) per
rollout, classifies rollouts via cv2 keypress (y/n/q) or a post-timeout
stdin prompt, and converts the session's labeled rollouts to a LeRobot v3.0
dataset on the way out.

CLI::

    python examples/yam/launch_yaml_eval_molmoact.py \
        --left_config_path examples/yam/configs/yam_left.yaml \
        --right_config_path examples/yam/configs/yam_right.yaml \
        -n 10
"""

from __future__ import annotations

import atexit
import logging
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import tyro
from omegaconf import OmegaConf

from camera_client import CameraClient
from chunked_rollout import run_chunked_rollout
from gello_min.realsense_camera import RealSenseCamera, get_device_ids
from gello_min.env import RobotEnv
from eval_utils import (
    EvalRolloutSaver,
    LiveCameraView,
    RolloutOutcome,
    convert_session_to_lerobot,
    move_rollout,
    prompt_instruction,
    resolve_label,
)
from gello_min.robot import BimanualRobot
from gello_min.launch_utils import instantiate_from_dict, move_to_start_position
from gello_min.logging_utils import log_collect_demos
from molmoact_client import MolmoAct, MolmoActLocal


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
DEVICE = os.environ.get("LEROBOT_TEST_DEVICE", "cuda") if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# atexit parking
# ---------------------------------------------------------------------------

_env: Optional[RobotEnv] = None
_bimanual: bool = False
_left_cfg: Optional[Dict[str, Any]] = None
_right_cfg: Optional[Dict[str, Any]] = None
_cleanup_done: bool = False


def _park_robot() -> None:
    """atexit hook: park arm back at start_joints regardless of exit path."""
    global _cleanup_done
    if _cleanup_done or _env is None:
        return
    _cleanup_done = True
    print("Parking robot at start position...")
    try:
        if _bimanual:
            move_to_start_position(_env, True, _left_cfg, _right_cfg)
        else:
            move_to_start_position(_env, False, _left_cfg)
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        logger.warning("Parking failed: %s", exc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@dataclass
class Args:
    left_config_path: str
    """Path to the left arm configuration YAML file."""

    right_config_path: Optional[str] = None
    """Path to the right arm configuration YAML file (for bimanual operation)."""

    num_rollouts: Annotated[int, tyro.conf.arg(aliases=("-n",))] = 1
    """How many rollouts to run in this session."""

    prefetch: bool = False
    """Overlap the policy round trip with arm motion instead of blocking on it.

    Off by default: this changes what the arm does at every chunk boundary and has not yet
    been validated on hardware. See ``chunked_rollout.py``. Measured in the hermetic replay
    harness, it removes the full ~135 ms boundary freeze (p95 135.1 ms -> 0.0 ms).
    """

    exec_steps: Optional[int] = None
    """Rows to execute per chunk before re-querying. ``None`` walks the whole chunk (stock).

    Lower values trade GPU duty cycle for fresher actions; only meaningful with ``prefetch``,
    since without it every re-query costs another blocking round trip.
    """

    max_jump: Optional[float] = None
    """Abort if a commanded row differs from the arm's actual pose by more than this (radians).

    No bracket has been characterised for MolmoAct2 on YAM, so the default is off and the
    loop says so at startup. Do not copy the SO-101 numbers here -- they are degrees, on a
    different arm."""


# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------


def _build_env(
    args: Args,
) -> Tuple[RobotEnv, Dict[str, Any], Optional[Dict[str, Any]], bool]:
    """Build cameras + robot(s) + RobotEnv from the launch configs.

    Camera source is decided by the ``eval.camera_server.enabled`` flag in the
    left config:

    * ``true``  -> connect to the long-lived camera server over ZMQ. RealSense
      devices are owned by that server; this process never opens them.
    * ``false`` -> open ``RealSenseCamera`` objects in-process (legacy path).
    """
    left_cfg = OmegaConf.to_container(OmegaConf.load(args.left_config_path), resolve=True)
    bimanual = args.right_config_path is not None
    right_cfg = (
        OmegaConf.to_container(OmegaConf.load(args.right_config_path), resolve=True)
        if bimanual else None
    )

    cam_server_cfg = ((left_cfg.get("eval") or {}).get("camera_server") or {})
    use_server = bool(cam_server_cfg.get("enabled", False))

    camera_dict = None
    camera_client = None
    if use_server:
        endpoint = str(cam_server_cfg.get("endpoint", "tcp://127.0.0.1:5555"))
        timeout_ms = int(cam_server_cfg.get("request_timeout_ms", 500))
        max_age = cam_server_cfg.get("max_frame_age_sec", 0.5)
        max_age = float(max_age) if max_age is not None else None
        print(f"[eval] Using camera server at {endpoint} (timeout={timeout_ms} ms)")
        camera_client = CameraClient(
            endpoint=endpoint,
            request_timeout_ms=timeout_ms,
            max_frame_age_sec=max_age,
        )
        if not camera_client.ping():
            raise RuntimeError(
                f"Camera server at {endpoint} did not respond to ping. "
                "Start it with scripts/start_camera_server.sh."
            )
    else:
        ids = get_device_ids()
        print(f"Found {len(ids)} camera devices: {ids}")
        camera_cfg = left_cfg["sensors"]["cameras"]
        camera_dict = {
            "left_camera": RealSenseCamera(camera_cfg["left_camera"]["device_id"]),
            "front_camera": RealSenseCamera(camera_cfg["front_camera"]["device_id"]),
            "right_camera": RealSenseCamera(camera_cfg["right_camera"]["device_id"]),
        }

    left_robot_cfg = left_cfg["robot"]
    if isinstance(left_robot_cfg.get("config"), str):
        left_robot_cfg["config"] = OmegaConf.to_container(
            OmegaConf.load(left_robot_cfg["config"]), resolve=True
        )
    left_robot = instantiate_from_dict(left_robot_cfg)

    if bimanual:
        right_robot_cfg = right_cfg["robot"]
        if isinstance(right_robot_cfg.get("config"), str):
            right_robot_cfg["config"] = OmegaConf.to_container(
                OmegaConf.load(right_robot_cfg["config"]), resolve=True
            )
        right_robot = instantiate_from_dict(right_robot_cfg)
        robot = BimanualRobot(left_robot, right_robot)
    else:
        robot = left_robot

    env = RobotEnv(
        robot,
        control_rate_hz=left_cfg.get("hz", 30),
        camera_dict=camera_dict,
        camera_client=camera_client,
    )
    return env, left_cfg, right_cfg, bimanual


# ---------------------------------------------------------------------------
# Inner loop
# ---------------------------------------------------------------------------


def dynamic_smoothing(env: RobotEnv, target_joints: np.ndarray) -> Dict[str, Any]:
    """Apply ``target_joints`` via sub-tick linear interpolation. Returns final obs.

    The interpolation sub-steps issue command-only ticks (no camera reads). A
    single ``get_obs()`` at the end produces the obs the caller actually
    consumes — this is what makes the rollout loop run at robot rate instead
    of camera rate.
    """
    curr_joints = env.get_robot_state()["joint_positions"]
    max_delta = float(np.abs(curr_joints - target_joints).max())
    steps = min(int(max_delta / 0.01), 100)
    if steps <= 1:
        env.step_command_only(target_joints)
    else:
        for jnt in np.linspace(curr_joints, target_joints, steps):
            env.step_command_only(jnt)
            time.sleep(0.001)
    return env.get_obs()


def run_one_rollout(
    env: RobotEnv,
    policy: MolmoAct,
    saver: EvalRolloutSaver,
    instruction: str,
    rollout_idx: int,
    num_rollouts: int,
    max_steps: int,
    live_view: LiveCameraView,
    prefetch: bool = False,
    exec_steps: Optional[int] = None,
    max_jump: Optional[float] = None,
) -> RolloutOutcome:
    """Execute one rollout and buffer per-step observations into ``saver``.

    End conditions:

    * ``cv2`` keypress ``y`` -> success (labeled)
    * ``cv2`` keypress ``n`` -> failure (labeled)
    * ``cv2`` keypress ``q`` -> quit (no label; rollout stays in ``eval/``)
    * step >= ``max_steps`` -> timeout (stdin prompt afterwards)

    Does NOT flush the saver — the caller does that so the Ctrl-C path can
    also flush the partial buffer.
    """
    # The robot bus and the camera reads are NOT thread-safe, and with prefetch enabled the
    # observation is captured on a background thread while this one is still commanding the
    # arm. Every touch of `env` goes through this lock. (The repo has already been bitten by
    # unserialized bus access: concurrent traffic corrupts Feetech packets under load.)
    env_lock = threading.Lock()

    def observe() -> Any:
        with env_lock:
            return env.get_obs()

    def infer(observation: Any) -> List[Any]:
        input_dict = policy.prepare_input(observation, instruction)
        t0 = time.time()
        actions = policy.inference(input_dict)["actions"]
        log_collect_demos(
            f"Policy inference {time.time() - t0:.3f}s ({len(actions)} actions)",
            "data_info",
        )
        return actions

    def current_joints() -> np.ndarray:
        with env_lock:
            return np.asarray(env.get_robot_state()["joint_positions"], dtype=float)

    def execute(action: np.ndarray, step: int) -> Optional[str]:
        with env_lock:
            obs_pre = env.get_obs()
            obs_post = dynamic_smoothing(env, action) or obs_pre

        saver.add_step(obs_pre=obs_pre, obs_post=obs_post)

        key = live_view.update(
            obs=obs_pre,
            rollout_idx=rollout_idx,
            num_rollouts=num_rollouts,
            step=step + 1,
            max_steps=max_steps,
            instruction=instruction,
        )
        if key in ("y", "n", "q"):
            return {"y": "success", "n": "failure", "q": "quit"}[key]
        return None

    report = run_chunked_rollout(
        observe=observe,
        infer=infer,
        execute=execute,
        current_joints=current_joints,
        max_steps=max_steps,
        exec_steps=exec_steps,
        max_jump=max_jump,
        prefetch=prefetch,
    )

    gap_p95 = report.boundary_gap_p95_ms()
    log_collect_demos(
        f"Rollout loop: prefetch={prefetch} chunks={len(report.chunks)} "
        f"steps={report.steps} boundary_gap_p95="
        f"{'n/a' if gap_p95 is None else f'{gap_p95:.1f}ms'} "
        f"late_prefetch={report.late_prefetch_rate()}",
        "data_info",
    )

    if report.aborted in ("success", "failure", "quit"):
        return RolloutOutcome(end_reason=report.aborted, last_step=report.steps)
    if report.aborted is not None:
        # A guard abort is a failure with a reason, not a timeout: say so rather than letting
        # it be recorded as a run that merely ran out of steps.
        print(f"  !! rollout aborted: {report.aborted}")
        return RolloutOutcome(end_reason="failure", last_step=report.steps)
    return RolloutOutcome(end_reason="timeout", last_step=max_steps)


# ---------------------------------------------------------------------------
# Session driver
# ---------------------------------------------------------------------------


def run_session(
    env: RobotEnv,
    policy: MolmoAct,
    left_cfg: Dict[str, Any],
    right_cfg: Optional[Dict[str, Any]],
    bimanual: bool,
    num_rollouts: int,
    prefetch: bool = False,
    exec_steps: Optional[int] = None,
    max_jump: Optional[float] = None,
) -> None:
    """Drive ``num_rollouts`` rollouts; convert the labeled set to LeRobot at the end.

    Catches ``KeyboardInterrupt`` so an in-progress rollout still gets flushed
    (as incomplete, with ``err.md``) and any rollouts already labeled in this
    session are still converted.
    """
    storage = left_cfg["storage"]
    base_save_dir = Path(storage["base_dir"]) / "data" / storage["task_directory"]
    max_steps = int(left_cfg.get("max_steps", 1000))
    last_prompt = storage.get("language_instruction") or ""

    eval_cfg = left_cfg.get("eval") or {}
    cam_srv_cfg = eval_cfg.get("camera_server") or {}
    pub_endpoint = cam_srv_cfg.get("pub_endpoint") if cam_srv_cfg.get("enabled") else None
    live_view = LiveCameraView(
        enabled=bool(eval_cfg.get("live_view_enabled", True)),
        pub_endpoint=pub_endpoint,
        recv_timeout_ms=int(cam_srv_cfg.get("recv_timeout_ms", 100)),
    )

    session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labeled_rollouts: List[Path] = []
    saver: Optional[EvalRolloutSaver] = None
    outcome: Optional[RolloutOutcome] = None

    try:
        for rollout_idx in range(num_rollouts):
            move_to_start_position(env, bimanual, left_cfg, right_cfg)
            instruction = prompt_instruction(rollout_idx, num_rollouts, last_prompt)
            last_prompt = instruction

            rollout_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rollout_dir = base_save_dir / "eval" / rollout_timestamp
            saver = EvalRolloutSaver(
                rollout_dir=rollout_dir,
                instruction=instruction,
                max_workers=int(storage.get("saver_max_workers", 2)),
                png_compress_level=int(storage.get("png_compress_level", 1)),
            )

            print(f"\n--- Rollout {rollout_idx + 1}/{num_rollouts} ---")
            print(f"  instruction: {instruction}")
            print(f"  rollout_dir: {rollout_dir}")

            outcome = run_one_rollout(
                env=env,
                policy=policy,
                saver=saver,
                instruction=instruction,
                rollout_idx=rollout_idx,
                num_rollouts=num_rollouts,
                max_steps=max_steps,
                live_view=live_view,
                prefetch=prefetch,
                exec_steps=exec_steps,
                max_jump=max_jump,
            )

            saver.flush()
            label = resolve_label(outcome)
            if label is not None:
                new_path = move_rollout(rollout_dir, label, base_save_dir)
                labeled_rollouts.append(new_path)
                print(f"  -> labeled '{label}': {new_path}")
            else:
                print(f"  -> kept in eval/: {rollout_dir}")

            saver = None
            outcome = None
    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl-C received — saving incomplete rollout, then converting...")
        if saver is not None:
            try:
                saver.flush()
                saver.write_err(
                    reason="KeyboardInterrupt",
                    step=outcome.last_step if outcome else saver.num_steps,
                )
                print(f"  -> incomplete rollout saved: {saver.rollout_dir}")
            except Exception as exc:  # noqa: BLE001 — best-effort cleanup
                logger.exception("Failed to flush incomplete rollout: %s", exc)
    finally:
        live_view.close()
        _convert_if_any(labeled_rollouts, base_save_dir, session_timestamp, left_cfg)


def _convert_if_any(
    labeled_rollouts: List[Path],
    base_save_dir: Path,
    session_timestamp: str,
    left_cfg: Dict[str, Any],
) -> None:
    """Best-effort LeRobot conversion of this session's labeled rollouts."""
    if not labeled_rollouts:
        print("\n[session] No labeled rollouts this session — nothing to convert.")
        return

    lerobot_cfg = left_cfg.get("lerobot", {}) or {}
    output_dir = base_save_dir / "eval_lerobot_v30" / session_timestamp
    print(
        f"\n[session] Converting {len(labeled_rollouts)} labeled rollouts "
        f"to LeRobot v3.0 at {output_dir} ..."
    )
    try:
        convert_session_to_lerobot(
            session_rollout_dirs=labeled_rollouts,
            output_dir=output_dir,
            fps=int(lerobot_cfg.get("fps", left_cfg.get("hz", 30))),
            robot_type=str(lerobot_cfg.get("robot_type", "molmoact_dual_arm")),
            repo_id=str(lerobot_cfg.get("hf_repo_id", "local/eval_session")),
            action_mode=str(lerobot_cfg.get("action_mode", "next_joint_fields")),
            vcodec=str(lerobot_cfg.get("vcodec", "libsvtav1")),
            sanitize_online_viz_meta=bool(lerobot_cfg.get("sanitize_online_viz_meta", True)),
            image_writer_processes=int(lerobot_cfg.get("image_writer_processes", 0)),
            image_writer_threads=int(lerobot_cfg.get("image_writer_threads", 0)),
            parallel_encoding=bool(lerobot_cfg.get("parallel_encoding", True)),
        )
    except Exception as exc:  # noqa: BLE001 — keep raw rollouts even if conversion fails
        logger.exception("LeRobot conversion failed: %s", exc)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    atexit.register(_park_robot)

    args = tyro.cli(Args)
    if args.num_rollouts < 1:
        raise SystemExit("--num_rollouts must be >= 1")

    env, left_cfg, right_cfg, bimanual = _build_env(args)

    global _env, _bimanual, _left_cfg, _right_cfg
    _env = env
    _bimanual = bimanual
    _left_cfg = left_cfg
    _right_cfg = right_cfg

    if bimanual:
        move_to_start_position(env, True, left_cfg, right_cfg)
    else:
        move_to_start_position(env, False, left_cfg)

    print(f"Launching robot: {env.robot().__class__.__name__}")
    print(f"Control loop: {left_cfg.get('hz', 30)} Hz")
    print(
        f"Rollouts this session: {args.num_rollouts}, "
        f"max_steps: {left_cfg.get('max_steps', 1000)}"
    )

    eval_cfg = left_cfg.get("eval") or {}
    mode = eval_cfg.get("mode", "server")
    if mode == "local":
        policy = MolmoActLocal(**(eval_cfg.get("local") or {}))
    elif mode == "server":
        policy = MolmoAct(server=eval_cfg.get("molmoact_server"))
    else:
        raise SystemExit(f"eval.mode must be 'server' or 'local', got {mode!r}")
    run_session(
        env=env,
        policy=policy,
        left_cfg=left_cfg,
        right_cfg=right_cfg,
        bimanual=bimanual,
        num_rollouts=args.num_rollouts,
        prefetch=args.prefetch,
        exec_steps=args.exec_steps,
        max_jump=args.max_jump,
    )


if __name__ == "__main__":
    main()
