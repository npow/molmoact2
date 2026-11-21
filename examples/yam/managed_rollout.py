import argparse
import csv
from contextlib import nullcontext
from datetime import datetime
import json
import math
import runpy
import sys
import threading
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np

from orchestral import Orchestral
from orchestral.core.tailnet import ManagedTailnetNode, TailscaleCli

DEFAULT_DEPLOYMENT = "steady-ape-3roj"
LEFT_CONFIG = "examples/yam/configs/pi05_holesaw_conveyor_left.yaml"
RIGHT_CONFIG = "examples/yam/configs/yam_deployment_right.yaml"
# Allow MolmoAct2 to overshoot aperture endpoints before the rollout clips to [0, 1].
MOLMOACT2_GRIPPER_TOLERANCE = 0.15

original_init = ManagedTailnetNode.__init__
original_run = TailscaleCli.run

def initialize_node(self, *args, **kwargs):
    kwargs["join_timeout_s"] = 120.0
    original_init(self, *args, **kwargs)

def run_tailscale(self, *args, timeout=10.0):
    if args and args[0] == "up":
        timeout = 90.0
    return original_run(self, *args, timeout=timeout)

ManagedTailnetNode.__init__ = initialize_node
TailscaleCli.run = run_tailscale

def policy_family(model_id):
    """Select a YAM action adapter using the registered model ID."""
    name = model_id.lower()
    for prefix, family in (("pi05", "pi05"), ("gr00t", "gr00t"),
                           ("molmoact2", "molmoact2"), ("dm05", "dexmal"),
                           ("praxis1", "praxis1"), ("verne", "verne"), ("isaac05", "isaac05")):
        if name == prefix or name.startswith(prefix + "-"):
            return family
    return None


def placeholder_state(family, dimension):
    if family == "praxis1":
        return np.array([0.3, 0, 0.3, 1, 0, 0, 0, 0.5], dtype=np.float32)
    if family == "verne":
        return np.array([0.3, 0, 0.3, 0, 0, 0, 0.5], dtype=np.float32)
    state = np.zeros(dimension, dtype=np.float32)
    state[0], state[6] = np.pi, 0.5
    if dimension == 14:
        state[13] = 0.5
    return state


def probe(host, family, instruction):
    import numpy as np

    if family == "pi05":
        if not host.supports_committed_actions():
            raise SystemExit("The pi0.5 Worker did not confirm committed-actions support.")
        print("Staging pi0.5 session opened and RTC support confirmed.")
        return

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    state = placeholder_state(family, host.identity["state_dim"])
    camera_mapping = host.identity["camera_mapping"]
    prediction = host.act(dict.fromkeys(camera_mapping, image), state, instruction)
    actions = np.asarray(prediction["actions"])
    expected = (host.identity["action_horizon"], host.identity["action_dim"])
    if actions.shape != expected or not np.isfinite(actions).all():
        raise SystemExit(f"{family} did not return a finite action chunk of shape {expected}.")
    host.begin_episode()
    print(f"Staging {family} returned a finite {expected[0]} × {expected[1]} action chunk using "
          f"{', '.join(camera_mapping.values())}.", flush=True)


def rollout_joint_policy(host, identity, args):
    import rerun as rr
    from omegaconf import OmegaConf
    from launch_yaml_eval_molmoact import (
        _build_env,
        resolve_bimanual_execution_mask,
        validate_bimanual_model_arm_order,
    )

    rollout = runpy.run_path(str(Path(__file__).with_name("groot_rollout.py")))
    family = args.policy_family
    checked_shape = (identity["action_horizon"], identity["action_dim"])
    state_width, action_width = {"praxis1": (8, 13), "verne": (7, 10)}.get(family, (7, 7))
    if identity["state_dim"] != state_width or checked_shape[1] != action_width or checked_shape[0] <= 0:
        raise SystemExit(f"Unexpected {family} contract for YAM: {checked_shape}")
    plan_rows = 4 if family == "verne" else checked_shape[0]
    if args.prefetch_lead_steps >= plan_rows:
        raise SystemExit(f"--prefetch-lead-steps must be below the rollout plan length ({plan_rows})")
    pose_actions = None
    if family in {"praxis1", "verne"}:
        from tool_pose_actions import ToolPoseActions

        pose_actions = ToolPoseActions(
            model_path=Path(__file__).with_name("configs") / "yam_holesaw_kinematics.urdf",
            base_body="base", tool_body="gripper",
            joint_names=[f"joint{i}" for i in range(1, 7)], max_joint_step_rad=0.12,
        )
    left = OmegaConf.to_container(OmegaConf.load(LEFT_CONFIG), resolve=True)
    right = OmegaConf.to_container(OmegaConf.load(RIGHT_CONFIG), resolve=True)
    validate_bimanual_model_arm_order(left, right)
    mask = resolve_bimanual_execution_mask(
        bimanual=True, active_arm_side="left",
        execution_mode="shadow" if args.shadow else "active_arm_hold",
        both_arm_max_delta=([0.05] * 6 + [0.025]) * 2,
    )
    rig_cameras = {"top": "front_camera", "left": "left_camera", "right": "right_camera"}
    camera_mapping = identity["camera_mapping"]
    bindings = {name: f"{rig_cameras[role]}_rgb" for role, name in camera_mapping.items()}
    for role in rig_cameras.keys() - camera_mapping.keys():
        left["sensors"]["cameras"][rig_cameras[role]]["enabled"] = False
    input(f"{family} left-arm rollout at {args.control_hz:g} Hz. Instruction: {args.instruction!r}. "
          "Clear the workspace, then press Enter to start (Ctrl+C stops)... ")
    output_path = (
        args.rrd.expanduser().resolve() if args.rrd else
        Path("yam_eval_runs/rerun") / f"{args.deployment}_{datetime.now().astimezone():%Y%m%d_%H%M%S}.rrd"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sensor_roles = {
        "top": "front_camera_rgb",
        "left": "left_camera_rgb",
        "right": "right_camera_rgb",
    }
    camera_paths = {
        sensor_role: model_camera
        for model_camera, rig_camera in bindings.items()
        for sensor_role, configured_rig_camera in sensor_roles.items()
        if rig_camera == configured_rig_camera
    }
    frame_counts = {name: 0 for name in bindings}
    count_lock = threading.Lock()
    recording_start = time.monotonic()
    recording = rr.RecordingStream(f"yam_{family}_rollout")
    print(f"[rerun] Recording every captured camera frame to {output_path}", flush=True)
    try:
        with recording:
            recording.save(output_path, default_blueprint=rollout["record_blueprint"](bindings))
            recording.log(
                "/task/instruction",
                rr.TextDocument(f"# Task instruction\n\n{args.instruction}", media_type="text/markdown"),
                static=True,
            )
            with TemporaryDirectory(prefix="holesaw-rollout-") as config_dir:
                left_config = Path(config_dir) / "left.yaml"
                OmegaConf.save(OmegaConf.create(left), left_config)
                env, *_ = _build_env(
                    SimpleNamespace(
                        config_path=left_config, right_config_path=RIGHT_CONFIG,
                        control_hz=args.control_hz, max_steps=None,
                    ),
                    camera_frame_callback=lambda camera, frame, timestamp: rollout["record_camera_frame"](
                        recording, camera_paths, frame_counts, count_lock, recording_start,
                        camera, frame, timestamp,
                    ),
                )
                try:
                    rollout["run"](
                        env, host.session, mask, seconds=args.seconds, instruction=args.instruction,
                        shadow=args.shadow, lead_steps=args.prefetch_lead_steps,
                        camera_bindings=bindings, history_offsets=(), control_hz=args.control_hz,
                        action_shape=checked_shape,
                        model_label=family, pose_actions=pose_actions, matrix_policy=family == "praxis1",
                        timestamp_observations=family == "verne", max_plan_rows=plan_rows,
                        gripper_tolerance=MOLMOACT2_GRIPPER_TOLERANCE if family == "molmoact2" else 0.05,
                    )
                finally:
                    env.close()
    finally:
        recording.disconnect()
    print(f"[rerun] Saved {', '.join(f'{name}={count}' for name, count in frame_counts.items())} frames to {output_path}",
          flush=True)


def soak_endpoint(host, identity, args):
    """Continuously query a YAM endpoint from live cameras without a robot."""
    import rerun as rr
    from omegaconf import OmegaConf
    from gello_min.v4l2_camera import V4L2Camera

    rollout = runpy.run_path(str(Path(__file__).with_name("groot_rollout.py")))
    config = OmegaConf.to_container(OmegaConf.load(LEFT_CONFIG), resolve=True)
    rig_cameras = {"top": "front_camera", "left": "left_camera", "right": "right_camera"}
    camera_mapping = identity["camera_mapping"]
    camera_configs = config["sensors"]["cameras"]
    duration = args.soak_hours * 3600.0
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    model_label = args.policy_family
    output_path = args.rrd.expanduser().resolve() if args.rrd else None
    metrics_path = (
        args.metrics.expanduser().resolve() if args.metrics else
        Path("yam_eval_runs/soak") / f"{model_label}_soak_{stamp}.metrics.csv"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = (
        output_path.with_suffix(".summary.json")
        if output_path is not None else metrics_path.with_suffix(".summary.json")
    )
    frame_counts = {name: 0 for name in camera_mapping.values()}
    count_lock = threading.Lock()
    cameras = {}
    latencies_ms = []
    server_ms = []
    session_ms = []
    outer_overhead_ms = []
    stage_values = {
        name: [] for name in (
            "encode_ms", "network_ms", "worker_decode_ms", "worker_queue_ms",
            "model_ms", "worker_overhead_ms", "client_decode_ms", "stage_sum_ms",
            "stage_residual_ms",
        )
    }
    calls = errors = camera_errors = 0
    interrupted = False
    state = placeholder_state(args.policy_family, identity["state_dim"])
    recording_start = time.monotonic()
    soak_started_at = None
    recording = rr.RecordingStream(f"yam_{model_label}_live_camera_soak") if output_path is not None else None
    print(f"[soak] {model_label} live camera input, no robot/CAN opened; fixed placeholder state, {args.soak_hours:g} h.",
          flush=True)
    if recording is None:
        print(f"[soak] RRD disabled. Metrics: {metrics_path}", flush=True)
    else:
        print(f"[soak] RRD: {output_path}\n[soak] Metrics: {metrics_path}", flush=True)
    try:
        with (recording if recording is not None else nullcontext()):
            if recording is not None:
                recording.save(
                    output_path,
                    default_blueprint=rollout["record_blueprint"](
                        {name: name for name in camera_mapping.values()}
                    ),
                )
                recording.log(
                    "/task/soak",
                    rr.TextDocument(
                        f"# {model_label} endpoint soak\n\n"
                        f"Deployment: `{args.deployment}`\n\n"
                        f"Duration: {args.soak_hours:g} hours. Live cameras; fixed placeholder robot state. "
                        "No robot or CAN interface is opened.",
                        media_type="text/markdown",
                    ),
                    static=True,
                )
            try:
                for role, model_name in camera_mapping.items():
                    camera_key = rig_cameras[role]
                    camera_config = camera_configs[camera_key]
                    if not camera_config.get("enabled", True):
                        raise RuntimeError(f"Required live camera {camera_key} is disabled in {LEFT_CONFIG}")
                    cameras[role] = V4L2Camera(
                        str(camera_config["device_id"]),
                        width=int(camera_config.get("width", 640)),
                        height=int(camera_config.get("height", 480)),
                        fps=int(camera_config.get("fps", 30)),
                        frame_callback=(
                            lambda frame, timestamp, role=role, model_name=model_name:
                                rollout["record_camera_frame"](
                                    recording, {role: model_name}, frame_counts, count_lock,
                                    recording_start, role, frame, timestamp,
                                )
                        ) if recording is not None else None,
                    )
                for role, camera in cameras.items():
                    camera.read()
                    print(f"[soak] Live {role} camera ready.", flush=True)

                started_at = time.monotonic()
                soak_started_at = started_at
                deadline = started_at + duration
                last_report = started_at
                print("[soak] Starting back-to-back predictions; Ctrl+C stops early.", flush=True)
                with metrics_path.open("w", newline="", encoding="utf-8") as metrics_file:
                    writer = csv.writer(metrics_file)
                    writer.writerow((
                        "call", "elapsed_sec", "result", "client_ms", "session_total_ms",
                        "encode_ms", "network_ms", "worker_decode_ms", "worker_queue_ms",
                        "model_ms", "worker_overhead_ms", "client_decode_ms", "stage_sum_ms",
                        "stage_residual_ms", "outer_overhead_ms", "trace_id", "error",
                    ))
                    while time.monotonic() < deadline:
                        call_started = time.monotonic()
                        try:
                            images = {role: camera.read()[0] for role, camera in cameras.items()}
                            result = host.act(images, state, instruction=args.instruction)
                            elapsed_ms = (time.monotonic() - call_started) * 1000.0
                            telemetry = result.get("telemetry") or {}
                            server_latency = telemetry.get("server_infer_ms")
                            to_ms = lambda key: float(telemetry.get(key, 0)) / 1000.0
                            encode = to_ms("encode_us")
                            network = to_ms("net_wire_us")
                            worker_decode = to_ms("decode_us")
                            worker_queue = to_ms("queue_us")
                            model = to_ms("accel_exec_us")
                            session_total = to_ms("total_rtt_us")
                            worker_wall_us = int(telemetry.get("worker_wall_us", 0) or 0)
                            worker_overhead = (
                                max(0.0, worker_wall_us / 1000.0 - worker_decode - worker_queue - model)
                                if worker_wall_us else 0.0
                            )
                            worker_total = worker_wall_us / 1000.0 if worker_wall_us else (
                                worker_decode + worker_queue + model
                            )
                            client_decode = max(0.0, session_total - encode - network - worker_total)
                            stage_sum = (
                                encode + network + worker_decode + worker_queue + model
                                + worker_overhead + client_decode
                            )
                            stage_residual = session_total - stage_sum
                            outer_overhead = elapsed_ms - session_total
                            calls += 1
                            latencies_ms.append(elapsed_ms)
                            session_ms.append(session_total)
                            outer_overhead_ms.append(outer_overhead)
                            row_stages = {
                                "encode_ms": encode,
                                "network_ms": network,
                                "worker_decode_ms": worker_decode,
                                "worker_queue_ms": worker_queue,
                                "model_ms": model,
                                "worker_overhead_ms": worker_overhead,
                                "client_decode_ms": client_decode,
                                "stage_sum_ms": stage_sum,
                                "stage_residual_ms": stage_residual,
                            }
                            for name, value in row_stages.items():
                                stage_values[name].append(value)
                            if isinstance(server_latency, (int, float)) and math.isfinite(server_latency):
                                server_ms.append(float(server_latency))
                            writer.writerow((
                                calls, time.monotonic() - started_at, "ok", f"{elapsed_ms:.3f}",
                                f"{session_total:.3f}", f"{encode:.3f}", f"{network:.3f}",
                                f"{worker_decode:.3f}", f"{worker_queue:.3f}", f"{model:.3f}",
                                f"{worker_overhead:.3f}", f"{client_decode:.3f}", f"{stage_sum:.3f}",
                                f"{stage_residual:.3f}", f"{outer_overhead:.3f}",
                                telemetry.get("trace_id", ""), "",
                            ))
                            metrics_file.flush()
                            if recording is not None:
                                recording.set_time("elapsed", duration=time.monotonic() - recording_start)
                                recording.log("/soak/client_latency_ms", rr.Scalars(elapsed_ms))
                                if isinstance(server_latency, (int, float)) and math.isfinite(server_latency):
                                    recording.log("/soak/server_infer_ms", rr.Scalars(float(server_latency)))
                        except Exception as exc:  # noqa: BLE001 -- count and continue the soak
                            errors += 1
                            if "camera" in str(exc).lower() or "frame" in str(exc).lower():
                                camera_errors += 1
                            writer.writerow((
                                calls + errors, time.monotonic() - started_at, "error",
                                *([""] * 12), "", f"{type(exc).__name__}: {exc}",
                            ))
                            metrics_file.flush()
                            print(f"[soak] Request/capture error #{errors}: {type(exc).__name__}: {exc}", flush=True)
                            time.sleep(1.0)

                        now = time.monotonic()
                        if now - last_report >= 60.0:
                            recent = latencies_ms[-100:]
                            p50 = float(np.percentile(recent, 50)) if recent else float("nan")
                            p95 = float(np.percentile(recent, 95)) if recent else float("nan")
                            print(f"[soak] elapsed={(now-started_at)/3600:.2f}h calls={calls} errors={errors} "
                                  f"recent_client_ms_p50={p50:.0f} p95={p95:.0f}", flush=True)
                            last_report = now
            except KeyboardInterrupt:
                interrupted = True
                print("[soak] Interrupted; closing cameras and finalizing artifacts.", flush=True)
            finally:
                for camera in cameras.values():
                    camera.close()
    finally:
        if recording is not None:
            recording.disconnect()

    elapsed_sec = time.monotonic() - (soak_started_at or recording_start)
    def stats(values):
        if not values:
            return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None,
                    "p99_ms": None, "max_ms": None}
        return {
            "count": len(values),
            "mean_ms": float(np.mean(values)),
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "p99_ms": float(np.percentile(values, 99)),
            "max_ms": float(np.max(values)),
        }
    summary = {
        "deployment": args.deployment,
        "model_family": args.policy_family,
        "requested_hours": args.soak_hours,
        "elapsed_sec": elapsed_sec,
        "predictions_ok": calls,
        "request_or_capture_errors": errors,
        "camera_errors": camera_errors,
        "interrupted": interrupted,
        "client_latency": stats(latencies_ms),
        "session_latency": stats(session_ms),
        "outer_overhead": stats(outer_overhead_ms),
        "per_request_stage_latency": {name: stats(values) for name, values in stage_values.items()},
        "server_infer_latency": stats(server_ms),
        "camera_frames": frame_counts if recording is not None else None,
        "rrd": str(output_path) if output_path is not None else None,
        "metrics_csv": str(metrics_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"[soak] Complete: {calls} predictions, {errors} errors in {elapsed_sec/3600:.2f} h. "
          f"Summary: {summary_path}", flush=True)


def main():
    from omegaconf import OmegaConf
    from orchestral_session_bridge import (
        TOOL_POSE_ADAPTER, TOOL_POSE_MATRIX_ADAPTER, OrchestralSessionHost,
    )

    parser = argparse.ArgumentParser(description="Run or probe a staging holesaw policy.")
    parser.add_argument("--deployment", "--orchestral-deployment", default=DEFAULT_DEPLOYMENT,
                        help=f"Staging deployment ID (default: {DEFAULT_DEPLOYMENT})")
    parser.add_argument("--probe", action="store_true", help="Test inference using synthetic observations")
    parser.add_argument("--soak-hours", type=float, default=0.0,
                        help="Run a robot-free live-camera endpoint soak for this many hours")
    target, rollout_args = parser.parse_known_args()
    if not math.isfinite(target.soak_hours) or target.soak_hours < 0:
        parser.error("--soak-hours must be finite and nonnegative")
    client = Orchestral()
    try:
        deployment = client.deployments.get(target.deployment)
        model_id = deployment.model_id
    finally:
        client._http.close()
    family = policy_family(model_id)
    if family is None:
        if target.soak_hours > 0 or target.probe:
            family = "generic"
        else:
            parser.error(f"{target.deployment} has no YAM joint-rollout adapter for model {model_id!r}")

    if family == "pi05" and not target.probe and target.soak_hours <= 0:
        sys.argv = [
            "examples/yam/rtc_rollout.py",
            "--config-path", LEFT_CONFIG,
            "--right-config-path", RIGHT_CONFIG,
            "--orchestral-deployment", target.deployment,
            "--active-arm-side", "left",
            "--control-hz", "15",
            "--seconds", "300",
            *rollout_args,
        ]
        runpy.run_path(sys.argv[0], run_name="__main__")
        return

    settings = argparse.ArgumentParser(prog=parser.prog)
    settings.add_argument("--seconds", type=float, default=300)
    settings.add_argument("--control-hz", type=float, default=15 if family in {"praxis1", "isaac05"} else 30)
    settings.add_argument("--instruction", default=OmegaConf.load(LEFT_CONFIG).storage.language_instruction)
    settings.add_argument("--prefetch-lead-steps", type=int, default=0 if family == "verne" else 8)
    settings.add_argument("--shadow", action="store_true")
    settings.add_argument("--rrd", type=Path, help="Rerun output path; opt in for soak runs")
    settings.add_argument("--metrics", type=Path, help="Soak CSV path (default: yam_eval_runs/soak/gr00t_soak_<timestamp>.metrics.csv)")
    args = settings.parse_args(rollout_args)
    args.deployment = target.deployment
    args.soak_hours = target.soak_hours
    args.policy_family = family
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        settings.error("--seconds must be positive and finite")
    if not math.isfinite(args.control_hz) or args.control_hz < 1:
        settings.error("--control-hz must be finite and at least 1")
    if not math.isfinite(args.soak_hours) or args.soak_hours < 0:
        settings.error("--soak-hours must be finite and nonnegative")
    if not 0 <= args.prefetch_lead_steps < 16:
        settings.error("--prefetch-lead-steps must be between 0 and 15")
    if family == "verne" and not (target.probe or target.soak_hours > 0 or args.shadow):
        settings.error("Verne hardware rollout is disabled after unexpected motion; use --shadow to test its policy without sending joint commands")
    adapter = {
        "praxis1": TOOL_POSE_MATRIX_ADAPTER,
        "verne": TOOL_POSE_ADAPTER,
        "pi05": "pi05_yam_red_cap" if target.soak_hours <= 0 else "identity",
    }.get(family, "identity")
    host = OrchestralSessionHost(
        deployment_id=target.deployment, instruction=args.instruction, policy_adapter=adapter,
        allow_camera_subset=family != "pi05" or target.soak_hours > 0,
    )
    try:
        identity = host.open()
        if target.soak_hours <= 0:
            probe(host, family, args.instruction)
        if not target.probe:
            if args.soak_hours > 0:
                soak_endpoint(host, identity, args)
            else:
                rollout_joint_policy(host, identity, args)
    finally:
        host.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
