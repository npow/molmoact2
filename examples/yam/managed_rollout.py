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

from chunk_fusion import MODES as CHUNK_FUSION_MODES
from gripper_aperture import clip_predicted_aperture
from rollout_compare import record_settings
from orchestral import Orchestral
from orchestral.core.tailnet import ManagedTailnetNode, TailscaleCli

DEFAULT_DEPLOYMENT = "steady-ape-3roj"
LEFT_CONFIG = "examples/yam/configs/pi05_holesaw_conveyor_left.yaml"
RIGHT_CONFIG = "examples/yam/configs/yam_deployment_right.yaml"
# Isaac 0.5 YAM checkpoints keep the release's bimanual layout: left arm in channels 0-6,
# right arm in 7-13. The Holesaw fine-tune saw the parked right arm as this constant
# (joints 0, gripper open), so the rollout sends exactly it and uses only the left columns.
ISAAC_PARKED_RIGHT_ARM = np.array([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
ISAAC_BIMANUAL_CONTROL_HZ = 30  # the release checkpoint's rate; one action per 30 Hz step
ISAAC_BIMANUAL_PLAN_ROWS = 30   # replan every second instead of running the full 2 s chunk
# Committed actions (RTC) are on for every family except these. TEMPORARY EXCEPTION: their Orchestral
# plugins do not support committed actions yet, so they default to off and opt in with
# --committed-actions. Remove a family from this set when its plugin supports them.
COMMITTED_ACTIONS_DEFAULT_OFF = frozenset({"rho", "dreamzero", "fastwam", "dw05", "aw0"})
# Families fine-tuned on the Holesaw task. None has a noise-seed path.
HOLESAW_INSTRUCTION = "Pick up the hole saws from the moving conveyor belt and place them in the red case"
HOLESAW_FINE_TUNED_FAMILIES = frozenset({"rho", "dreamzero", "fastwam", "dw05", "aw0"})
# Checkpoints trained at 30 Hz; every other family defaults to 15 Hz.
CONTROL_HZ_30_FAMILIES = frozenset({"flux3-action", "longwam", "rho", "aw0"})
# AW-0 keeps the RoboTwin two-arm layout: our arm is in dims 0..6 and the other arm is the constant
# the checkpoint was trained with. Only the left seven action columns ever reach the robot.
PARKED_RIGHT_ARM_FAMILIES = frozenset({"isaac05", "aw0"})
# Rows of each chunk the rollout executes before replanning. Upstream DW05 runs 8 of its 32 rows.
DEFAULT_PLAN_ROWS = {"verne": 4, "dw05": 8}
NOISE_SEED_FAMILIES = frozenset({"pi05", "flux3-action", "longwam", "praxis1"})
# The rig's arm swap belongs to this checkpoint, not to every bimanual Praxis model.
PRAXIS_HOLESAW_BIMANUAL_CHECKPOINT = (
    "file:///cache/private-checkpoints/praxis1-yam-holesaw-15hz-full-step001155"
)

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

# (model ID prefix, family). Every family here always uses committed actions.
FAMILY_PREFIXES = (("pi05", "pi05"), ("gr00t", "gr00t"),
                   ("molmoact2", "molmoact2"), ("dm05", "dexmal"),
                   ("praxis1", "praxis1"), ("verne", "verne"), ("isaac05", "isaac05"),
                   ("flux3-action", "flux3-action"), ("longwam", "longwam"), ("long-wam", "longwam"),
                   ("rho", "rho"), ("dreamzero", "dreamzero"), ("fastwam", "fastwam"),
                   ("dw05", "dw05"), ("aw0", "aw0"))


def policy_family(model_id):
    """Select a YAM action adapter using the registered model ID."""
    name = model_id.lower()
    for prefix, family in FAMILY_PREFIXES:
        if name == prefix or name.startswith(prefix + "-"):
            return family
    return None


def placeholder_state(family, dimension):
    if family == "praxis1":
        if dimension == 16:
            from tool_pose_actions import HolesawBimanualActions

            return np.r_[HolesawBimanualActions.parked_state,
                         np.array([0.7, -0.23, 0.38, 1, 0, 0, 0, 0.5], dtype=np.float32)]
        return np.array([0.3, 0, 0.3, 1, 0, 0, 0, 0.5], dtype=np.float32)
    if family == "verne":
        return np.array([0.3, 0, 0.3, 0, 0, 0, 0.5], dtype=np.float32)
    if family == "rho":
        return np.array([0.3, 0, 0.3, 0, 0, 0, 1, 0.5], dtype=np.float32)
    state = np.zeros(dimension, dtype=np.float32)
    state[0], state[6] = np.pi, 0.5
    if dimension == 14:
        state[13] = 0.5
        if family == "aw0":
            state[7:] = ISAAC_PARKED_RIGHT_ARM
    return state


def committed_contract(family, identity):
    """What a Worker must accept as ``committed_actions`` for this family, for the refusal message."""
    width, horizon = identity["action_dim"], identity["action_horizon"]
    space = {
        "rho": "absolute end-effector rows [x y z qx qy qz qw gripper] in the rho_tool frame (the commands the "
               "arm will run, as forward kinematics), gripper 1 = open",
        "aw0": "absolute 14-value joint rows: this arm in 0..6 and the parked right arm [0,0,0,0,0,0,1] in 7..13",
        "praxis1": "absolute tool-pose rows in the layout the Worker returns",
        "verne": "absolute tool-pose rows in the layout the Worker returns",
    }.get(family, "absolute rows in the same space and width as the returned actions")
    return (f"a committed_actions input of shape (k, {width}), 1 <= k < {horizon}, holding {space}, "
            f"and a returned chunk whose first k rows equal it exactly")


def require_committed_actions(host, family):
    """Committed actions (RTC) are mandatory: a Worker without them is refused before any motion."""
    if not host.supports_committed_actions():
        model = host.identity.get("model_ref") or family
        raise SystemExit(
            f"The {family} Worker ({model}) does not support committed actions, which every Worker must. "
            f"Its plugin must implement {committed_contract(family, host.identity)}."
        )


def committed_enabled(family, args):
    """Committed actions are on unless the family is a temporary exception and the user did not opt in."""
    return family not in COMMITTED_ACTIONS_DEFAULT_OFF or bool(getattr(args, "committed_actions", False))


def probe(host, family, instruction, *, committed_actions=True, noise_seed=None):
    if committed_actions:
        require_committed_actions(host, family)
    if family == "pi05":
        if noise_seed is None:
            print("Staging pi0.5 session opened and RTC support confirmed.")
            return
    if family == "longwam" and not getattr(host.session, "supports_observe", False):
        raise SystemExit("Long-WAM requires a Worker and SDK that support observation-only requests.")

    state = placeholder_state(family, host.identity["state_dim"])
    camera_mapping = host.identity["camera_mapping"]
    spec = host.session.grant.io_spec
    cameras = {name: np.zeros((*shape, 3), np.uint8) for name, shape in spec.cameras.items()}
    inputs = {"state": state}
    if "history_rows" in host.identity:
        from temporal_joint_history import TemporalJointHistory

        history = TemporalJointHistory(host.identity["history_rows"])
        history.observe(cameras, state)
        cameras, inputs = history.inputs()
    elif family in PARKED_RIGHT_ARM_FAMILIES and len(state) == 14:
        state[7:] = ISAAC_PARKED_RIGHT_ARM

    predict = dict(cameras=cameras, instruction=instruction, **inputs)
    noise = {} if noise_seed is None else {"noise_seed": noise_seed}
    actions = np.asarray(host.session.predict(**predict, **noise, episode_start=True).actions, dtype=np.float32)
    expected = (host.identity["action_horizon"], host.identity["action_dim"])
    if actions.shape != expected or not np.isfinite(actions).all():
        raise SystemExit(f"{family} did not return a finite action chunk of shape {expected}.")
    if family == "flux3-action":
        clip_predicted_aperture(actions[:, 6], family)
    if family == "longwam":
        host.session.observe(**predict)
    if not committed_actions:
        host.begin_episode()
        print(f"Staging {family} returned a finite {expected[0]} × {expected[1]} action chunk "
              f"using {', '.join(camera_mapping.values())}.", flush=True)
        return
    prefix = actions[:1].copy()
    if family in {"rho", "verne"}:  # pose rows end with the gripper (praxis1's inverted one is left as is)
        prefix[:, -1] = np.clip(prefix[:, -1], 0, 1)
    elif family != "praxis1":
        prefix[:, 6] = np.clip(prefix[:, 6], 0, 1)
    if family in PARKED_RIGHT_ARM_FAMILIES and expected[1] == 14:
        prefix[:, 7:] = ISAAC_PARKED_RIGHT_ARM
    continued = np.asarray(
        host.session.predict(**predict, **noise, committed_actions=prefix).actions, dtype=np.float32,
    )
    if continued.shape != expected or not np.isfinite(continued).all() or not np.array_equal(continued[:1], prefix):
        raise SystemExit(f"{family} did not return a finite chunk preserving the committed action.")
    if family == "flux3-action":
        clip_predicted_aperture(continued[:, 6], family)
    host.begin_episode()
    status = " and preserved a committed row"
    print(f"Staging {family} returned a finite {expected[0]} × {expected[1]} action chunk "
          f"using {', '.join(camera_mapping.values())}{status}.", flush=True)


def rollout_joint_policy(host, identity, args):
    import rerun as rr
    from omegaconf import OmegaConf
    from launch_yaml_eval_molmoact import (
        _build_env,
        resolve_bimanual_execution_mask,
        validate_bimanual_model_arm_order,
    )

    rollout = runpy.run_path(str(Path(__file__).with_name("rollout_loop.py")))
    family = args.policy_family
    checked_shape = (identity["action_horizon"], identity["action_dim"])
    bimanual_layout = family in PARKED_RIGHT_ARM_FAMILIES and (identity["state_dim"], checked_shape[1]) == (14, 14)
    praxis_bimanual = family == "praxis1" and identity["state_dim"] == 16
    state_width, action_width = {"praxis1": (16, 26) if praxis_bimanual else (8, 13), "verne": (7, 10), "rho": (8, 8),
                                         "aw0": (14, 14)}.get(
        family, (14, 14) if bimanual_layout else (7, 7))
    if identity["state_dim"] != state_width or checked_shape[1] != action_width or checked_shape[0] <= 0:
        raise SystemExit(f"Unexpected {family} contract for YAM: {checked_shape}")
    if args.control_hz is None:
        args.control_hz = (ISAAC_BIMANUAL_CONTROL_HZ if bimanual_layout and family == "isaac05"
                           else 30 if family in CONTROL_HZ_30_FAMILIES else 15)
    plan_rows = getattr(args, "plan_rows", None) or min(DEFAULT_PLAN_ROWS.get(family, checked_shape[0]), checked_shape[0])
    if getattr(args, "plan_rows", None) is None and bimanual_layout and family == "isaac05":
        plan_rows = min(plan_rows, ISAAC_BIMANUAL_PLAN_ROWS)
    if plan_rows > checked_shape[0]:
        raise SystemExit(f"--plan-rows must not exceed the chunk length ({checked_shape[0]})")
    lead_steps = args.prefetch_lead_steps
    if lead_steps is None:
        # The default lead is about 0.5 s of queued rows: 8 at 15 Hz, 16 at 30 Hz (Rho, AW-0).
        default_lead = {"verne": 1, "flux3-action": 4, "longwam": 16,
                        "rho": 16, "aw0": 16}.get(family, 8)
        lead_steps = min(default_lead, plan_rows - 1)
    if lead_steps >= plan_rows:
        raise SystemExit(f"--prefetch-lead-steps must be below the rollout plan length ({plan_rows})")
    committed = committed_enabled(family, args)
    if committed and (args.chunk_fusion != "splice" or lead_steps < 1):
        raise SystemExit("Committed actions are on: use --chunk-fusion splice and at least one prefetch lead step.")
    temporal_history = None
    if "history_rows" in identity:
        from temporal_joint_history import TemporalJointHistory

        temporal_history = TemporalJointHistory(identity["history_rows"])
    pose_actions = None
    if family in {"praxis1", "verne", "rho"}:
        from tool_pose_actions import HolesawBimanualActions, QuatToolPoseActions, ToolPoseActions

        adapter = QuatToolPoseActions if family == "rho" else HolesawBimanualActions if praxis_bimanual else ToolPoseActions
        pose_actions = adapter(
            model_path=Path(__file__).with_name("configs") / "yam_holesaw_kinematics.urdf",
            base_body="base",
            tool_body="rho_tool" if family == "rho" else "grasp_site" if praxis_bimanual else "gripper",
            joint_names=[f"joint{i}" for i in range(1, 7)],
            # The 0.12 rad bound is per 15 Hz row; keep the same joint speed at 30 Hz.
            max_joint_step_rad=0.12 * 15 / args.control_hz if family == "rho" else 0.12,
        )
    left = OmegaConf.to_container(OmegaConf.load(LEFT_CONFIG), resolve=True)
    right = OmegaConf.to_container(OmegaConf.load(RIGHT_CONFIG), resolve=True)
    validate_bimanual_model_arm_order(left, right)
    mask = resolve_bimanual_execution_mask(
        bimanual=True, active_arm_side="left",
        execution_mode="shadow" if args.shadow else "active_arm_hold",
        # Per-tick cap against encoder feedback, sized from the Holesaw demos at 15 Hz:
        # joints reach 0.19 rad/step at p99, the gripper up to 0.67/step when closing.
        # A tighter gripper cap stops the jaws at the object and leaves no grip force.
        # The joint cap is a speed limit, so it shrinks with the tick when the rate rises.
        both_arm_max_delta=([0.2 * 15 / args.control_hz] * 6 + [1.0]) * 2,
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
    # What rollout_compare.py groups runs by when it ranks the settings tried.
    record_settings(output_path, dict(
        deployment=args.deployment, family=args.policy_family, control_hz=args.control_hz,
        chunk_fusion=args.chunk_fusion, fusion_rows=args.fusion_rows,
        prefetch_lead_steps=lead_steps, committed_actions=committed,
        execution_mode="shadow" if args.shadow else "active_arm_hold",
        seconds=args.seconds, noise_seed=args.noise_seed, instruction=args.instruction, plan_rows=plan_rows))
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
            recording.save(output_path, default_blueprint=rollout["record_blueprint"](bindings, control_trace=True))
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
                trace = None
                try:
                    trace = rollout["RolloutTrace"](recording, recording_start, host.session)
                    rollout["run"](
                        env, host.session, mask, seconds=args.seconds, instruction=args.instruction,
                        shadow=args.shadow, lead_steps=lead_steps,
                        camera_bindings=bindings, history_offsets=(), control_hz=args.control_hz,
                        action_shape=checked_shape,
                        model_label=family, pose_actions=pose_actions, matrix_policy=family == "praxis1",
                        # Isaac is conditioned on time since the episode began.
                        timestamp_observations=family in {"verne", "isaac05"}, max_plan_rows=plan_rows,
                        temporal_history=temporal_history,
                        chunk_fusion=args.chunk_fusion, fusion_rows=args.fusion_rows,
                        state_padding=ISAAC_PARKED_RIGHT_ARM if bimanual_layout else None,
                        trace=trace,
                        committed_actions=committed,
                        stream_observations=family in {"longwam", "praxis1"},
                        noise_seed=args.noise_seed,
                    )
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


def soak_endpoint(host, identity, args):
    """Continuously query a YAM endpoint from live cameras without a robot."""
    import rerun as rr
    from omegaconf import OmegaConf
    from gello_min.v4l2_camera import V4L2Camera

    rollout = runpy.run_path(str(Path(__file__).with_name("rollout_loop.py")))
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
                            result = host.act(images, state, instruction=args.instruction, noise_seed=args.noise_seed)
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
        GR00T_YAM_ADAPTER, PRAXIS_HOLESAW_BIMANUAL, SINGLE_ARM_IN_14, TEMPORAL_JOINT_ADAPTER, TOOL_POSE_ADAPTER,
        TOOL_POSE_MATRIX_ADAPTER, TOOL_POSE_QUAT_ADAPTER,
        OrchestralSessionHost, parse_noise_seed,
    )

    parser = argparse.ArgumentParser(description="Run or probe a staging holesaw policy.")
    parser.add_argument("--deployment", "--orchestral-deployment", default=DEFAULT_DEPLOYMENT,
                        help=f"Staging deployment ID (default: {DEFAULT_DEPLOYMENT})")
    parser.add_argument("--probe", action="store_true", help="Test inference using synthetic observations")
    parser.add_argument("--soak-hours", type=float, default=0.0,
                        help="Run a robot-free live-camera endpoint soak for this many hours")
    parser.add_argument("--noise-seed", type=parse_noise_seed, help="Fixed noise seed for each prediction")
    parser.add_argument("--praxis-holesaw-bimanual", action="store_true",
                        help="Select the Holesaw grasp-site rig for a checkpoint without a known layout")
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
    if target.noise_seed is not None and family not in NOISE_SEED_FAMILIES:
        parser.error(f"{family} does not support --noise-seed")
    if target.praxis_holesaw_bimanual and family != "praxis1":
        parser.error("--praxis-holesaw-bimanual requires a Praxis-1 checkpoint")

    if family == "pi05" and not target.probe and target.soak_hours <= 0:
        sys.argv = [
            "examples/yam/rtc_rollout.py",
            "--config-path", LEFT_CONFIG,
            "--right-config-path", RIGHT_CONFIG,
            "--orchestral-deployment", target.deployment,
            "--active-arm-side", "left",
            "--control-hz", "15",
            "--seconds", "300",
            *([] if target.noise_seed is None else ["--noise-seed", str(target.noise_seed)]),
            *rollout_args,
        ]
        runpy.run_path(sys.argv[0], run_name="__main__")
        return

    settings = argparse.ArgumentParser(prog=parser.prog)
    settings.add_argument("--seconds", type=float, default=300)
    # One action per control step: 15 Hz for the checkpoints trained on 15 Hz data, 30 Hz for
    # an Isaac 0.5 checkpoint in the release's bimanual layout.
    settings.add_argument("--control-hz", type=float)
    settings.add_argument("--instruction", help="default: the training instruction of the selected family")
    settings.add_argument("--committed-actions", action="store_true",
                          help="condition predictions on the queued commands; needed only for the families in "
                               "COMMITTED_ACTIONS_DEFAULT_OFF, all others always do")
    settings.add_argument("--plan-rows", type=int,
                          help="rows of each chunk to execute before replanning (default: the family's, else the whole chunk)")
    settings.add_argument("--prefetch-lead-steps", type=int)
    # How successive action chunks are joined; chunk_fusion.py describes each mode.
    settings.add_argument("--chunk-fusion", choices=CHUNK_FUSION_MODES, default="splice")
    settings.add_argument("--fusion-rows", type=int, default=4,
                          help="Ticks over which crossfade and rail move onto a new chunk")
    settings.add_argument("--shadow", action="store_true")
    settings.add_argument("--rrd", type=Path, help="Rerun output path; opt in for soak runs")
    settings.add_argument("--metrics", type=Path, help="Soak CSV path (default: yam_eval_runs/soak/gr00t_soak_<timestamp>.metrics.csv)")
    args = settings.parse_args(rollout_args)
    args.deployment = target.deployment
    args.soak_hours = target.soak_hours
    args.policy_family = family
    if args.instruction is None:
        args.instruction = (HOLESAW_INSTRUCTION if family in HOLESAW_FINE_TUNED_FAMILIES
                            else OmegaConf.load(LEFT_CONFIG).storage.language_instruction)
    if args.plan_rows is not None and args.plan_rows < 1:
        settings.error("--plan-rows must be at least 1")
    args.noise_seed = target.noise_seed
    if not math.isfinite(args.seconds) or args.seconds <= 0:
        settings.error("--seconds must be positive and finite")
    if args.control_hz is not None and (not math.isfinite(args.control_hz) or args.control_hz < 1):
        settings.error("--control-hz must be finite and at least 1")
    if not math.isfinite(args.soak_hours) or args.soak_hours < 0:
        settings.error("--soak-hours must be finite and nonnegative")
    if args.prefetch_lead_steps is not None and args.prefetch_lead_steps < 0:
        settings.error("--prefetch-lead-steps must be nonnegative")
    # Committed actions (RTC) need splice handoffs and at least one queued row.
    args.committed_actions = committed_enabled(family, args)
    if target.soak_hours <= 0 and args.committed_actions and (
            args.chunk_fusion != "splice" or args.prefetch_lead_steps == 0):
        settings.error("committed actions are on: use --chunk-fusion splice and --prefetch-lead-steps of at least 1")
    if family in COMMITTED_ACTIONS_DEFAULT_OFF and not args.committed_actions and target.soak_hours <= 0:
        print(f"Committed actions are off for {family} until its plugin supports them; "
              "pass --committed-actions to use them.", flush=True)
    if args.fusion_rows < 1:
        settings.error("--fusion-rows must be at least 1")
    if family == "verne" and not (target.probe or target.soak_hours > 0 or args.shadow):
        settings.error("Verne hardware rollout is disabled after unexpected motion; use --shadow to test its policy without sending joint commands")
    if family == "flux3-action" and target.soak_hours > 0 and not target.probe:
        settings.error("FLUX 3 Action requires robot command history; use --probe or a supervised --shadow rollout")
    adapter = {
        "gr00t": GR00T_YAM_ADAPTER,
        "flux3-action": TEMPORAL_JOINT_ADAPTER,
        "praxis1": PRAXIS_HOLESAW_BIMANUAL if (
            target.praxis_holesaw_bimanual or deployment.checkpoint_uri == PRAXIS_HOLESAW_BIMANUAL_CHECKPOINT
        ) else TOOL_POSE_MATRIX_ADAPTER,
        "verne": TOOL_POSE_ADAPTER,
        "rho": TOOL_POSE_QUAT_ADAPTER,
        "aw0": SINGLE_ARM_IN_14,
        "pi05": "pi05_yam_red_cap" if target.soak_hours <= 0 else "identity",
    }.get(family, "identity")
    host = OrchestralSessionHost(
        deployment_id=target.deployment, instruction=args.instruction, policy_adapter=adapter,
        allow_camera_subset=family != "pi05" or target.soak_hours > 0,
    )
    try:
        identity = host.open()
        if target.soak_hours <= 0:
            probe(host, family, args.instruction, committed_actions=args.committed_actions,
                  noise_seed=args.noise_seed)
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
