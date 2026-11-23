"""Holesaw fine-tuned families: Rho's end-effector adapter, the 14-value single-arm layout, family defaults.

Run from the repository root (the rollout reads ``examples/yam/configs``), as the other rollout tests do.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from orchestral import IOSpec

import managed_rollout
from gripper_aperture import clip_predicted_aperture
from rollout_loop import checked_chunk
from orchestral_session_bridge import (
    SINGLE_ARM_IN_14,
    TOOL_POSE_QUAT_ADAPTER,
    OrchestralBridgeError,
    OrchestralSessionHost,
)
from tool_pose_actions import QuatToolPoseActions, UnreachableToolPose

YAM = Path(__file__).resolve().parents[1]
URDF = YAM / "configs" / "yam_holesaw_kinematics.urdf"
JOINTS = [f"joint{i}" for i in range(1, 7)]
HOLESAW = "Pick up the hole saws from the moving conveyor belt and place them in the red case"

# Rho's own conversion (rho/src/convert_to_rho_v3.py + yam_fk.py), re-implemented here with numpy only.
_FK = [
    ((0, 0, 0.068), (0, 1.5708, 3.1416), (-1, 0, 0)),
    ((-0.0455, -0.0339, -0.02), (0, 0, -3.14159), (0, 1, 0)),
    ((0, -0.0688, 0.264), (0, 0, -3.14159), (0, 1, 0)),
    ((-0.0600003, -0.0688, -0.244999), (0, 0, 0), (0, 1, 0)),
    ((-0.0405003, 0.0338995, -0.0739989), (0, 0, 0), (1, 0, 0)),
    ((0.0404996, 0, -0.0356), (3.1415926535897927, 0, 1.5707963267949736), (0, 0, 1)),
]
T_TOOL = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1.0]])
D_LEFT = np.array([-0.00304, 0.00147, 0.00349])


def rho_eef(joints):
    """Position and rotation matrix of Rho's tool frame in the arm base frame."""
    def rpy(r, p, y):
        cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
        return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                         [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                         [-sp, cp * sr, cp * cr]])

    position, rotation = np.zeros(3), np.eye(3)
    for (xyz, angles, axis), q in zip(_FK, joints[:6]):
        a = np.asarray(axis, float)
        k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        position = position + rotation @ np.asarray(xyz)
        rotation = rotation @ rpy(*angles) @ (np.eye(3) + np.sin(q) * k + (1 - np.cos(q)) * (k @ k))
    return position + rotation @ D_LEFT, rotation @ T_TOOL


def quat_matrix(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def angle_deg(a, b):
    return np.degrees(np.arccos(np.clip((np.trace(a @ b.T) - 1) / 2, -1, 1)))


def adapter(max_step=0.06, **kwargs):
    return QuatToolPoseActions(model_path=URDF, base_body="base", tool_body="rho_tool", joint_names=JOINTS,
                               max_joint_step_rad=max_step, **kwargs)


# Realistic Holesaw-range configurations: [j1..j6, gripper].
CONFIGS = np.array([
    [-0.03, 2.15, 1.62, -0.82, -0.35, -1.55, 0.00],
    [-0.90, 1.83, 1.47, -1.23, 0.09, 0.87, 0.04],
    [0.05, 0.90, 0.80, -0.60, 0.20, 0.30, 0.90],
    [-0.60, 1.40, 2.10, -1.10, -0.20, -0.70, 0.50],
    [0.10, 2.40, 2.00, -0.30, 0.30, 1.30, 1.00],
])


def test_state_is_rhos_tool_frame_and_the_gripper_is_not_inverted():
    ik = adapter()
    for joints in CONFIGS:
        state = ik.state(joints)
        assert state.shape == (8,) and state.dtype == np.float32
        position, rotation = rho_eef(joints)
        np.testing.assert_allclose(state[:3], position, atol=2e-6)
        assert angle_deg(quat_matrix(state[3:7].astype(float)), rotation) < 1e-3
        assert state[6] >= 0  # canonical hemisphere
        assert state[7] == pytest.approx(joints[6], abs=1e-6)


def test_tool_frame_matches_the_public_busybox_reference_without_a_base_offset():
    fixture = json.loads((Path(__file__).parent / "fixtures" / "rho_yambox_reference.json").read_text())
    ik, position_mm, rotation_deg = adapter(), [], []
    for row in fixture["rows"]:
        state = ik.state(np.r_[row["joints"][:6], np.clip(row["joints"][6], 0, 1)])
        position_mm.append(1000 * np.linalg.norm(state[:3] - row["eef"][:3]))
        rotation_deg.append(angle_deg(quat_matrix(state[3:7].astype(float)), quat_matrix(np.array(row["eef"][3:7]))))
    # The tool transform was fitted to this data with no base offset; a wrong base frame would be a bias, not noise.
    assert np.median(position_mm) < 1.0 and max(position_mm) < 3.5
    assert max(rotation_deg) < 0.05


def test_round_trip_recovers_the_joints_from_a_warm_start():
    ik, rng = adapter(), np.random.default_rng(0)
    for joints in CONFIGS:
        row = ik.state(joints)
        start = joints.copy()
        start[:6] += rng.uniform(-0.05, 0.05, 6)
        solved, = ik.actions([row], start)
        np.testing.assert_allclose(solved[:6], joints[:6], atol=1e-3)
        assert solved[6] == pytest.approx(joints[6], abs=1e-6)


def test_a_chunk_is_solved_row_by_row_with_a_bounded_step_and_clipped_gripper():
    ik = adapter()
    start = CONFIGS[0]
    goal = CONFIGS[0] + np.array([0.3, -0.2, 0.1, 0.2, 0.25, -0.2, 0.0])
    path = np.linspace(start, goal, 32)
    rows = np.stack([ik.state(joint) for joint in path]).astype(np.float64)
    rows[:, 7] = np.linspace(0.3, 1.2, 32)  # an overshooting gripper is clipped, never inverted
    rows[3, 7] = -0.1
    solved = ik.actions(rows, start)
    assert solved.shape == (32, 7)
    np.testing.assert_allclose(solved[:, :6], path[:, :6], atol=1e-3)
    assert np.max(np.abs(np.diff(np.vstack([start[:6], solved[:, :6]]), axis=0))) <= 0.06 + 1e-6
    np.testing.assert_allclose(solved[:, 6], np.clip(rows[:, 7], 0, 1), atol=1e-6)


def test_the_first_row_may_lead_the_measured_joints_but_later_rows_stay_step_bounded():
    # In the 30 Hz Holesaw data the first action row is more than 0.06 rad from the measured joints in 59% of frames
    # (actions are leader commands), so row 0 must not be held to the per-row step bound.
    ik = adapter()
    measured = CONFIGS[0]
    lead = np.array([0.3, -0.2, 0.1, 0.2, 0.25, -0.2, 0.0])
    path = np.linspace(measured + lead, measured + lead + np.array([0.2, 0.1, -0.1, 0.05, 0.1, 0.1, 0.0]), 32)
    rows = np.stack([ik.state(joint) for joint in path]).astype(np.float64)
    solved = ik.actions(rows, measured)
    assert solved.shape == (32, 7)
    np.testing.assert_allclose(solved[:, :6], path[:, :6], atol=1e-3)
    assert np.max(np.abs(solved[0, :6] - measured[:6])) > 0.06
    assert np.max(np.abs(np.diff(solved[:, :6], axis=0))) <= 0.06 + 1e-6

    class StrictFirstRow(QuatToolPoseActions):
        first_row_max_step_rad = 0.06

    strict = StrictFirstRow(model_path=URDF, base_body="base", tool_body="rho_tool", joint_names=JOINTS,
                            max_joint_step_rad=0.06)
    with pytest.raises(UnreachableToolPose, match="row 0"):
        strict.actions(rows, measured)


def test_a_fixed_base_transform_moves_both_state_and_targets():
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])
    shifted = adapter(base_rotation=rotation, base_translation=[0.3, -0.3, 0.3])
    plain = adapter()
    joints = CONFIGS[1]
    np.testing.assert_allclose(shifted.state(joints)[:3], rotation @ plain.state(joints)[:3] + [0.3, -0.3, 0.3], atol=1e-6)
    solved, = shifted.actions([shifted.state(joints)], joints)
    np.testing.assert_allclose(solved[:6], joints[:6], atol=1e-3)
    with pytest.raises(ValueError, match="rotation"):
        adapter(base_rotation=np.eye(3) * 2)


def test_unreachable_and_malformed_rows():
    ik = adapter()
    start = CONFIGS[0]
    good = ik.state(start).astype(np.float64)
    far = good.copy()
    far[:3] = 10
    command, = ik.actions([good], start)
    # A later unreachable row ends the chunk; an unreachable first row is an error.
    np.testing.assert_array_equal(ik.actions([good, far], start), [command])
    with pytest.raises(UnreachableToolPose, match="unreachable"):
        ik.actions([far], start)
    for bad in (np.r_[good[:3], 0, 0, 0, 0, good[7]], np.r_[good[:3], 5 * good[3:7], good[7]], np.full(8, np.nan)):
        with pytest.raises(ValueError):
            ik.actions([bad], start)
    with pytest.raises(ValueError, match="width 8"):
        ik.actions(np.zeros((2, 10)), start)


def test_checked_chunk_drives_the_adapter_through_the_rollout_path():
    ik, start = adapter(), CONFIGS[2]
    path = np.linspace(start, start + np.array([0.2, 0.1, -0.1, 0.1, 0.1, 0.1, 0.0]), 32)
    rows = np.stack([ik.state(joint) for joint in path])
    seen = {}

    def predict(*, cameras, state, instruction, episode_start, **kwargs):
        seen.update(state=state, kwargs=kwargs)
        return SimpleNamespace(actions=rows)

    chunk, _ = checked_chunk(SimpleNamespace(predict=predict), {}, start, HOLESAW, episode_start=True,
                             action_shape=(32, 8), pose_actions=ik, model_label="rho", max_plan_rows=32)
    np.testing.assert_allclose(seen["state"], ik.state(start), atol=1e-6)
    assert seen["kwargs"] == {}  # no committed actions or noise seed
    np.testing.assert_allclose(chunk[:, :6], path[:, :6], atol=1e-3)


@pytest.mark.parametrize("names", [("observation.image.0", "observation.image.1"),
                                   ("observation.image.0", "observation.image.1", "observation.image.2")])
def test_rho_cameras_and_geometry(names):
    host = OrchestralSessionHost(deployment_id="metadata-only", policy_adapter=TOOL_POSE_QUAT_ADAPTER)
    host._configure(IOSpec(cameras={name: (256, 256) for name in reversed(names)}, state_dim=8, action_shape=(32, 8)))
    assert host.identity["camera_mapping"] == dict(zip(("top", "left", "right"), names))
    assert (host.identity["state_dim"], host.identity["action_dim"], host.identity["action_horizon"]) == (8, 8, 32)
    for state_dim, action in ((7, 7), (8, 13), (16, 16)):
        with pytest.raises(OrchestralBridgeError, match="unsupported checkpoint state/action"):
            OrchestralSessionHost(deployment_id="m", policy_adapter=TOOL_POSE_QUAT_ADAPTER)._configure(
                IOSpec(cameras={n: (8, 8) for n in names}, state_dim=state_dim, action_shape=(32, action)))
    # The numbering is only meaningful for this adapter, and a lone wrist view is never guessed.
    with pytest.raises(OrchestralBridgeError):
        OrchestralSessionHost(deployment_id="m")._configure(
            IOSpec(cameras={n: (8, 8) for n in names}, state_dim=7, action_shape=(32, 7)))
    with pytest.raises(OrchestralBridgeError, match="overhead"):
        OrchestralSessionHost(deployment_id="m", policy_adapter=TOOL_POSE_QUAT_ADAPTER)._configure(
            IOSpec(cameras={"observation.image.1": (8, 8)}, state_dim=8, action_shape=(32, 8)))


def test_aw0_single_arm_layout_geometry_and_cameras():
    spec = IOSpec(cameras={"cam_left_wrist": (8, 8), "cam_high": (8, 8)}, state_dim=14, action_shape=(32, 14))
    host = OrchestralSessionHost(deployment_id="metadata-only", policy_adapter=SINGLE_ARM_IN_14)
    host._configure(spec)
    assert host.identity["camera_mapping"] == {"top": "cam_high", "left": "cam_left_wrist"}
    assert (host.identity["state_dim"], host.identity["action_dim"]) == (14, 14)
    with pytest.raises(OrchestralBridgeError):  # cam_high is not a role word for the generic mapping
        OrchestralSessionHost(deployment_id="m")._configure(spec)
    with pytest.raises(OrchestralBridgeError, match="unsupported checkpoint state/action"):
        OrchestralSessionHost(deployment_id="m", policy_adapter=SINGLE_ARM_IN_14)._configure(
            IOSpec(cameras={"cam_high": (8, 8)}, state_dim=7, action_shape=(32, 7)))


def test_the_parked_right_arm_never_reaches_the_robot():
    parked = managed_rollout.ISAAC_PARKED_RIGHT_ARM
    measured = np.array([0.1, 1.0, 1.0, -0.5, 0.0, 0.2, 0.4])
    left = np.linspace(measured, measured + 0.1, 32)
    worker_chunk = np.concatenate([left, np.full((32, 7), 9.0)], axis=1).astype(np.float32)  # right half is junk
    seen = {}

    def predict(*, cameras, state, instruction, episode_start, **kwargs):
        seen["state"] = state
        return SimpleNamespace(actions=worker_chunk)

    chunk, _ = checked_chunk(SimpleNamespace(predict=predict), {}, measured, HOLESAW, episode_start=True,
                             action_shape=(32, 14), model_label="aw0", state_padding=parked)
    assert chunk.shape == (32, 7)
    np.testing.assert_allclose(chunk, left, atol=1e-6)
    assert not np.any(chunk == 9.0)
    np.testing.assert_array_equal(seen["state"][7:], parked)
    np.testing.assert_allclose(seen["state"][:7], measured, atol=1e-6)


@pytest.mark.parametrize("model, family", [
    ("rho", "rho"), ("rho-holesaw-ft", "rho"), ("dreamzero", "dreamzero"), ("dreamzero-yam", "dreamzero"),
    ("fastwam", "fastwam"), ("fastwam-yam-2cam", "fastwam"), ("dw05", "dw05"), ("dw05-holesaw", "dw05"),
    ("aw0", "aw0"), ("aw0-holesaw", "aw0"),
    ("pi05-holesaw", "pi05"), ("praxis1", "praxis1"), ("longwam-x", "longwam"), ("isaac05", "isaac05"),
    ("rhodium", None), ("dw055", None), ("aw01", None),
])
def test_policy_family_selection(model, family):
    assert managed_rollout.policy_family(model) == family


@pytest.mark.parametrize("family, identity, hz, lead, rows", [
    ("rho", dict(state_dim=8, action_dim=8, action_horizon=32), 30, 16, 32),
    ("aw0", dict(state_dim=14, action_dim=14, action_horizon=32), 30, 16, 32),
    ("dreamzero", dict(state_dim=7, action_dim=7, action_horizon=24), 15, 8, 24),
    ("fastwam", dict(state_dim=7, action_dim=7, action_horizon=32), 15, 8, 32),
    ("dw05", dict(state_dim=7, action_dim=7, action_horizon=32), 15, 7, 8),
    ("gr00t", dict(state_dim=7, action_dim=7, action_horizon=16), 15, 8, 16),
    ("longwam", dict(state_dim=7, action_dim=7, action_horizon=32), 30, 16, 32),
])
def test_rollout_defaults_per_family(family, identity, hz, lead, rows, tmp_path, monkeypatch):
    class SettingsSaved(Exception):
        pass

    saved = {}

    def save_settings(path, settings):
        saved.update(settings)
        raise SettingsSaved

    monkeypatch.chdir(YAM.parents[1])
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(managed_rollout, "record_settings", save_settings)
    args = SimpleNamespace(policy_family=family, control_hz=None, prefetch_lead_steps=None, plan_rows=None,
                           chunk_fusion="splice", fusion_rows=4, shadow=False,
                           instruction="go", rrd=tmp_path / "run.rrd", deployment="d", seconds=30, noise_seed=None)
    identity = dict(identity, camera_mapping={"top": "a", "left": "b"})
    with pytest.raises(SettingsSaved):
        managed_rollout.rollout_joint_policy(None, identity, args)
    assert (saved["control_hz"], saved["prefetch_lead_steps"], saved["plan_rows"]) == (hz, lead, rows)
    assert saved["committed_actions"] is (family not in managed_rollout.COMMITTED_ACTIONS_DEFAULT_OFF)


def test_a_rollout_rejects_a_contract_the_family_cannot_drive(tmp_path, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "")
    args = SimpleNamespace(policy_family="aw0", control_hz=None, prefetch_lead_steps=None, plan_rows=None,
                           chunk_fusion="splice")
    with pytest.raises(SystemExit, match="Unexpected aw0 contract"):
        managed_rollout.rollout_joint_policy(
            None, dict(state_dim=7, action_dim=7, action_horizon=32, camera_mapping={}), args)


@pytest.mark.parametrize("model, instruction, adapter", [
    ("rho", HOLESAW, TOOL_POSE_QUAT_ADAPTER),
    ("dreamzero", HOLESAW, "identity"),
    ("fastwam", HOLESAW, "identity"),
    ("dw05", HOLESAW, "identity"),
    ("aw0", HOLESAW, SINGLE_ARM_IN_14),
    ("gr00t", "Pick three hole saws from the moving conveyor belt and place them in the red case.", "gr00t_yam"),
])
def test_cli_selects_the_family_instruction_and_adapter(model, instruction, adapter, monkeypatch):
    class Opened(Exception):
        pass

    captured = {}

    def host(**kwargs):
        captured.update(kwargs)
        raise Opened

    monkeypatch.chdir(YAM.parents[1])
    monkeypatch.setattr(sys, "argv", ["managed_rollout.py"])
    monkeypatch.setattr(managed_rollout, "Orchestral", lambda: SimpleNamespace(
        deployments=SimpleNamespace(get=lambda _: SimpleNamespace(model_id=model, checkpoint_uri="")),
        _http=SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr("orchestral_session_bridge.OrchestralSessionHost", host)
    with pytest.raises(Opened):
        managed_rollout.main()
    assert (captured["instruction"], captured["policy_adapter"]) == (instruction, adapter)


@pytest.mark.parametrize("model", ["rho", "dreamzero", "fastwam", "dw05", "aw0"])
def test_fine_tuned_families_take_no_noise_seed(model, monkeypatch, capsys):
    monkeypatch.setattr(managed_rollout, "Orchestral", lambda: SimpleNamespace(
        deployments=SimpleNamespace(get=lambda _: SimpleNamespace(model_id=model)),
        _http=SimpleNamespace(close=lambda: None)))
    monkeypatch.setattr(sys, "argv", ["managed_rollout.py", "--noise-seed", "0"])
    with pytest.raises(SystemExit):
        managed_rollout.main()
    assert "does not support --noise-seed" in capsys.readouterr().err


def test_overshooting_gripper_is_clipped_for_diffusion_families_only():
    overshoot = np.array([-0.2, 0.5, 1.3])
    for family in ("dreamzero", "fastwam", "dw05", "aw0"):
        np.testing.assert_array_equal(clip_predicted_aperture(overshoot, family), [0, 0.5, 1])
    with pytest.raises(ValueError, match="outside the accepted"):
        clip_predicted_aperture(overshoot, "gr00t")
    with pytest.raises(ValueError, match="non-finite"):
        clip_predicted_aperture(np.array([np.nan]), "dreamzero")


# --- Committed actions (RTC): always on, in the worker's own action space ------------------------------------

IDENTITIES = {  # a plausible contract per family of the table; pi05 runs through rtc_rollout.py instead
    "gr00t": dict(state_dim=7, action_dim=7, action_horizon=16),
    "molmoact2": dict(state_dim=7, action_dim=7, action_horizon=30),
    "dexmal": dict(state_dim=7, action_dim=7, action_horizon=16),
    "praxis1": dict(state_dim=8, action_dim=13, action_horizon=24),
    "verne": dict(state_dim=7, action_dim=10, action_horizon=16),
    "isaac05": dict(state_dim=7, action_dim=7, action_horizon=30),
    "flux3-action": dict(state_dim=7, action_dim=7, action_horizon=32),
    "longwam": dict(state_dim=7, action_dim=7, action_horizon=32),
    "rho": dict(state_dim=8, action_dim=8, action_horizon=32),
    "dreamzero": dict(state_dim=7, action_dim=7, action_horizon=24),
    "fastwam": dict(state_dim=7, action_dim=7, action_horizon=32),
    "dw05": dict(state_dim=7, action_dim=7, action_horizon=32),
    "aw0": dict(state_dim=14, action_dim=14, action_horizon=32),
}


def _families():
    return sorted({family for _, family in managed_rollout.FAMILY_PREFIXES})


def _rollout_settings(family, tmp_path, monkeypatch, **extra):
    saved = {}

    class Stop(Exception):
        pass

    def save(path, settings):
        saved.update(settings)
        raise Stop

    monkeypatch.chdir(YAM.parents[1])
    monkeypatch.setattr("builtins.input", lambda _: "")
    monkeypatch.setattr(managed_rollout, "record_settings", save)
    args = SimpleNamespace(policy_family=family, control_hz=None, prefetch_lead_steps=None, plan_rows=None,
                           chunk_fusion="splice", fusion_rows=4, shadow=False, instruction="go",
                           rrd=tmp_path / "run.rrd", deployment="d", seconds=30, noise_seed=None, **extra)
    with pytest.raises(Stop):
        managed_rollout.rollout_joint_policy(None, dict(IDENTITIES[family], camera_mapping={"top": "a", "left": "b"}), args)
    return saved


def test_committed_actions_rule_and_its_temporary_exceptions(tmp_path, monkeypatch):
    """On for every family; only the five Holesaw families are default-off until their plugins support it."""
    assert set(IDENTITIES) | {"pi05"} == set(_families()), "a new family needs an entry in IDENTITIES and its tests"
    assert managed_rollout.COMMITTED_ACTIONS_DEFAULT_OFF == {"rho", "dreamzero", "fastwam", "dw05", "aw0"}
    for family in IDENTITIES:
        saved = _rollout_settings(family, tmp_path, monkeypatch)
        if family in managed_rollout.COMMITTED_ACTIONS_DEFAULT_OFF:
            assert saved["committed_actions"] is False, family
            opted_in = _rollout_settings(family, tmp_path, monkeypatch, committed_actions=True)
            assert opted_in["committed_actions"] is True and opted_in["prefetch_lead_steps"] >= 1, family
        else:
            assert saved["committed_actions"] is True and saved["prefetch_lead_steps"] >= 1, family
            # No opt-out: the attribute cannot turn it off.
            assert _rollout_settings(family, tmp_path, monkeypatch, committed_actions=False)["committed_actions"] is True


def test_pi05_runs_through_real_time_chunking(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["managed_rollout.py"])
    monkeypatch.setattr(managed_rollout, "Orchestral", lambda: SimpleNamespace(
        deployments=SimpleNamespace(get=lambda _: SimpleNamespace(model_id="pi05")),
        _http=SimpleNamespace(close=lambda: None)))
    started = []
    monkeypatch.setattr(managed_rollout.runpy, "run_path", lambda path, **kw: started.append(path))
    managed_rollout.main()
    assert started and started[0].endswith("rtc_rollout.py")


def _cli(model, monkeypatch):
    monkeypatch.setattr(managed_rollout, "Orchestral", lambda: SimpleNamespace(
        deployments=SimpleNamespace(get=lambda _: SimpleNamespace(model_id=model, checkpoint_uri="")),
        _http=SimpleNamespace(close=lambda: None)))


def test_there_is_no_opt_out_for_the_other_families(monkeypatch, capsys):
    _cli("gr00t", monkeypatch)
    for argv, message in ((["--no-committed-actions"], "unrecognized arguments"),
                          (["--chunk-fusion", "rail"], "committed actions are on"),
                          (["--prefetch-lead-steps", "0"], "committed actions are on")):
        monkeypatch.setattr(sys, "argv", ["managed_rollout.py", *argv])
        with pytest.raises(SystemExit) as error:
            managed_rollout.main()
        assert error.value.code == 2 and message in capsys.readouterr().err


class _Opened(Exception):
    pass


def _fake_host(supported, calls):
    cameras = {"a": (4, 4), "b": (4, 4)}

    def predict(**kwargs):
        calls.append(kwargs)
        horizon, width = IDENTITY["action_horizon"], IDENTITY["action_dim"]
        return SimpleNamespace(actions=np.full((horizon, width), 0.5, np.float32))

    return SimpleNamespace(
        identity=dict(IDENTITY, camera_mapping={"top": "a"}, model_ref="mdl_example"),
        supports_committed_actions=lambda: supported, begin_episode=lambda: None,
        session=SimpleNamespace(predict=predict, grant=SimpleNamespace(io_spec=SimpleNamespace(cameras=cameras))))


IDENTITY = dict(state_dim=7, action_dim=7, action_horizon=8)


@pytest.mark.parametrize("family", ["dreamzero", "fastwam", "dw05"])
def test_default_off_families_start_against_a_worker_without_committed_actions(family):
    calls = []
    managed_rollout.probe(_fake_host(False, calls), family, "go", committed_actions=False)
    assert len(calls) == 1 and "committed_actions" not in calls[0]


@pytest.mark.parametrize("family", ["dreamzero", "fastwam", "dw05"])
def test_opting_in_on_a_worker_without_support_is_a_hard_error(family):
    calls = []
    with pytest.raises(SystemExit, match="plugin must implement"):
        managed_rollout.probe(_fake_host(False, calls), family, "go", committed_actions=True)
    assert not calls
    managed_rollout.probe(_fake_host(True, calls), family, "go", committed_actions=True)
    assert "committed_actions" in calls[-1]


@pytest.mark.parametrize("model", ["rho", "aw0", "dreamzero"])
def test_the_cli_runs_default_off_families_without_committed_actions_and_opts_in_on_request(model, monkeypatch, capsys):
    seen = {}

    def host(**kwargs):
        raise _Opened

    def run(argv):
        seen.clear()
        monkeypatch.chdir(YAM.parents[1])
        monkeypatch.setattr(sys, "argv", ["managed_rollout.py", *argv])
        _cli(model, monkeypatch)
        monkeypatch.setattr("orchestral_session_bridge.OrchestralSessionHost", host)
        monkeypatch.setattr(managed_rollout, "probe", lambda h, f, i, **kw: seen.update(kw))
        monkeypatch.setattr(managed_rollout, "rollout_joint_policy", lambda *a: None)

        class Host(SimpleNamespace):
            def open(self):
                return {}

            def close(self):
                pass

        monkeypatch.setattr("orchestral_session_bridge.OrchestralSessionHost", lambda **kw: Host(**kw))
        managed_rollout.main()
        return seen["committed_actions"]

    assert run([]) is False
    assert f"Committed actions are off for {model}" in capsys.readouterr().out
    assert run(["--chunk-fusion", "rail"]) is False  # the older fusion modes stay available while it is off
    assert run(["--committed-actions"]) is True
    monkeypatch.setattr(sys, "argv", ["managed_rollout.py", "--committed-actions", "--chunk-fusion", "rail"])
    with pytest.raises(SystemExit):
        managed_rollout.main()


@pytest.mark.parametrize("family", sorted(IDENTITIES) + ["pi05"])
def test_a_worker_without_committed_actions_is_refused_with_its_contract(family):
    identity = IDENTITIES.get(family, dict(state_dim=7, action_dim=7, action_horizon=15))
    host = SimpleNamespace(identity=dict(identity, model_ref="mdl_example"), supports_committed_actions=lambda: False,
                           session=None)
    with pytest.raises(SystemExit) as error:
        managed_rollout.probe(host, family, "go", committed_actions=True)
    message = str(error.value)
    assert family in message and "mdl_example" in message and "plugin must implement" in message
    assert f"(k, {identity['action_dim']})" in message and "first k rows equal it exactly" in message


def _echo(seen, tail):
    def predict(*, cameras, state, instruction, episode_start, committed_actions=None, **kwargs):
        seen["state"], seen["committed"] = state, committed_actions
        prefix = np.zeros((0,) + tail.shape[1:], np.float32) if committed_actions is None else committed_actions
        return SimpleNamespace(actions=np.concatenate([prefix, tail[len(prefix):]]).astype(np.float32))
    return predict


def test_rho_commits_the_queued_joint_targets_as_tool_poses():
    ik = adapter()
    start = CONFIGS[2]
    path = np.linspace(start, start + np.array([0.3, 0.2, -0.2, 0.2, 0.2, 0.2, 0.0]), 36)
    queued = path[:6].astype(np.float32)
    tail = np.stack([ik.state(joint) for joint in path[2:34]])
    seen = {}
    chunk, _ = checked_chunk(SimpleNamespace(predict=_echo(seen, tail)), {}, start, HOLESAW, episode_start=False,
                             action_shape=(32, 8), pose_actions=ik, model_label="rho", max_plan_rows=32,
                             committed_actions=queued)
    assert seen["committed"].shape == (6, 8)
    np.testing.assert_array_equal(seen["committed"], np.stack([ik.state(q) for q in queued]))
    np.testing.assert_array_equal(chunk[:6, :6], queued[:, :6])  # the executed targets stay exactly as queued
    assert chunk.shape == (32, 7)
    # Rows after the prefix start at the last queued target, within the step bound.
    assert np.max(np.abs(chunk[6, :6] - queued[-1, :6])) <= 0.06 + 1e-6


def test_aw0_commits_padded_rows_and_the_right_arm_never_leaks():
    parked = managed_rollout.ISAAC_PARKED_RIGHT_ARM
    measured = np.array([0.1, 1.0, 1.0, -0.5, 0.0, 0.2, 0.4], np.float32)
    queued = np.linspace(measured, measured + 0.05, 4).astype(np.float32)
    tail = np.concatenate([np.linspace(measured, measured + 0.3, 32), np.full((32, 7), 9.0)], axis=1)
    seen = {}
    chunk, _ = checked_chunk(SimpleNamespace(predict=_echo(seen, tail)), {}, measured, HOLESAW, episode_start=False,
                             action_shape=(32, 14), model_label="aw0", state_padding=parked, committed_actions=queued)
    assert seen["committed"].shape == (4, 14)
    np.testing.assert_array_equal(seen["committed"][:, :7], queued)
    np.testing.assert_array_equal(seen["committed"][:, 7:], np.broadcast_to(parked, (4, 7)))
    assert chunk.shape == (32, 7) and not np.any(chunk == 9.0)
    np.testing.assert_array_equal(chunk[:4], queued)


@pytest.mark.parametrize("family", ["dreamzero", "fastwam", "dw05"])
def test_joint_families_commit_the_absolute_queued_joint_rows(family):
    measured = np.array([0.1, 1.0, 1.0, -0.5, 0.0, 0.2, 0.4], np.float32)
    queued = np.linspace(measured, measured + 0.05, 3).astype(np.float32)
    tail = np.linspace(measured, measured + 0.3, 24).astype(np.float32)
    seen = {}
    chunk, _ = checked_chunk(SimpleNamespace(predict=_echo(seen, tail)), {}, measured, HOLESAW, episode_start=False,
                             action_shape=(24, 7), model_label=family, committed_actions=queued)
    np.testing.assert_array_equal(seen["committed"], queued)
    np.testing.assert_array_equal(chunk[:3], queued)


def test_verne_commits_tool_poses_in_the_rows_its_worker_returns():
    from tool_pose_actions import ToolPoseActions

    ik = ToolPoseActions(model_path=URDF, base_body="base", tool_body="gripper", joint_names=JOINTS,
                         max_joint_step_rad=0.12)
    queued = np.linspace(CONFIGS[2], CONFIGS[2] + 0.05, 3).astype(np.float32)
    rows = ik.committed(queued, matrix_policy=False)
    assert rows.shape == (3, 10)
    solved = ik.actions(rows, queued[0])
    np.testing.assert_allclose(solved[:, :6], queued[:, :6], atol=2e-3)
    np.testing.assert_allclose(solved[:, 6], queued[:, 6], atol=1e-6)
    assert ik.committed(queued).shape == (3, 13)  # the Praxis layout is unchanged
