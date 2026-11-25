"""Tool-pose checkpoints driven from their declared semantics alone (declared_tool_poses.py)."""
from pathlib import Path

import numpy as np
import pytest

from declared_tool_poses import DeclaredToolPoses, decode_rotations, encode_rotations

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
JOINTS = [f"joint{i}" for i in range(1, 7)]

# The FLUX 3 Action EE-50 fine-tune's declaration: poses of the tool frame it was trained in, on the pinned
# i2rt YAM description (configs/i2rt_yam_v1.urdf): the gripper link turned 90 degrees about z, at the
# finger joints' midpoint.
HAND = {"urdf_sha256": "5b9ccf966aa353ccf1e82ad6e751b899333fab7c0b4d1071316172201da68c4e",
        "base_link": "base", "link": "gripper", "xyz": [0, 0, 0.0566829], "rpy": [0, 0, np.pi / 2]}
GRIPPER = {"name": "gripper", "kind": "gripper", "width": 1, "open_value": 1, "closed_value": 0}
FLUX_EE50 = {
    "control_rate_hz": 30, "tools": {"hand": HAND},
    "state": [{"name": "tool", "kind": "tool_position", "width": 3, "tool": "hand"},
              {"name": "rotation", "kind": "tool_rotation", "width": 3, "tool": "hand", "encoding": "axis_angle"},
              GRIPPER],
    "action": [{"name": "tool", "kind": "tool_position", "width": 3, "tool": "hand", "reference": "absolute"},
               {"name": "rotation", "kind": "tool_rotation", "width": 6, "tool": "hand", "encoding": "matrix_rows",
                "reference": "absolute"},
               {**GRIPPER, "reference": "absolute"}],
}

# Training poses: the fine-tune's own kinematics (kin/yam_fk.py) at two recorded Holesaw joint states.
TRAINING_POSES = [
    ([-0.08476011, 2.51449609, 1.72549498, -0.08929376, -0.90112638, -0.85578346, 0.95836365],
     [0.48470464, -0.05776855, 0.08994497],
     [[0.87162585, 0.16584866, -0.46126197], [-0.48248363, 0.45630013, -0.74766285],
      [0.08647502, 0.87423361, 0.47774225]]),
    ([-0.11348897, 0.03681239, 0.01049058, -0.0616083, 0.04329747, 0.0257496, 0.00363373],
     [0.05514982, -0.00536674, 0.17210037],
     [[-0.15816969, 0.08320413, -0.98390011], [-0.98717032, -0.03536744, 0.15570453],
      [-0.02184277, 0.99590472, 0.0877307]]),
]


def _poses(semantics=FLUX_EE50, **options):
    return DeclaredToolPoses(semantics, state_dim=7, action_shape=(32, 10), descriptions=[CONFIGS],
                             joint_names=JOINTS, max_joint_step_rad=0.12, **options)


def test_the_declared_tool_frame_reproduces_the_poses_the_checkpoint_was_trained_on():
    poses = _poses()
    for joints, position, rotation in TRAINING_POSES:
        state = poses.state(joints)
        np.testing.assert_allclose(state[:3], position, atol=1e-6)
        np.testing.assert_allclose(decode_rotations(state[None, 3:6], "axis_angle")[0], rotation, atol=1e-6)
        assert state[6] == pytest.approx(joints[6])
        # The Worker's row for that pose solves back to the recorded joints, and they commit as that row.
        row = np.r_[position, np.ravel(rotation)[:6], joints[6]][None]
        solved = poses.actions(row, np.array(joints))
        np.testing.assert_allclose(solved[0], joints, atol=2e-3)
        np.testing.assert_allclose(poses.committed(solved), row, atol=2e-3)


def test_another_robot_description_is_refused():
    other = {**FLUX_EE50, "tools": {"hand": {**HAND, "urdf_sha256": "0" * 64}}}
    with pytest.raises(ValueError, match="no robot description with sha256 0000"):
        _poses(other)


def test_an_inverted_gripper_is_converted_both_ways():
    inverted = {**FLUX_EE50, "state": [*FLUX_EE50["state"][:2], {**GRIPPER, "open_value": 0, "closed_value": 1}],
                "action": [*FLUX_EE50["action"][:2], {**GRIPPER, "open_value": 0, "closed_value": 1,
                                                      "reference": "absolute"}]}
    poses = _poses(inverted)
    joints, position, rotation = TRAINING_POSES[0]
    assert poses.state(joints)[6] == pytest.approx(1 - joints[6])
    row = np.r_[position, np.ravel(rotation)[:6], 0.25][None]
    assert poses.actions(row, np.array(joints))[0, 6] == pytest.approx(0.75)


@pytest.mark.parametrize("encoding, turn", [  # a 90 degree turn about z, written out in each encoding
    ("axis_angle", [0, 0, np.pi / 2]), ("matrix_rows", [0, -1, 0, 1, 0, 0]), ("matrix", [0, -1, 0, 1, 0, 0, 0, 0, 1]),
    ("quaternion_wxyz", [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]), ("quaternion_xyzw", [0, 0, np.sqrt(0.5), np.sqrt(0.5)])])
def test_rotation_encodings(encoding, turn):
    rz = np.array([[[0.0, -1, 0], [1, 0, 0], [0, 0, 1]]])
    np.testing.assert_allclose(encode_rotations(rz, encoding)[0], turn, atol=1e-12)
    np.testing.assert_allclose(decode_rotations(np.array([turn]), encoding), rz, atol=1e-12)
