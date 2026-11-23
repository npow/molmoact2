import tempfile
import unittest
from pathlib import Path

import numpy as np

from tool_pose_actions import HolesawBimanualActions, ToolPoseActions


class ToolPoseActionsTests(unittest.TestCase):
    def test_holesaw_bimanual_grasp_frame_and_committed_handoff(self):
        from types import SimpleNamespace
        from scipy.spatial.transform import Rotation
        from rollout_loop import checked_chunk

        adapter = HolesawBimanualActions(
            model_path=Path(__file__).resolve().parents[1] / "configs/yam_holesaw_kinematics.urdf",
            base_body="base", tool_body="grasp_site", joint_names=[f"joint{i}" for i in range(1, 7)],
            max_joint_step_rad=0.12,
        )
        # Independent pose/joint labels from Holesaw validation episode 4, frame 75.
        joints = np.array([-.02880140, 2.15362024, 1.61650264, -.81731135, -.35114825, -1.54631114, .00371233],
                          dtype=np.float32)
        xyz = np.array([.75806512, -.25975462, .33331013])
        rotation = Rotation.from_euler("xyz", [-2.92022300, .34809756, -1.54646909])
        state = adapter.state(joints)
        np.testing.assert_allclose(state[:3], [.06576730, .37384747, .45302705], atol=3e-6)
        np.testing.assert_allclose(state[8:11], xyz, atol=3e-6)
        np.testing.assert_allclose(Rotation.from_quat(state[11:15], scalar_first=True).as_matrix(),
                                   rotation.as_matrix(), atol=5e-6)
        self.assertAlmostEqual(state[15], .99628764, places=6)
        queued = joints[None].copy()
        queued[0, 0] += .02
        target = queued.copy()
        target[0, 1] += .04
        target[0, 6] = .75

        def predict(*, state, noise_seed, committed_actions, **kwargs):
            self.assertEqual(noise_seed, 0)
            np.testing.assert_array_equal(committed_actions, adapter.committed(queued))
            response = np.concatenate((committed_actions, adapter.committed(target)))
            # Model changes in the parked slot never become commands to the working arm.
            response[-1, :3] += 1
            return SimpleNamespace(actions=response)

        result, _ = checked_chunk(SimpleNamespace(predict=predict), {}, joints, "pick", episode_start=False,
                                  action_shape=(2, 26), pose_actions=adapter, matrix_policy=True,
                                  committed_actions=queued, noise_seed=0)
        np.testing.assert_array_equal(result[:1], queued)
        np.testing.assert_allclose(adapter.committed(result[1:]), adapter.committed(target), atol=2e-5)

    def test_round_trip_and_reject_unreachable_action(self):
        # A six-joint chain with a fixed tool offset and a full-rank workspace.
        links = [
            ("0 0 0.1", "0 0 1"),
            ("0.1 0 0", "0 1 0"),
            ("0.18 0 0", "0 1 0"),
            ("0.14 0 0", "1 0 0"),
            ("0.07 0 0", "0 1 0"),
            ("0.04 0 0", "1 0 0"),
        ]
        bodies = "".join(
            f'<body pos="{position}"><joint name="j{i}" axis="{axis}" range="-3 3" />'
            '<geom type="sphere" size="0.01" mass="0.1" />'
            for i, (position, axis) in enumerate(links)
        )
        xml = (
            '<mujoco><compiler angle="radian"/><worldbody><body name="base">'
            f'{bodies}<body name="tool" pos="0.05 0 0"/></body>'
            f'{"</body>" * len(links)}</worldbody></mujoco>'
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "arm.xml"
            path.write_text(xml)
            adapter = ToolPoseActions(
                model_path=path, base_body="base", tool_body="tool",
                joint_names=[f"j{i}" for i in range(6)], max_joint_step_rad=0.12,
            )
        start = np.array([0.15, 0.45, -0.7, 0.1, 0.2, 0.1, 0.6])
        goal = start + np.array([0.025, 0.04, -0.03, 0.03, 0.02, -0.04, -0.2])
        state = adapter.state(goal)
        angle = np.linalg.norm(state[3:6])
        x, y, z = state[3:6] / angle
        skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
        rotation = np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
        pose_row = np.concatenate((state[:3], rotation[:2].ravel(), state[6:]))
        matrix_state = adapter.state(goal, matrix_policy=True)
        self.assertAlmostEqual(float(np.linalg.norm(matrix_state[3:7])), 1.0)
        self.assertAlmostEqual(float(matrix_state[-1]), 1.0 - goal[-1])
        matrix_row = np.r_[matrix_state[:3], rotation.ravel(), matrix_state[-1]]
        matrix_command, = adapter.actions([matrix_row], start, matrix_policy=True)
        np.testing.assert_allclose(matrix_command, goal, atol=0.01)
        command, = adapter.actions([pose_row], start)
        np.testing.assert_allclose(adapter.state(command)[:3], state[:3], atol=0.002)
        np.testing.assert_allclose(adapter.state(command)[3:6], state[3:6], atol=0.02)
        self.assertAlmostEqual(float(command[6]), 0.4)
        self.assertLessEqual(np.max(np.abs(command[:6] - start[:6])), 0.12)
        unreachable = pose_row.copy()
        unreachable[:3] = 10
        prefix = adapter.actions([pose_row, unreachable], start)
        np.testing.assert_array_equal(prefix, [command])
        with self.assertRaisesRegex(ValueError, "unreachable"):
            adapter.actions([unreachable], start)
