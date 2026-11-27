import tempfile
import unittest
from pathlib import Path

import numpy as np

from tool_pose_actions import HolesawBimanualActions, HolesawSingleArmActions, ToolPoseActions


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
        # The solver stops at 0.2 mm and 0.002 rad (a tenth of its tolerances), so poses match to about 1e-3.
        np.testing.assert_allclose(adapter.committed(result[1:]), adapter.committed(target), atol=2e-3)

    def test_single_arm_holesaw_keeps_a_real_chunk_whose_orientation_is_slightly_off(self):
        # A 24-row chunk the checkpoint returned on the robot (run nimble-zebra-t8qb_20261126_153632) and the measured
        # joints it was solved from. At the default 0.02 rad rotation tolerance IK kept only its first row.
        sample = np.load(Path(__file__).with_name("fixtures") / "praxis1_single_arm_chunk.npz")
        rows, start = sample["rows"].astype(np.float64), sample["joints"].astype(np.float64)
        kwargs = dict(model_path=Path(__file__).resolve().parents[1] / "configs/yam_holesaw_kinematics.urdf",
                      base_body="base", tool_body="grasp_site", joint_names=[f"joint{i}" for i in range(1, 7)],
                      max_joint_step_rad=0.12)
        adapter = HolesawSingleArmActions(**kwargs)
        solved = adapter.actions(rows, start, matrix_policy=True)
        self.assertEqual(len(solved), len(rows))
        data = adapter._mj.MjData(adapter.model)
        for joints, row in zip(solved, rows):
            position, _ = adapter._pose(data, joints[:6].astype(np.float64))
            self.assertLess(np.linalg.norm(position - row[:3]), 4e-3)
        self.assertLessEqual(np.abs(np.diff(np.vstack((start[:6], solved[:, :6])), axis=0)[1:]).max(), 0.2 + 1e-6)
        strict = HolesawSingleArmActions(**kwargs)
        strict.rotation_tolerance = ToolPoseActions.rotation_tolerance
        self.assertEqual(len(strict.actions(rows, start, matrix_policy=True)), 1)

    def test_a_run_starts_from_the_near_singular_zero_pose_without_truncating_the_chunk(self):
        adapter = HolesawBimanualActions(
            model_path=Path(__file__).resolve().parents[1] / "configs/yam_holesaw_kinematics.urdf",
            base_body="base", tool_body="grasp_site", joint_names=[f"joint{i}" for i in range(1, 7)],
            max_joint_step_rad=0.12,
        )
        # The arm starts a run near all-zero joints. Unfolding the wrist there takes more than the 0.12 rad
        # per-row bound but less than the 0.2 rad per tick the executor follows, so a plan must not stop early.
        start = np.array([0.0, 0.02, 0.01, -0.06, 0.0, 0.0, 0.0], dtype=np.float32)
        path = [start.copy()]
        for _ in range(7):
            step = path[-1].copy()
            step[4] += 0.18
            step[3] += 0.03
            path.append(step)
        rows = adapter.committed(np.array(path[1:]))
        solved = adapter.actions(rows, start)
        self.assertEqual(len(solved), len(rows))
        np.testing.assert_allclose(solved[:, :6], np.array(path[1:])[:, :6], atol=2e-2)

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
