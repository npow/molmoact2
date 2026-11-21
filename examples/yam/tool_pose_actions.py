"""Robot-side conversion between tool poses and joint targets."""
from pathlib import Path

import numpy as np


class UnreachableToolPose(ValueError):
    pass


class ToolPoseActions:
    """Use a configured MuJoCo robot model and keep commands within its joint limits."""

    def __init__(self, *, model_path, base_body, tool_body, joint_names, max_joint_step_rad):
        import mujoco

        self._mj = mujoco
        spec = mujoco.MjSpec.from_file(str(Path(model_path).expanduser()))
        # Preserve fixed tool and base links in URDFs.
        spec.compiler.fusestatic = False
        self.model = spec.compile()
        self.base = self.model.body(base_body).id
        self.tool = self.model.body(tool_body).id
        if self.base == self.tool or len(joint_names) != 6 or len(set(joint_names)) != 6:
            raise ValueError("Tool pose conversion needs distinct base/tool bodies and six ordered joints")
        joints = np.array([self.model.joint(name).id for name in joint_names])
        if not all(self.model.jnt_type[joints] == mujoco.mjtJoint.mjJNT_HINGE) or not all(self.model.jnt_limited[joints]):
            raise ValueError("Tool pose conversion requires six limited hinge joints")
        self.qpos = self.model.jnt_qposadr[joints]
        self.lower, self.upper = self.model.jnt_range[joints].T
        if not np.isfinite(max_joint_step_rad) or not 0 < max_joint_step_rad < np.pi:
            raise ValueError("max_joint_step_rad must be between zero and pi")
        self.max_step = max_joint_step_rad

    def _pose(self, data, joints):
        data.qpos[self.qpos] = joints
        self._mj.mj_kinematics(self.model, data)
        base_rotation = data.xmat[self.base].reshape(3, 3)
        offset = data.xpos[self.tool] - data.xpos[self.base]
        return base_rotation.T @ offset, base_rotation.T @ data.xmat[self.tool].reshape(3, 3)

    def _rotvec(self, rotation):
        quat, vector = np.empty(4), np.empty(3)
        self._mj.mju_mat2Quat(quat, rotation.ravel())
        self._mj.mju_quat2Vel(vector, quat if quat[0] >= 0 else -quat, 1.0)
        return vector

    def _checked_state(self, state):
        state = np.asarray(state, dtype=np.float64)
        if state.shape != (7,) or not np.isfinite(state).all() or not 0 <= state[6] <= 1:
            raise ValueError("Robot state must have six finite joint angles and a normalized gripper")
        return state

    def state(self, joint_state, *, matrix_policy=False):
        state = self._checked_state(joint_state)
        position, rotation = self._pose(self._mj.MjData(self.model), state[:6])
        if matrix_policy:
            quaternion = np.empty(4)
            self._mj.mju_mat2Quat(quaternion, rotation.ravel())
            return np.r_[position, quaternion, 1.0 - state[6]].astype(np.float32)
        return np.concatenate((position, self._rotvec(rotation), state[6:])).astype(np.float32)

    def _solve(self, data, target_position, target_rotation, previous):
        lower = np.maximum(self.lower, previous - self.max_step)
        upper = np.minimum(self.upper, previous + self.max_step)
        if np.any(lower >= upper):
            raise UnreachableToolPose("No joint target lies within both the joint limits and step bound")
        from scipy.optimize import least_squares

        position_tolerance = 0.002
        rotation_tolerance = 0.02

        def residual(joints):
            position, rotation = self._pose(data, joints)
            return np.concatenate((
                (target_position - position) / position_tolerance,
                self._rotvec(target_rotation @ rotation.T) / rotation_tolerance,
            ))

        joints = np.clip(previous, lower, upper)
        error = residual(joints)
        if np.linalg.norm(error[:3]) <= 1 and np.linalg.norm(error[3:]) <= 1:
            return joints
        result = least_squares(residual, joints, bounds=(lower, upper), max_nfev=40)
        if np.linalg.norm(result.fun[:3]) <= 1 and np.linalg.norm(result.fun[3:]) <= 1:
            return result.x
        raise UnreachableToolPose("Tool target is unreachable within the configured joint limits and step bound")

    def actions(self, rows, joint_state, *, matrix_policy=False):
        rows = np.asarray(rows, dtype=np.float64)
        width = 13 if matrix_policy else 10
        if rows.ndim != 2 or not len(rows) or rows.shape[1] != width or not np.isfinite(rows).all():
            raise ValueError(f"Tool actions must be finite rows of width {width}")
        joints = self._checked_state(joint_state)[:6]
        data = self._mj.MjData(self.model)
        output = np.empty((len(rows), 7), dtype=np.float32)
        for i, row in enumerate(rows):
            gripper = 1.0 - row[-1] if matrix_policy else row[-1]
            if matrix_policy:
                matrix = row[3:12].reshape(3, 3)
                if np.linalg.det(matrix) <= 0 or not np.allclose(matrix @ matrix.T, np.eye(3), atol=0.05):
                    raise ValueError(f"Invalid rotation matrix in tool action row {i}")
            first, second = row[3:6].copy(), row[6:9].copy()
            norm = np.linalg.norm(first)
            if norm < 1e-6 or not 0 <= gripper <= 1:
                raise ValueError(f"Invalid rotation or gripper in tool action row {i}")
            first /= norm
            second -= first * np.dot(first, second)
            norm = np.linalg.norm(second)
            if norm < 1e-6:
                raise ValueError(f"Degenerate rotation in tool action row {i}")
            second /= norm
            try:
                joints = self._solve(data, row[:3], np.stack((first, second, np.cross(first, second))), joints)
            except UnreachableToolPose as error:
                if i:
                    return output[:i].copy()
                raise UnreachableToolPose(f"Tool action row {i}: {error}") from error
            output[i, :6], output[i, 6] = joints, gripper
        return output
