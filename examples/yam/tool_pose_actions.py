"""Robot-side conversion between tool poses and joint targets."""
import math
from pathlib import Path

import numpy as np


class UnreachableToolPose(ValueError):
    pass


class ToolPoseActions:
    """Use a configured MuJoCo robot model and keep commands within its joint limits."""

    # Residuals are scaled by these tolerances. A start pose inside one tolerance is kept as is; a solved
    # pose is accepted within ``accept_norm`` tolerances.
    position_tolerance = 0.002
    rotation_tolerance = 0.02
    accept_norm = 1
    # Row 0 may start farther from the measured joints than one row's step; None keeps the per-row bound.
    first_row_max_step_rad = None

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
        self.dof = self.model.jnt_dofadr[joints]
        self.lower, self.upper = self.model.jnt_range[joints].T
        if not np.isfinite(max_joint_step_rad) or not 0 < max_joint_step_rad < np.pi:
            raise ValueError("max_joint_step_rad must be between zero and pi")
        self.max_step = max_joint_step_rad
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))
        self._quat = np.empty(4)
        self._vec = np.empty(3)

    def _pose(self, data, joints):
        data.qpos[self.qpos] = joints
        self._mj.mj_kinematics(self.model, data)
        base_rotation = data.xmat[self.base].reshape(3, 3)
        offset = data.xpos[self.tool] - data.xpos[self.base]
        return base_rotation.T @ offset, base_rotation.T @ data.xmat[self.tool].reshape(3, 3)

    def _rotvec(self, rotation):
        quat, vector = self._quat, self._vec
        self._mj.mju_mat2Quat(quat, rotation.ravel())
        if quat[0] < 0:
            quat *= -1
        self._mj.mju_quat2Vel(vector, quat, 1.0)
        return vector.copy()

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

    def committed(self, joint_targets, *, matrix_policy=True):
        """Encode the exact queued joint commands as absolute pose rows in the worker's own layout.

        Praxis rows are position, rotation matrix and an inverted gripper (width 13). With
        ``matrix_policy=False`` they are the rows ``actions`` reads: position, the first two rotation-matrix
        rows and the gripper (width 10).
        """
        data = self._mj.MjData(self.model)
        rows = []
        for target in joint_targets:
            target = self._checked_state(target)
            position, rotation = self._pose(data, target[:6])
            if matrix_policy:
                rows.append(np.r_[position, rotation.ravel(), 1.0 - target[6]])
            else:
                rows.append(np.r_[position, rotation[:2].ravel(), target[6]])
        return np.asarray(rows, dtype=np.float32)

    def _frame_rotation(self):
        """Rotation from the base frame to the policy frame; None when they coincide."""
        return None

    def _jacobian(self, data, joints, kinematics_current=False):
        """Position and angular Jacobians of the tool in the policy frame (3 x 6 each).

        ``kinematics_current`` says ``data`` already holds the forward kinematics of ``joints``.
        """
        mj = self._mj
        if not kinematics_current:
            data.qpos[self.qpos] = joints
            mj.mj_kinematics(self.model, data)
        mj.mj_comPos(self.model, data)
        jacp, jacr = self._jacp, self._jacr
        mj.mj_jacBody(self.model, data, jacp, jacr, self.tool)
        base = data.xmat[self.base].reshape(3, 3).T
        position, angular = base @ jacp[:, self.dof], base @ jacr[:, self.dof]
        frame = self._frame_rotation()
        return (position, angular) if frame is None else (frame @ position, frame @ angular)

    def _damped_solve(self, data, target_position, target_rotation, start, lower, upper, start_error=None):
        """Levenberg-Marquardt with an analytic Jacobian, inside the joint box; None when it does not converge.

        Same residual and acceptance as the ``least_squares`` fallback in ``_solve``, which handles every
        case this does not reach, so a target is reachable exactly when it was before.
        """
        position_tolerance, rotation_tolerance = self.position_tolerance, self.rotation_tolerance
        accept = self.accept_norm * self.accept_norm
        scale = np.r_[np.full(3, 1 / position_tolerance), np.full(3, 1 / rotation_tolerance)]

        def residual(joints):
            position, rotation = self._pose(data, joints)
            return np.concatenate((
                (target_position - position) / position_tolerance,
                self._rotvec(target_rotation @ rotation.T) / rotation_tolerance,
            ))

        def converged(error, limit):
            return error[:3] @ error[:3] <= limit and error[3:] @ error[3:] <= limit

        joints = start
        error = residual(joints) if start_error is None else start_error
        cost = error @ error
        kinematics_current = True  # residual(start) was the last pose evaluated, here or in ``_solve``
        damping = 1e-2
        for _ in range(25):
            if converged(error, 0.01):
                break
            position, angular = self._jacobian(data, joints, kinematics_current)
            jacobian = -np.vstack((position, angular)) * scale[:, None]
            gradient = jacobian.T @ error
            normal = jacobian.T @ jacobian
            diagonal = np.diag(normal) + 1e-9
            for _ in range(8):
                lhs = normal.copy()
                lhs.flat[::7] += damping * diagonal
                step = -np.linalg.solve(lhs, gradient)
                trial = np.clip(joints + step, lower, upper)
                trial_error = residual(trial)
                trial_cost = trial_error @ trial_error
                if trial_cost < cost:
                    joints, error, cost = trial, trial_error, trial_cost
                    kinematics_current = True  # the accepted trial was the last pose evaluated
                    damping = max(damping * 0.3, 1e-9)
                    break
                damping *= 10
                kinematics_current = False
            else:
                break
        return joints if converged(error, accept) else None

    def _solve(self, data, target_position, target_rotation, previous, max_step=None):
        max_step = self.max_step if max_step is None else max_step
        lower = np.maximum(self.lower, previous - max_step)
        upper = np.minimum(self.upper, previous + max_step)
        if np.any(lower >= upper):
            raise UnreachableToolPose("No joint target lies within both the joint limits and step bound")
        from scipy.optimize import least_squares

        position_tolerance, rotation_tolerance = self.position_tolerance, self.rotation_tolerance

        def residual(joints):
            position, rotation = self._pose(data, joints)
            return np.concatenate((
                (target_position - position) / position_tolerance,
                self._rotvec(target_rotation @ rotation.T) / rotation_tolerance,
            ))

        joints = np.clip(previous, lower, upper)
        error = residual(joints)
        if error[:3] @ error[:3] <= 1 and error[3:] @ error[3:] <= 1:
            return joints
        fast = self._damped_solve(data, target_position, target_rotation, joints, lower, upper, error)
        if fast is not None:
            return fast
        scale = np.r_[np.full(3, 1 / position_tolerance), np.full(3, 1 / rotation_tolerance)]

        def jacobian(q):  # analytic, instead of 6 finite-difference residuals per iteration
            return -np.vstack(self._jacobian(data, q)) * scale[:, None]

        result = least_squares(residual, joints, jac=jacobian, bounds=(lower, upper), max_nfev=40)
        if np.linalg.norm(result.fun[:3]) <= self.accept_norm and np.linalg.norm(result.fun[3:]) <= self.accept_norm:
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
        if matrix_policy:
            matrices = rows[:, 3:12].reshape(-1, 3, 3)
            bad = (np.linalg.det(matrices) <= 0) | (np.abs(matrices @ matrices.transpose(0, 2, 1) - np.eye(3)).max((1, 2)) > 0.05)
            if bad.any():
                raise ValueError(f"Invalid rotation matrix in tool action row {int(np.argmax(bad))}")
        for i, row in enumerate(rows):
            gripper = 1.0 - row[-1] if matrix_policy else row[-1]
            first, second = row[3:6].copy(), row[6:9].copy()
            norm = math.sqrt(first @ first)
            if norm < 1e-6 or not 0 <= gripper <= 1:
                raise ValueError(f"Invalid rotation or gripper in tool action row {i}")
            first /= norm
            second -= first * (first @ second)
            norm = math.sqrt(second @ second)
            if norm < 1e-6:
                raise ValueError(f"Degenerate rotation in tool action row {i}")
            second /= norm
            third = np.array((first[1] * second[2] - first[2] * second[1],
                              first[2] * second[0] - first[0] * second[2],
                              first[0] * second[1] - first[1] * second[0]))
            try:
                joints = self._solve(data, row[:3], np.stack((first, second, third)), joints,
                                     max_step=max(self.max_step, self.first_row_max_step_rad)
                                     if i == 0 and self.first_row_max_step_rad else None)
            except UnreachableToolPose as error:
                if i:
                    return output[:i].copy()
                raise UnreachableToolPose(f"Tool action row {i}: {error}") from error
            output[i, :6], output[i, 6] = joints, gripper
        return output


class QuatToolPoseActions(ToolPoseActions):
    """Rho's end-effector contract: rows ``[x y z qx qy qz qw gripper]``, absolute, gripper 1 = open.

    Use the ``rho_tool`` body of ``configs/yam_holesaw_kinematics.urdf``. Rho's tool frame is the
    flange FK composed with its fitted tool transform, so the URDF holds that transform and this class
    only converts quaternions. The Rho base frame is this URDF's ``base`` frame (the BusyBox calibration
    fit needed no base offset); ``base_rotation`` and ``base_translation`` give a fixed transform
    from the model base to the policy base for a rig where that stops being true.
    """

    # Solve tightly (0.2 mm, 0.002 rad) but accept a pose within 2 mm, 0.02 rad, the other adapters' tolerance.
    position_tolerance = 0.0002
    rotation_tolerance = 0.002
    accept_norm = 10
    # Row 0 is solved from the MEASURED joints, which lag the model's first command: in the 30 Hz Holesaw data the first
    # action row is more than 0.06 rad from the measured joints in 59% of frames (p99 0.40 rad), because actions are leader
    # commands. Row 0 is therefore bounded only by the joint limits and this generous cap; the executor's per-tick clamp
    # limits the actual motion. Later rows keep the per-row step bound (consecutive rows are one control tick apart).
    first_row_max_step_rad = 1.0

    def __init__(self, *, base_rotation=None, base_translation=None, **kwargs):
        super().__init__(**kwargs)
        self._base_rotation = np.eye(3) if base_rotation is None else np.asarray(base_rotation, dtype=np.float64)
        self._base_translation = np.zeros(3) if base_translation is None else np.asarray(base_translation, dtype=np.float64)
        if (self._base_rotation.shape != (3, 3) or self._base_translation.shape != (3,)
                or not np.allclose(self._base_rotation @ self._base_rotation.T, np.eye(3), atol=1e-6)
                or np.linalg.det(self._base_rotation) <= 0):
            raise ValueError("Base transform needs a proper 3x3 rotation and a 3-vector translation")

    def _frame_rotation(self):
        return self._base_rotation

    def _pose(self, data, joints):
        position, rotation = super()._pose(data, joints)
        return self._base_rotation @ position + self._base_translation, self._base_rotation @ rotation

    def state(self, joint_state, *, matrix_policy=False):
        state = self._checked_state(joint_state)
        position, rotation = self._pose(self._mj.MjData(self.model), state[:6])
        wxyz = np.empty(4)
        self._mj.mju_mat2Quat(wxyz, rotation.ravel())
        if wxyz[0] < 0:
            wxyz = -wxyz
        return np.r_[position, wxyz[1:], wxyz[0], state[6]].astype(np.float32)

    def committed(self, joint_targets, *, matrix_policy=False):
        """Rho rows for the exact queued joint commands: forward kinematics of each target in ``rho_tool``.

        These are what the arm will run, so the worker conditions on the executed plan rather than on its own
        earlier rows, which differ from it by the IK residual (under 2 mm and 0.02 rad, usually far less).
        """
        return np.stack([self.state(target) for target in joint_targets])

    def actions(self, rows, joint_state, *, matrix_policy=False):
        rows = np.asarray(rows, dtype=np.float64)
        if rows.ndim != 2 or not len(rows) or rows.shape[1] != 8 or not np.isfinite(rows).all():
            raise ValueError("Quaternion tool actions must be finite rows of width 8")
        joints = self._checked_state(joint_state)[:6]
        data = self._mj.MjData(self.model)
        output = np.empty((len(rows), 7), dtype=np.float32)
        for i, row in enumerate(rows):
            norm = np.linalg.norm(row[3:7])
            if abs(norm - 1) > 0.1:
                raise ValueError(f"Invalid quaternion in tool action row {i}")
            matrix = np.empty(9)
            # MuJoCo orders quaternions w, x, y, z.
            self._mj.mju_quat2Mat(matrix, np.r_[row[6], row[3:6]] / norm)
            try:
                joints = self._solve(data, row[:3], matrix.reshape(3, 3), joints,
                                     max_step=max(self.max_step, self.first_row_max_step_rad) if i == 0 else None)
            except UnreachableToolPose as error:
                if i:
                    return output[:i].copy()
                raise UnreachableToolPose(f"Tool action row {i}: {error}") from error
            output[i, :6], output[i, 6] = joints, np.clip(row[7], 0.0, 1.0)
        return output


class HolesawBimanualActions(ToolPoseActions):
    """The Holesaw training rig puts the physical working arm in the right policy slot."""

    # Row 0 is solved from the MEASURED joints, which lag the model's first command (Holesaw actions are leader
    # commands), so it needs the same generous cap as QuatToolPoseActions; the executor's per-tick cap
    # (both_arm_max_delta) limits the actual motion. Later rows keep the per-row step bound.
    first_row_max_step_rad = 1.0
    working_mount = np.array([0.29985, -0.30516, 0.30165])
    parked_state = np.array([0.06576730, 0.37384747, 0.45302705, -0.13859866,
                             0.73765181, -0.05479971, -0.65852654, 0.009264], dtype=np.float32)

    def _pose(self, data, joints):
        position, rotation = super()._pose(data, joints)
        return self.working_mount + position, rotation

    def state(self, joint_state, *, matrix_policy=True):
        return np.r_[self.parked_state, super().state(joint_state, matrix_policy=True)]

    def committed(self, joint_targets, *, matrix_policy=True):
        working = super().committed(joint_targets)
        matrix = np.empty(9)
        self._mj.mju_quat2Mat(matrix, self.parked_state[3:7].astype(np.float64))
        parked = np.r_[self.parked_state[:3], matrix, self.parked_state[-1]].astype(np.float32)
        return np.column_stack((np.broadcast_to(parked, (len(working), 13)), working))

    def actions(self, rows, joint_state, *, matrix_policy=True):
        rows = np.asarray(rows)
        if rows.ndim != 2 or rows.shape[1] != 26 or not np.isfinite(rows).all():
            raise ValueError("Bimanual Holesaw actions must have 26 finite values per row")
        return super().actions(rows[:, 13:], joint_state, matrix_policy=True)
