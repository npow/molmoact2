"""Real-robot predictive velocity-matched conveyor catching controller for YAM.

Runs against the physical bimanual YAM rig (`gello_min.env.RobotEnv`) configured
by ``examples/yam/configs/pi05_red_lid_conveyor_left.yaml``:
  - Active arm: left arm (mounted rotated 180 deg, base frame q0 ~ +3.04 rad)
  - Overhead camera: ``front_camera_rgb`` (``/dev/yam-cameras/middle-ir`` 640x480)
  - Gripper convention: normalized [0.0 = closed, 1.0 = open]

Detects the moving lid via background subtraction on the conveyor belt ROI
(or red color mask when ``use_color_mask=True``), fits belt velocity online,
smoothly accelerates along the belt to match velocity, descends & grasps while
tracking, and lifts cleanly.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple

import cv2
import mujoco
import numpy as np

logger = logging.getLogger("predictive_conveyor")

YAM_JOINT_MIN = np.array([-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944], dtype=np.float64)
YAM_JOINT_MAX = np.array([3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944], dtype=np.float64)

GRIPPER_OPEN = 0.95
GRIPPER_HOLD_LID = 0.34


@dataclass
class BeltCalib:
    """Maps overhead camera (u_px, v_px) -> robot base frame (x_base, y_base)."""

    homography: Optional[np.ndarray] = None
    # Default linear fit from captured left-arm conveyor rig (2026-11-18 middle-ir + left FK):
    #   u=481 -> y_base=0.000, dy_du=+0.00233 m/px; v=294 -> x_base=-0.476, z_grasp=-0.030.
    u0_px: float = 481.0
    v0_px: float = 294.0
    x0_base: float = -0.476
    y0_base: float = 0.0
    dy_du: float = 0.00233
    dx_dv: float = -0.00233
    z_grasp: float = -0.030
    roi_u: Tuple[int, int] = (145, 472)
    roi_v: Tuple[int, int] = (268, 330)

    def pixel_to_base_xy(self, u: float, v: float) -> Tuple[float, float]:
        if self.homography is not None:
            pt = np.array([[[float(u), float(v)]]], dtype=np.float64)
            mapped = cv2.perspectiveTransform(pt, self.homography)[0, 0]
            return float(mapped[0]), float(mapped[1])
        x_b = self.x0_base + self.dx_dv * (float(v) - self.v0_px)
        y_b = self.y0_base + self.dy_du * (float(u) - self.u0_px)
        return float(x_b), float(y_b)

    @classmethod
    def load(cls, path: Optional[Path]) -> "BeltCalib":
        if path is None or not path.is_file():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        H = np.asarray(raw["homography"], dtype=np.float64) if "homography" in raw else None
        return cls(
            homography=H,
            u0_px=float(raw.get("u0_px", cls.u0_px)),
            v0_px=float(raw.get("v0_px", cls.v0_px)),
            x0_base=float(raw.get("x0_base", cls.x0_base)),
            y0_base=float(raw.get("y0_base", cls.y0_base)),
            dy_du=float(raw.get("dy_du", cls.dy_du)),
            dx_dv=float(raw.get("dx_dv", cls.dx_dv)),
            z_grasp=float(raw.get("z_grasp", cls.z_grasp)),
            roi_u=tuple(raw.get("roi_u", cls.roi_u)),
            roi_v=tuple(raw.get("roi_v", cls.roi_v)),
        )

    def save(self, path: Path) -> None:
        payload: dict[str, Any] = {
            "u0_px": float(self.u0_px),
            "v0_px": float(self.v0_px),
            "x0_base": float(self.x0_base),
            "y0_base": float(self.y0_base),
            "dy_du": float(self.dy_du),
            "dx_dv": float(self.dx_dv),
            "z_grasp": float(self.z_grasp),
            "roi_u": list(self.roi_u),
            "roi_v": list(self.roi_v),
        }
        if self.homography is not None:
            payload["homography"] = self.homography.tolist()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img.astype(np.float32)
    return (0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]).astype(np.float32)


def detect_lid_px(
    img: np.ndarray,
    bg_gray: Optional[np.ndarray],
    calib: BeltCalib,
    *,
    use_color_mask: bool = False,
    diff_thresh: float = 28.0,
) -> Optional[Tuple[float, float, int]]:
    """Detects the lid centroid (u_px, v_px, area) inside the conveyor belt ROI."""
    u0, u1 = calib.roi_u
    v0, v1 = calib.roi_v
    roi = img[v0:v1, u0:u1]

    if use_color_mask and img.ndim == 3:
        r = roi[:, :, 0].astype(np.int16)
        g = roi[:, :, 1].astype(np.int16)
        b = roi[:, :, 2].astype(np.int16)
        mask = (r > 110) & (r > 1.55 * g) & (r > 1.55 * b)
    elif bg_gray is not None:
        diff = _to_gray(roi) - bg_gray[v0:v1, u0:u1]
        mask = diff > diff_thresh
    else:
        return None

    mask_u8 = mask.astype(np.uint8) * 255
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    best_area = 0
    best_uv = None
    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if area < 100 or area > 750 or w < 8 or h < 8 or w > 45 or h > 45:
            continue
        aspect = w / max(h, 1)
        if not 0.55 <= aspect <= 1.8:
            continue
        if area > best_area:
            best_area = area
            best_uv = (float(centroids[lbl, 0] + u0), float(centroids[lbl, 1] + v0))
    if best_uv is None:
        return None
    return best_uv[0], best_uv[1], best_area


class LidBeltTracker:
    """Online OLS tracker for (x_base, y_base(t), vy_belt) along the conveyor."""

    def __init__(self, max_history: int = 24, default_x: float = -0.476) -> None:
        self.max_history = max_history
        self.default_x = float(default_x)
        self.history: list[tuple[float, float, float]] = []
        self.t_ref: float = 0.0
        self.y_ref: float = 0.0
        self.vy: float = 0.0
        self.x_est: float = self.default_x
        self.locked: bool = False

    def reset(self) -> None:
        self.history.clear()
        self.t_ref = 0.0
        self.y_ref = 0.0
        self.vy = 0.0
        self.x_est = self.default_x
        self.locked = False

    def update(self, t: float, x_b: float, y_b: float) -> None:
        if self.locked and len(self.history) >= 5:
            px, py, _ = self.predict(t)
            if abs(y_b - py) > 0.04 or abs(x_b - px) > 0.04:
                return
        self.history.append((t, x_b, y_b))
        if len(self.history) > self.max_history:
            self.history.pop(0)
        if len(self.history) < 2:
            self.t_ref = t
            self.x_est = x_b
            self.y_ref = y_b
            return

        arr = np.asarray(self.history, dtype=np.float64)
        ts, xs, ys = arr[:, 0], arr[:, 1], arr[:, 2]
        dt_span = float(ts[-1] - ts[0])
        if dt_span < 1e-3:
            return
        t_mean = float(ts.mean())
        dt = ts - t_mean
        denom = float(np.dot(dt, dt))
        vy = float(np.dot(dt, ys - ys.mean()) / denom) if denom > 1e-9 else 0.0

        self.t_ref = t_mean
        self.y_ref = float(ys.mean())
        self.vy = max(0.0, vy)
        self.x_est = float(np.median(xs))
        rms = float(np.sqrt(np.mean((ys - (self.y_ref + self.vy * dt)) ** 2)))
        if len(self.history) >= 4 and dt_span >= 0.10 and self.vy >= 0.04 and rms < 0.015:
            self.locked = True

    def predict(self, t: float) -> Tuple[float, float, float]:
        y_pred = self.y_ref + self.vy * (t - self.t_ref)
        x_pred = float(np.clip(self.x_est, -0.54, -0.36))
        return x_pred, float(y_pred), self.vy


def _smoothstep(u: float) -> Tuple[float, float]:
    u = float(np.clip(u, 0.0, 1.0))
    u2 = u * u
    u3 = u2 * u
    s = 10.0 * u3 - 15.0 * u2 * u2 + 6.0 * u3 * u2
    ds = 30.0 * u2 - 60.0 * u3 + 30.0 * u2 * u2
    return s, ds


class YamRealKinematics:
    """Fast MuJoCo FK/IK solver for a real YAM + linear_4310 arm."""

    def __init__(self) -> None:
        from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

        xml_path = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.site_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site"))
        self.lo = self.model.jnt_range[:6, 0].copy()
        self.hi = self.model.jnt_range[:6, 1].copy()

        # Angled approach matching dataset ep0..86 (-X_base & -Z_base), keeping J6 near 0.
        z_app = np.array([-0.70, 0.0, -0.71], dtype=np.float64)
        z_app /= np.linalg.norm(z_app)
        x_hint = np.array([0.71, 0.0, -0.70], dtype=np.float64)
        x_site = x_hint - np.dot(x_hint, z_app) * z_app
        x_site /= np.linalg.norm(x_site)
        y_site = np.cross(z_app, x_site)
        self.R_grasp = np.column_stack([x_site, y_site, z_app])

    def fk(self, q6: np.ndarray) -> np.ndarray:
        self.data.qpos[:6] = np.asarray(q6, dtype=np.float64)[:6]
        mujoco.mj_kinematics(self.model, self.data)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        T[:3, 3] = self.data.site_xpos[self.site_id]
        return T

    def solve_ik(
        self,
        target_pos: np.ndarray,
        q_seed: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        iters: int = 16,
    ) -> np.ndarray:
        rot_goal = self.R_grasp if target_rot is None else target_rot
        q = np.clip(np.asarray(q_seed, dtype=np.float64)[:6], self.lo, self.hi).copy()
        q_err = np.zeros(4, dtype=np.float64)
        vel_body = np.zeros(3, dtype=np.float64)
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        eye6 = np.eye(6, dtype=np.float64)

        for _ in range(iters):
            self.data.qpos[:6] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            p_cur = self.data.site_xpos[self.site_id]
            r_cur = self.data.site_xmat[self.site_id].reshape(3, 3)
            e_p = target_pos - p_cur
            mujoco.mju_mat2Quat(q_err, (r_cur.T @ rot_goal).ravel())
            mujoco.mju_quat2Vel(vel_body, q_err, 1.0)
            e_r = r_cur @ vel_body
            if float(np.linalg.norm(e_p)) < 3e-4 and float(np.linalg.norm(e_r)) < 3e-3:
                break
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
            J = np.vstack([jacp[:, :6], jacr[:, :6]])
            err6 = np.concatenate([e_p, 0.5 * e_r])
            dq = np.linalg.solve(J.T @ J + 1e-3 * eye6, J.T @ err6)
            norm = float(np.linalg.norm(dq))
            if norm > 0.20:
                dq *= 0.20 / norm
            q = np.clip(q + dq, self.lo, self.hi)
        return q


class RealYamPredictiveController:
    """Predictive conveyor-catching state machine for the real rotated-180 left YAM arm."""

    def __init__(
        self,
        calib: BeltCalib,
        *,
        use_color_mask: bool = False,
        auto_reset: bool = True,
        lead_time_s: float = 0.020,
    ) -> None:
        self.calib = calib
        self.use_color_mask = use_color_mask
        self.auto_reset = auto_reset
        self.lead_time_s = float(lead_time_s)
        self.kin = YamRealKinematics()
        self.tracker = LidBeltTracker(default_x=calib.x0_base)
        self.bg_gray: Optional[np.ndarray] = None
        self._prev_roi_gray: Optional[np.ndarray] = None

        # Catch zone along belt in rotated-180 left arm base frame:
        # Lid enters from left (y_base < 0) and travels toward +y_base.
        self.y_wait = 0.01
        self.y_max_catch = 0.28
        self.z_hover = calib.z_grasp + 0.075
        self.z_grasp = calib.z_grasp
        self.z_lift = calib.z_grasp + 0.165

        self.phase = "OBSERVE"
        self.q_des = self.ready_joints(
            np.array([3.11, 1.88, 1.02, 0.07, 0.0, 0.0], dtype=np.float64)
        )[:6].copy()
        self.grip_cmd = GRIPPER_OPEN
        self.t_accel_start = 0.0
        self.t_accel_dur = 0.14
        self.y_accel_start = self.y_wait
        self.t_phase = 0.0
        self.lift_y0 = 0.0
        self.lift_vy0 = 0.0
        self.hold_pos = np.array([calib.x0_base, 0.16, self.z_lift], dtype=np.float64)
        self.catches = 0

    def ready_joints(self, q_seed: Optional[np.ndarray] = None) -> np.ndarray:
        """Return 7-DOF wait pose above the start of the catching zone."""
        seed = (
            np.array([3.11, 1.88, 1.02, 0.07, 0.0, 0.0], dtype=np.float64)
            if q_seed is None
            else np.asarray(q_seed, dtype=np.float64)[:6]
        )
        # Seed IK via hover posture so solutions from parked arm rest ([3.05, 0, 0, ...]) stay elbow-up.
        hover_seed = np.array([3.11, 1.88, 1.02, 0.07, 0.0, 0.0], dtype=np.float64)
        p_wait = np.array([self.calib.x0_base, self.y_wait, self.z_hover], dtype=np.float64)
        q6 = self.kin.solve_ik(p_wait, q_seed=hover_seed if np.linalg.norm(seed - hover_seed) > 1.2 else seed, iters=24)
        return np.append(q6, GRIPPER_OPEN).astype(np.float32)

    def capture_background(self, frames: list[np.ndarray]) -> None:
        grays = [_to_gray(f) for f in frames]
        self.bg_gray = np.median(np.stack(grays, axis=0), axis=0)
        logger.info("Captured belt background reference from %d frames", len(frames))

    def reset_episode(self, q_actual: np.ndarray) -> None:
        self.tracker.reset()
        self._prev_roi_gray = None
        self.phase = "OBSERVE"
        self.q_des = np.asarray(q_actual, dtype=np.float64)[:6].copy()
        self.grip_cmd = GRIPPER_OPEN

    def step(
        self,
        t: float,
        front_img: Optional[np.ndarray],
        q_actual: np.ndarray,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        """Returns 7-DOF target [joint1..6, gripper_0_to_1] and diagnostics."""
        if front_img is not None and self.phase in ("OBSERVE", "PRE_POSITION"):
            u0, u1 = self.calib.roi_u
            v0, v1 = self.calib.roi_v
            roi_gray = _to_gray(front_img[v0:v1, u0:u1])
            fresh_frame = (
                self._prev_roi_gray is None
                or float(np.mean(np.abs(roi_gray - self._prev_roi_gray))) >= 0.25
            )
            if fresh_frame:
                self._prev_roi_gray = roi_gray
                det = detect_lid_px(
                    front_img,
                    self.bg_gray,
                    self.calib,
                    use_color_mask=self.use_color_mask,
                )
                if det is not None:
                    x_b, y_b = self.calib.pixel_to_base_xy(det[0], det[1])
                    if y_b < self.y_wait - 0.04:
                        self.tracker.update(t, x_b, y_b)

        p_ee = self.kin.fk(q_actual)[:3, 3]
        x_pred, y_raw, vy_est = self.tracker.predict(t)
        lead = self.lead_time_s if self.phase in ("SYNC_ACCEL", "DESCEND", "GRASP") else 0.0
        y_pred = y_raw + vy_est * lead

        if self.phase == "OBSERVE":
            p_des = np.array([x_pred, self.y_wait, self.z_hover])
            self.grip_cmd = GRIPPER_OPEN
            if self.tracker.locked:
                self.t_accel_dur = float(np.clip(0.12 + 0.02 * vy_est, 0.12, 0.18))
                y_trigger = self.y_wait - 0.5 * vy_est * self.t_accel_dur
                if y_raw >= y_trigger:
                    self.phase = "SYNC_ACCEL"
                    self.t_accel_start = t
                    self.y_accel_start = float(p_ee[1])
                else:
                    self.phase = "PRE_POSITION"

        if self.phase == "PRE_POSITION":
            self.t_accel_dur = float(np.clip(0.12 + 0.02 * vy_est, 0.12, 0.18))
            y_trigger = self.y_wait - 0.5 * vy_est * self.t_accel_dur
            p_des = np.array([x_pred, self.y_wait, self.z_hover])
            self.grip_cmd = GRIPPER_OPEN
            if y_raw >= y_trigger:
                self.phase = "SYNC_ACCEL"
                self.t_accel_start = t
                self.y_accel_start = float(p_ee[1])

        if self.phase == "SYNC_ACCEL":
            tau = (t - self.t_accel_start) / self.t_accel_dur
            if tau >= 1.0:
                self.phase = "DESCEND"
                self.t_phase = t
            else:
                s, _ = _smoothstep(tau)
                y_des = (1.0 - s) * self.y_accel_start + s * y_pred
                z_u = max(0.0, (tau - 0.30) / 0.70)
                zs, _ = _smoothstep(z_u)
                z_mid = self.z_hover - 0.50 * (self.z_hover - self.z_grasp)
                z_des = self.z_hover + zs * (z_mid - self.z_hover)
                p_des = np.array([x_pred, y_des, z_des])
                self.grip_cmd = GRIPPER_OPEN

        if self.phase == "DESCEND":
            desc_dur = float(np.clip(0.12 - 0.025 * vy_est, 0.06, 0.12))
            tau = (t - self.t_phase) / desc_dur
            z_mid = self.z_hover - 0.50 * (self.z_hover - self.z_grasp)
            if tau >= 1.0:
                self.phase = "GRASP"
                self.t_phase = t
            else:
                s, _ = _smoothstep(tau)
                z_des = z_mid + s * (self.z_grasp - z_mid)
                p_des = np.array([x_pred, y_pred, z_des])
                self.grip_cmd = GRIPPER_OPEN

        if self.phase == "GRASP":
            grasp_dur = 0.09
            tau = (t - self.t_phase) / grasp_dur
            p_des = np.array([x_pred, min(y_pred, self.y_max_catch), self.z_grasp])
            self.grip_cmd = GRIPPER_HOLD_LID
            if tau >= 1.0 or y_pred >= self.y_max_catch:
                self.phase = "LIFT"
                self.t_phase = t
                self.lift_y0 = float(p_des[1])
                self.lift_vy0 = vy_est

        if self.phase == "LIFT":
            lift_dur = 0.35
            tau = (t - self.t_phase) / lift_dur
            if tau >= 1.0:
                self.phase = "HOLD"
                self.t_phase = t
                self.catches += 1
                y_end = min(self.y_max_catch + 0.02, self.lift_y0 + 0.06)
                self.hold_pos = np.array([x_pred, y_end, self.z_lift])
                p_des = self.hold_pos
            else:
                s, _ = _smoothstep(tau)
                u_dec = min(1.0, (t - self.t_phase) / 0.18)
                y_off = self.lift_vy0 * 0.18 * (u_dec - (2.5 * u_dec**4 - 3.0 * u_dec**5 + u_dec**6))
                y_des = min(self.y_max_catch + 0.02, self.lift_y0 + y_off)
                z_des = self.z_grasp + s * (self.z_lift - self.z_grasp)
                p_des = np.array([x_pred, y_des, z_des])
            self.grip_cmd = GRIPPER_HOLD_LID

        if self.phase == "HOLD":
            p_des = self.hold_pos
            self.grip_cmd = GRIPPER_HOLD_LID
            if self.auto_reset and (t - self.t_phase) >= 1.5:
                self.reset_episode(q_actual)

        # Solve IK seeded from current target for smooth trajectory.
        seed = self.q_des if np.linalg.norm(self.q_des - q_actual[:6]) < 0.6 else q_actual[:6]
        self.q_des = self.kin.solve_ik(p_des, q_seed=seed, iters=16)
        q7 = np.append(self.q_des, self.grip_cmd).astype(np.float32)
        return q7, {
            "phase": self.phase,
            "est_vx_mps": self.tracker.vy,
            "p_des": p_des.tolist(),
            "locked": self.tracker.locked,
        }


def run_predictive_calibration(env, mask, calib_path: Path) -> BeltCalib:
    """Interactive 4-point belt-to-camera homography calibration on the real rig."""
    kin = YamRealKinematics()
    ep_slice = slice(0, 7) if mask.active_arm_side == "left" else slice(7, 14)
    pts_uv: list[list[float]] = []
    pts_xy: list[list[float]] = []
    zs: list[float] = []

    print("\n=== Conveyor Calibration (4 points on belt) ===")
    print("Move/place the active arm's fingertips at 4 corners of the belt catching zone.")
    for idx in range(4):
        input(f"  Point {idx + 1}/4: position gripper at belt point and press Enter...")
        obs = env.get_obs()
        q7 = mask.validate_state(obs["joint_positions"], name="calib_state")[ep_slice]
        T = kin.fk(q7[:6])
        x_b, y_b, z_b = float(T[0, 3]), float(T[1, 3]), float(T[2, 3])
        zs.append(z_b)
        pts_xy.append([x_b, y_b])

        img = obs["front_camera_rgb"]
        # Let user enter pixel coords from live view or auto-click if GUI is available
        raw_uv = input(f"    FK=({x_b:+.3f}, {y_b:+.3f}, {z_b:+.3f}). Enter 'u,v' in front_camera (640x480): ").strip()
        u_s, v_s = raw_uv.replace(" ", "").split(",")
        pts_uv.append([float(u_s), float(v_s)])

    H, _ = cv2.findHomography(np.asarray(pts_uv, dtype=np.float64), np.asarray(pts_xy, dtype=np.float64))
    if H is None:
        raise RuntimeError("Failed to compute belt homography from provided points")
    calib = BeltCalib(homography=H, z_grasp=float(np.mean(zs)))
    calib.save(calib_path)
    print(f"Saved conveyor calibration to {calib_path}")
    return calib


def run_predictive_rollout(
    env,
    mask,
    *,
    control_hz: float = 60.0,
    seconds: float = 60.0,
    calib_path: Optional[Path] = None,
    use_color_mask: bool = False,
    auto_reset: bool = True,
    dry_run: bool = False,
    max_joint_step: float = 0.12,
) -> dict[str, Any]:
    """Closed-loop predictive conveyor catching rollout on the real YAM rig."""
    ep_slice = slice(0, 7) if mask.active_arm_side == "left" else slice(7, 14)
    calib = BeltCalib.load(calib_path)
    ctrl = RealYamPredictiveController(
        calib,
        use_color_mask=use_color_mask,
        auto_reset=auto_reset,
    )

    print("Capturing empty belt background (keep lid off the belt for ~0.5s)...")
    bg_frames = []
    for _ in range(12):
        obs = env.get_obs()
        bg_frames.append(obs["front_camera_rgb"])
        time.sleep(0.03)
    ctrl.capture_background(bg_frames)

    measured0 = mask.validate_state(env.get_robot_state()["joint_positions"], name="start_state")
    ctrl.reset_episode(measured0[ep_slice])

    dt = 1.0 / max(control_hz, 1.0)
    t_start = time.monotonic()
    deadline = t_start + seconds
    ticks = 0
    last_log_t = 0.0

    while time.monotonic() < deadline:
        t_tick = time.monotonic()
        t_rel = t_tick - t_start
        obs = env.get_obs()
        measured = mask.validate_state(obs["joint_positions"], name="measured_state")
        q7_act = measured[ep_slice]

        q7_cmd, info = ctrl.step(t_rel, obs.get("front_camera_rgb"), q7_act)
        # Safety clamp per-tick joint delta and physical joint limits.
        delta = np.clip(q7_cmd[:6] - q7_act[:6], -max_joint_step, max_joint_step)
        q7_safe = q7_cmd.copy()
        q7_safe[:6] = np.clip(q7_act[:6] + delta, YAM_JOINT_MIN, YAM_JOINT_MAX)
        q7_safe[6] = float(np.clip(q7_safe[6], 0.0, 1.0))

        if not dry_run:
            target = measured.copy()
            target[ep_slice] = q7_safe
            env.step_command_only(mask.command_target(target, measured), wait=False)

        if t_rel - last_log_t >= 0.5:
            last_log_t = t_rel
            logger.info(
                "t=%.2fs phase=%s locked=%s est_vx=%.2f m/s p_des=%s dry_run=%s",
                t_rel,
                info["phase"],
                info["locked"],
                info["est_vx_mps"],
                [round(v, 3) for v in info["p_des"]],
                dry_run,
            )

        ticks += 1
        sleep_s = (t_tick + dt) - time.monotonic()
        if sleep_s > 0.0:
            time.sleep(sleep_s)

    return {
        "ticks": ticks,
        "catches": ctrl.catches,
        "final_phase": ctrl.phase,
        "dry_run": dry_run,
    }
