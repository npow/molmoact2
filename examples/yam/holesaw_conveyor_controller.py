"""Vision-guided predictive conveyor pick-and-place controller for the YAM holesaw task.

Built directly for the physical left-arm YAM holesaw conveyor rig
(``examples/yam/configs/pi05_holesaw_conveyor_left.yaml`` +
``examples/yam/configs/yam_holesaw_kinematics.urdf``):
  - Overhead camera: Intel RealSense D435 (640x480 RGB)
  - Conveyor runs left-to-right in image space (base-frame ``-Y`` direction,
    ``x_base ~ +0.46..+0.57 m``, ``y_base`` decreasing from ``+0.48 m`` to ``-0.14 m``)
  - Targets: white metal hole saws (various diameters, upright or lying flat)
  - Distractors: high-saturation yellow 3D-printed triangles (rejected by HSV + ring guard)
  - Drop target: open red foam-lined case at ``x ~ +0.285 m, y ~ +0.075 m``
  - Grasp verification: measured gripper aperture ``>= 0.06`` confirms capture

Default belt-to-base affine calibration and grasp/drop orientations were fitted from
real robot grasps recorded in ``yam_eval_runs/rerun/*.rrd`` (median 8.0 mm inlier
error across multiple runs and belt speeds).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import mujoco
import numpy as np

logger = logging.getLogger("holesaw_conveyor")

DEFAULT_URDF_PATH = Path(__file__).resolve().parent / "configs" / "yam_holesaw_kinematics.urdf"
JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")

YAM_JOINT_MIN = np.array([-2.61799, 0.0, 0.0, -1.69297, -1.5708, -2.0944], dtype=np.float64)
YAM_JOINT_MAX = np.array([3.14159, 3.66519, 3.14159, 1.5708, 1.5708, 2.0944], dtype=np.float64)

GRIPPER_OPEN = 0.95
GRIPPER_CLOSE = 0.02
GRIPPER_VERIFY_MIN_APERTURE = 0.06

# Hover seed posture matching measured holesaw rollouts (left arm facing conveyor).
DEFAULT_HOVER_SEED_Q6 = np.array([-0.05, 2.15, 1.62, -0.50, -0.72, -1.24], dtype=np.float64)

# SVD-orthonormalized median grasp_site orientations from real holesaw rollouts.
DEFAULT_R_GRASP = np.array(
    [
        [-0.21589, -0.85229, 0.47644],
        [-0.61092, 0.49854, 0.61500],
        [-0.76169, -0.15830, -0.62831],
    ],
    dtype=np.float64,
)

DEFAULT_R_DROP = np.array(
    [
        [-0.54655, -0.75728, 0.35749],
        [-0.41236, 0.61493, 0.67218],
        [-0.72886, 0.21997, -0.64836],
    ],
    dtype=np.float64,
)

# Default affine map [u, v_contact, 1] @ M -> [x_base, y_base] fitted from real
# grasps in yam_eval_runs/rerun/*.rrd (v_contact = footprint center y + h - 0.35*min(w,h)).
DEFAULT_AFFINE_UV_TO_XY = np.array(
    [
        [-0.00029840, -0.00094310],
        [-0.00114488, 0.00032505],
        [0.76015143, 0.49669752],
    ],
    dtype=np.float64,
)

# Matte black belt surface polygon in 640x480 overhead RGB camera (inset ~8px from silver rails).
DEFAULT_BELT_POLY_640X480 = (
    (0, 112),
    (639, 10),
    (639, 122),
    (0, 232),
)

# Drop offsets inside the open red case so consecutive hole saws spread out.
DEFAULT_DROP_OFFSETS: Tuple[Tuple[float, float], ...] = (
    (0.000, 0.000),
    (0.025, -0.020),
    (-0.025, 0.015),
    (0.018, 0.020),
)


@dataclass
class HolesawBeltCalib:
    """Calibration mapping overhead camera pixels (u, v_bot) -> robot base (x, y)."""

    affine_uv_to_xy: np.ndarray = field(default_factory=lambda: DEFAULT_AFFINE_UV_TO_XY.copy())
    homography: Optional[np.ndarray] = None
    belt_poly: Tuple[Tuple[int, int], ...] = DEFAULT_BELT_POLY_640X480
    ref_size: Tuple[int, int] = (640, 480)  # (width, height)

    # New tracks only spawn in the far-upstream entry zone (left of arm reach);
    # existing tracks continue updating up to u_track_max.
    u_spawn_max: float = 320.0
    u_track_max: float = 445.0

    # Catch workspace in left-arm base frame.
    x_bounds: Tuple[float, float] = (0.44, 0.58)
    x_hover_default: float = 0.51
    y_wait: float = 0.06
    y_wait_max: float = 0.25
    y_min_catch: float = -0.24
    z_grasp: float = -0.016
    z_hover: float = 0.095
    z_lift: float = 0.135

    # Red case drop target in left-arm base frame.
    drop_center: Tuple[float, float, float] = (0.285, 0.075, 0.040)
    drop_offsets: Tuple[Tuple[float, float], ...] = DEFAULT_DROP_OFFSETS

    def scaled_belt_poly(self, width: int, height: int) -> np.ndarray:
        sx = float(width) / float(self.ref_size[0])
        sy = float(height) / float(self.ref_size[1])
        pts = [(int(round(u * sx)), int(round(v * sy))) for u, v in self.belt_poly]
        return np.asarray(pts, dtype=np.int32)

    def pixel_to_base_xy(self, u: float, v_bot: float, *, width: int = 640, height: int = 480) -> Tuple[float, float]:
        # Normalize pixel coordinates back to ref_size (640x480) before mapping.
        u_ref = float(u) * (float(self.ref_size[0]) / float(width))
        v_ref = float(v_bot) * (float(self.ref_size[1]) / float(height))
        if self.homography is not None:
            pt = np.array([[[u_ref, v_ref]]], dtype=np.float64)
            mapped = cv2.perspectiveTransform(pt, self.homography)[0, 0]
            return float(mapped[0]), float(mapped[1])
        vec = np.array([u_ref, v_ref, 1.0], dtype=np.float64)
        xy = vec @ self.affine_uv_to_xy
        return float(xy[0]), float(xy[1])

    def base_xy_to_pixel(self, x_b: float, y_b: float, *, width: int = 640, height: int = 480) -> Tuple[float, float]:
        """Inverse mapping (base xy -> image u, v_bot) for debug overlays."""
        if self.homography is not None:
            H_inv = np.linalg.inv(self.homography)
            pt = np.array([[[float(x_b), float(y_b)]]], dtype=np.float64)
            uv = cv2.perspectiveTransform(pt, H_inv)[0, 0]
            u_ref, v_ref = float(uv[0]), float(uv[1])
        else:
            A = self.affine_uv_to_xy[:2, :].T  # 2x2
            b = self.affine_uv_to_xy[2, :]  # 2
            uv = np.linalg.solve(A, np.array([x_b, y_b], dtype=np.float64) - b)
            u_ref, v_ref = float(uv[0]), float(uv[1])
        return (
            u_ref * (float(width) / float(self.ref_size[0])),
            v_ref * (float(height) / float(self.ref_size[1])),
        )

    def drop_pose_for_index(self, catch_index: int) -> np.ndarray:
        dx, dy = self.drop_offsets[catch_index % len(self.drop_offsets)]
        return np.array(
            [self.drop_center[0] + dx, self.drop_center[1] + dy, self.drop_center[2]],
            dtype=np.float64,
        )

    @classmethod
    def load(cls, path: Optional[Path]) -> "HolesawBeltCalib":
        if path is None or not Path(path).is_file():
            return cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        H = np.asarray(raw["homography"], dtype=np.float64) if raw.get("homography") is not None else None
        aff = (
            np.asarray(raw["affine_uv_to_xy"], dtype=np.float64)
            if raw.get("affine_uv_to_xy") is not None
            else DEFAULT_AFFINE_UV_TO_XY.copy()
        )
        poly = tuple((int(p[0]), int(p[1])) for p in raw.get("belt_poly", DEFAULT_BELT_POLY_640X480))
        drop_c = tuple(float(v) for v in raw.get("drop_center", cls.drop_center))
        drop_offs = tuple((float(o[0]), float(o[1])) for o in raw.get("drop_offsets", DEFAULT_DROP_OFFSETS))
        return cls(
            affine_uv_to_xy=aff,
            homography=H,
            belt_poly=poly,
            ref_size=tuple(int(v) for v in raw.get("ref_size", (640, 480))),
            u_spawn_max=float(raw.get("u_spawn_max", cls.u_spawn_max)),
            u_track_max=float(raw.get("u_track_max", cls.u_track_max)),
            x_bounds=tuple(float(v) for v in raw.get("x_bounds", cls.x_bounds)),
            x_hover_default=float(raw.get("x_hover_default", cls.x_hover_default)),
            y_wait=float(raw.get("y_wait", cls.y_wait)),
            y_min_catch=float(raw.get("y_min_catch", cls.y_min_catch)),
            z_grasp=float(raw.get("z_grasp", cls.z_grasp)),
            z_hover=float(raw.get("z_hover", cls.z_hover)),
            z_lift=float(raw.get("z_lift", cls.z_lift)),
            drop_center=(drop_c[0], drop_c[1], drop_c[2]),
            drop_offsets=drop_offs,
        )

    def save(self, path: Path) -> None:
        payload: Dict[str, Any] = {
            "affine_uv_to_xy": self.affine_uv_to_xy.tolist(),
            "homography": self.homography.tolist() if self.homography is not None else None,
            "belt_poly": [list(p) for p in self.belt_poly],
            "ref_size": list(self.ref_size),
            "u_spawn_max": float(self.u_spawn_max),
            "u_track_max": float(self.u_track_max),
            "x_bounds": list(self.x_bounds),
            "x_hover_default": float(self.x_hover_default),
            "y_wait": float(self.y_wait),
            "y_min_catch": float(self.y_min_catch),
            "z_grasp": float(self.z_grasp),
            "z_hover": float(self.z_hover),
            "z_lift": float(self.z_lift),
            "drop_center": list(self.drop_center),
            "drop_offsets": [list(o) for o in self.drop_offsets],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


@dataclass
class HolesawDetection:
    """Single white hole-saw candidate detected on the conveyor belt."""

    u: float
    v_centroid: float
    v_bot: float
    area_px: int
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    aspect: float
    major_angle_deg: float
    is_lying_down: bool
    x_base: float
    y_base: float


def detect_holesaws(
    img_rgb: np.ndarray,
    calib: HolesawBeltCalib,
    *,
    v_min: int = 165,
    s_max: int = 75,
    min_area: int = 260,
    max_area: int = 8500,
    max_yellow_ring_frac: float = 0.12,
    u_max: Optional[float] = None,
) -> List[HolesawDetection]:
    """Detect white metal hole saws on the matte black belt while rejecting yellow distractors."""
    if img_rgb is None or img_rgb.ndim != 3:
        return []
    h_im, w_im = img_rgb.shape[:2]
    scale = (w_im * h_im) / float(calib.ref_size[0] * calib.ref_size[1])

    belt_mask = np.zeros((h_im, w_im), dtype=np.uint8)
    cv2.fillPoly(belt_mask, [calib.scaled_belt_poly(w_im, h_im)], 255)

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Bright low-saturation white pixels inside the belt surface.
    white = ((val >= v_min) & (sat <= s_max) & (belt_mask > 0)).astype(np.uint8) * 255
    # High-saturation yellow pixels (distractor triangles) for specular-glare guard.
    yellow = ((hue >= 12) & (hue <= 42) & (sat >= 100) & (val >= 100) & (belt_mask > 0)).astype(np.uint8)

    # Close small gaps (side slots, text logos, or thin dangling cable crossing a white saw).
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, k_close)
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, k_open)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(white, connectivity=8)
    dets: List[HolesawDetection] = []
    min_a = max(40, int(round(min_area * scale)))
    max_a = int(round(max_area * scale))
    max_dim = int(round(145 * np.sqrt(scale)))
    min_dim = max(6, int(round(12 * np.sqrt(scale))))
    u_limit = calib.u_track_max if u_max is None else float(u_max)
    u_limit_scaled = u_limit * (float(w_im) / float(calib.ref_size[0]))

    for lbl in range(1, num_labels):
        area = int(stats[lbl, cv2.CC_STAT_AREA])
        x = int(stats[lbl, cv2.CC_STAT_LEFT])
        y = int(stats[lbl, cv2.CC_STAT_TOP])
        w = int(stats[lbl, cv2.CC_STAT_WIDTH])
        h = int(stats[lbl, cv2.CC_STAT_HEIGHT])
        if area < min_a or area > max_a:
            continue
        if w < min_dim or h < min_dim or w > max_dim or h > max_dim:
            continue

        u_c = float(centroids[lbl, 0])
        v_c = float(centroids[lbl, 1])
        if u_c > u_limit_scaled:
            continue

        # Specular-glare guard: reject white glints surrounded by yellow plastic.
        pad = 6
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w_im, x + w + pad), min(h_im, y + h + pad)
        yellow_frac = float(np.mean(yellow[y0:y1, x0:x1]))
        if yellow_frac > max_yellow_ring_frac:
            continue

        # Fit oriented bounding box on component mask for pose/aspect estimation.
        comp_crop = (labels[y : y + h, x : x + w] == lbl).astype(np.uint8)
        contours, _ = cv2.findContours(comp_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        rect = cv2.minAreaRect(contours[0])
        (_, _), (rw, rh), ang = rect
        long_side = max(rw, rh, 1e-3)
        short_side = max(min(rw, rh), 1e-3)
        aspect = float(long_side / short_side)
        major_angle = float(ang if rw >= rh else ang + 90.0)
        # Upright saws are taller than wide (h >= w) under the oblique overhead view;
        # lying-down saws are either horizontally wider (w > 1.28 * h) or strongly elongated.
        is_lying = (w > 1.28 * h) or (aspect >= 1.85)

        # Inset from bottom edge by the foreshortened base radius so wide and narrow
        # cylinders alike map to their belt footprint center rather than their front rim.
        v_bot = float(y + h - 0.35 * min(w, h))
        x_b, y_b = calib.pixel_to_base_xy(u_c, v_bot, width=w_im, height=h_im)
        dets.append(
            HolesawDetection(
                u=u_c,
                v_centroid=v_c,
                v_bot=v_bot,
                area_px=area,
                bbox=(x, y, w, h),
                aspect=aspect,
                major_angle_deg=major_angle,
                is_lying_down=is_lying,
                x_base=x_b,
                y_base=y_b,
            )
        )

    # Sort left-to-right (smallest u first).
    dets.sort(key=lambda d: d.u)
    return dets


@dataclass
class HolesawTrack:
    """Constant-velocity OLS tracker for one hole saw moving along the belt."""

    track_id: int
    max_history: int = 36
    # History entries: (t, x_base, y_base, u, v_bot, aspect, major_angle_deg, is_lying)
    history: List[Tuple[float, float, float, float, float, float, float, bool]] = field(default_factory=list)
    t_ref: float = 0.0
    x_ref: float = 0.0
    y_ref: float = 0.0
    vx: float = 0.0
    vy: float = -0.095  # default prior (~9.5 cm/s in -Y)
    u_ref: float = 0.0
    v_ref: float = 0.0
    du_dt: float = 95.0
    dv_dt: float = -18.0
    rms_m: float = 1.0
    locked: bool = False
    consumed: bool = False
    last_seen_t: float = 0.0

    def add_observation(self, t: float, det: HolesawDetection) -> None:
        self.history.append(
            (t, det.x_base, det.y_base, det.u, det.v_bot, det.aspect, det.major_angle_deg, det.is_lying_down)
        )
        if len(self.history) > self.max_history:
            self.history.pop(0)
        self.last_seen_t = t

        if len(self.history) == 1:
            self.t_ref = t
            self.x_ref = det.x_base
            self.y_ref = det.y_base
            self.u_ref = det.u
            self.v_ref = det.v_bot
            return

        arr = np.asarray([(h[0], h[1], h[2], h[3], h[4]) for h in self.history], dtype=np.float64)
        ts, xs, ys, us, vs = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
        dt_span = float(ts[-1] - ts[0])
        if dt_span < 1e-3:
            return
        t_mean = float(ts.mean())
        dt = ts - t_mean
        denom = float(np.dot(dt, dt))
        if denom < 1e-9:
            return

        vx = float(np.dot(dt, xs - xs.mean()) / denom)
        vy = float(np.dot(dt, ys - ys.mean()) / denom)
        du = float(np.dot(dt, us - us.mean()) / denom)
        dv = float(np.dot(dt, vs - vs.mean()) / denom)

        self.t_ref = t_mean
        self.x_ref = float(np.median(xs))
        self.y_ref = float(ys.mean())
        self.vx = vx
        self.vy = vy
        self.u_ref = float(us.mean())
        self.v_ref = float(vs.mean())
        self.du_dt = du
        self.dv_dt = dv

        y_res = ys - (self.y_ref + self.vy * dt)
        x_res = xs - (self.x_ref + self.vx * dt)
        self.rms_m = float(np.sqrt(np.mean(x_res**2 + y_res**2)))
        dy_travel = float(ys[0] - ys[-1])  # positive when moving along -Y conveyor direction

        # Continuously evaluate lock condition while observed upstream so a briefly
        # wiggling stationary object never stays latched once it stops moving.
        # Supports belt speeds from 0.025 m/s up to 1.50 m/s (15x nominal).
        min_pts = 4 if self.vy <= -0.25 else 6
        min_span = 0.08 if self.vy <= -0.25 else 0.25
        self.locked = bool(
            len(self.history) >= min_pts
            and dt_span >= min_span
            and dy_travel >= 0.025
            and (-1.50 <= self.vy <= -0.025)
            and (abs(self.vx) <= max(0.045, 0.12 * abs(self.vy)))
            and self.rms_m < 0.020
        )

    def predict_base(self, t: float) -> Tuple[float, float, float, float]:
        dt = float(t - self.t_ref)
        # Belt moves almost purely along -Y; damp tiny lateral vx drift during extrapolation.
        x_pred = self.x_ref + float(np.clip(self.vx, -0.015, 0.015)) * dt
        y_pred = self.y_ref + self.vy * dt
        return float(x_pred), float(y_pred), float(self.vx), float(self.vy)

    def predict_pixel(self, t: float) -> Tuple[float, float]:
        dt = float(t - self.t_ref)
        return float(self.u_ref + self.du_dt * dt), float(self.v_ref + self.dv_dt * dt)

    @property
    def is_lying_down(self) -> bool:
        if not self.history:
            return False
        recent = self.history[-min(5, len(self.history)) :]
        return sum(1 for h in recent if h[7]) > len(recent) // 2

    @property
    def median_major_angle_deg(self) -> float:
        if not self.history:
            return 0.0
        recent = [h[6] for h in self.history[-min(5, len(self.history)) :]]
        return float(np.median(recent))


class MultiHolesawTracker:
    """Multi-object tracker associating upstream hole saws across camera frames."""

    def __init__(
        self,
        calib: HolesawBeltCalib,
        *,
        assoc_gate_m: float = 0.055,
        stale_timeout_s: float = 1.6,
    ) -> None:
        self.calib = calib
        self.assoc_gate_m = float(assoc_gate_m)
        self.stale_timeout_s = float(stale_timeout_s)
        self.tracks: List[HolesawTrack] = []
        self._next_id = 1

    def reset(self) -> None:
        self.tracks.clear()

    def update(self, t: float, detections: Sequence[HolesawDetection]) -> List[HolesawTrack]:
        """Associate detections to active tracks and create new tracks in upstream zone."""
        unmatched_dets = list(detections)
        used_track_ids = set()

        # Greedy nearest-neighbor matching prioritized by locked tracks first.
        ordered_tracks = sorted(self.tracks, key=lambda tr: (not tr.locked, -len(tr.history)))
        for tr in ordered_tracks:
            dt = t - tr.last_seen_t
            if dt <= 0.0 or dt > self.stale_timeout_s:
                continue
            best_idx = None
            if len(tr.history) == 1:
                # Speed unknown on first association: search a corridor along -Y up to 1.5 m/s.
                _, x0, y0, *_ = tr.history[0]
                max_dy_down = 1.60 * dt + 0.030
                best_score = 1e9
                for idx, det in enumerate(unmatched_dets):
                    dx = abs(det.x_base - x0)
                    dy = det.y_base - y0  # expected negative (moving along -Y)
                    if dx <= 0.045 and (-max_dy_down <= dy <= 0.025):
                        score = dx + 0.35 * abs(dy)
                        if score < best_score:
                            best_score = score
                            best_idx = idx
            else:
                px, py, _, vy = tr.predict_base(t)
                gate = max(self.assoc_gate_m, 0.25 * abs(vy) * dt + 0.030)
                best_dist = gate
                for idx, det in enumerate(unmatched_dets):
                    dist = float(np.hypot(det.x_base - px, det.y_base - py))
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = idx
            if best_idx is not None:
                tr.add_observation(t, unmatched_dets.pop(best_idx))
                used_track_ids.add(tr.track_id)

        # Spawn new tracks ONLY in the far-upstream entry zone (u <= u_spawn_max) so the
        # robot's own white wrist camera mount entering from the right can never spawn a track.
        for det in unmatched_dets:
            if det.u <= self.calib.u_spawn_max and det.y_base > self.calib.y_wait + 0.08:
                tr = HolesawTrack(track_id=self._next_id)
                self._next_id += 1
                tr.add_observation(t, det)
                self.tracks.append(tr)

        # Prune tracks that exited well past the catch zone or timed out before locking.
        kept: List[HolesawTrack] = []
        for tr in self.tracks:
            _, py, _, _ = tr.predict_base(t)
            age_unseen = t - tr.last_seen_t
            if py < self.calib.y_min_catch - 0.16:
                continue
            if not tr.locked and age_unseen > 0.65:
                continue
            if tr.locked and age_unseen > 4.5:
                continue
            kept.append(tr)
        self.tracks = kept
        return self.tracks

    def select_target(self, t: float) -> Optional[HolesawTrack]:
        """Choose the earliest locked, unconsumed hole saw still reachable before y_min_catch."""
        candidates: List[Tuple[float, HolesawTrack]] = []
        x_lo, x_hi = self.calib.x_bounds
        for tr in self.tracks:
            if not tr.locked or tr.consumed:
                continue
            px, py, _, vy = tr.predict_base(t)
            if not (x_lo - 0.03 <= px <= x_hi + 0.03):
                continue
            if vy >= -0.02:
                continue
            speed = -vy
            # Need enough runway remaining for sync + descend + brief grasp before y_min_catch.
            min_runway = 0.04 + 0.24 * speed
            if py < self.calib.y_min_catch + min_runway:
                continue
            candidates.append((py, tr))
        if not candidates:
            return None
        # Smallest y is furthest downstream (closest to catch zone).
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def select_preposition_target(self, t: float) -> Optional[HolesawTrack]:
        """Choose a plausible moving track early so the arm can stage before lock."""
        candidates: List[Tuple[float, HolesawTrack]] = []
        x_lo, x_hi = self.calib.x_bounds
        for tr in self.tracks:
            if tr.consumed or len(tr.history) < 2:
                continue
            px, py, _, vy = tr.predict_base(t)
            if not (x_lo - 0.03 <= px <= x_hi + 0.03):
                continue
            if not (-1.50 <= vy <= -0.025):
                continue
            # A tentative estimate is only for staging; grasp selection still
            # requires the stricter locked-track test in select_target().
            if py < self.calib.y_min_catch + 0.04 + 0.12 * abs(vy):
                continue
            candidates.append((py, tr))
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]


def _smoothstep(u: float) -> Tuple[float, float]:
    u = float(np.clip(u, 0.0, 1.0))
    u2 = u * u
    u3 = u2 * u
    s = 10.0 * u3 - 15.0 * u2 * u2 + 6.0 * u3 * u2
    ds = 30.0 * u2 - 60.0 * u3 + 30.0 * u2 * u2
    return s, ds


def _slerp_rotation(R0: np.ndarray, R1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolate cleanly between two 3x3 rotation matrices."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    dR = R0.T @ R1
    rvec, _ = cv2.Rodrigues(dR)
    step_R, _ = cv2.Rodrigues(rvec * alpha)
    return R0 @ step_R


def _rot_z(theta_rad: float) -> np.ndarray:
    c, s = float(np.cos(theta_rad)), float(np.sin(theta_rad))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


class HolesawYamKinematics:
    """Fast MuJoCo FK/IK solver using ``yam_holesaw_kinematics.urdf`` (no i2rt dependency)."""

    def __init__(self, urdf_path: Path = DEFAULT_URDF_PATH) -> None:
        spec = mujoco.MjSpec.from_file(str(Path(urdf_path).expanduser()))
        spec.compiler.fusestatic = False
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.base_id = self.model.body("base").id
        self.tool_id = self.model.body("grasp_site").id
        jnt_ids = np.array([self.model.joint(n).id for n in JOINT_NAMES], dtype=np.int32)
        self.qpos_ids = self.model.jnt_qposadr[jnt_ids]
        self.dof_ids = self.model.jnt_dofadr[jnt_ids]
        self.lo = self.model.jnt_range[jnt_ids, 0].copy()
        self.hi = self.model.jnt_range[jnt_ids, 1].copy()
        self.R_grasp = DEFAULT_R_GRASP.copy()
        self.R_drop = DEFAULT_R_DROP.copy()
        self._jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        self._jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        self._q_err = np.zeros(4, dtype=np.float64)
        self._vel_body = np.zeros(3, dtype=np.float64)
        self._eye6 = np.eye(6, dtype=np.float64)

    def fk(self, q6: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.data.qpos[self.qpos_ids] = np.asarray(q6, dtype=np.float64)[:6]
        mujoco.mj_kinematics(self.model, self.data)
        Rb = self.data.xmat[self.base_id].reshape(3, 3)
        p = Rb.T @ (self.data.xpos[self.tool_id] - self.data.xpos[self.base_id])
        R = Rb.T @ self.data.xmat[self.tool_id].reshape(3, 3)
        return p.copy(), R.copy()

    def _solve_ik_once(
        self,
        target_pos: np.ndarray,
        q_seed: np.ndarray,
        rot_goal: np.ndarray,
        *,
        iters: int,
        rot_weight: float,
        posture_weight: float,
    ) -> Tuple[np.ndarray, float]:
        q = np.clip(np.asarray(q_seed, dtype=np.float64)[:6], self.lo, self.hi).copy()
        pos_err = 1.0

        for _ in range(iters):
            self.data.qpos[self.qpos_ids] = q
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            Rb = self.data.xmat[self.base_id].reshape(3, 3)
            p_cur = Rb.T @ (self.data.xpos[self.tool_id] - self.data.xpos[self.base_id])
            r_cur = Rb.T @ self.data.xmat[self.tool_id].reshape(3, 3)
            e_p = target_pos - p_cur
            pos_err = float(np.linalg.norm(e_p))

            mujoco.mju_mat2Quat(self._q_err, (r_cur.T @ rot_goal).ravel())
            if self._q_err[0] < 0.0:
                self._q_err *= -1.0
            mujoco.mju_quat2Vel(self._vel_body, self._q_err, 1.0)
            e_r = r_cur @ self._vel_body

            if pos_err < 4e-4 and float(np.linalg.norm(e_r)) < 8e-3:
                break

            mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, self.tool_id)
            Jp = Rb.T @ self._jacp[:, self.dof_ids]
            Jr = Rb.T @ self._jacr[:, self.dof_ids]
            J = np.vstack([Jp, rot_weight * Jr])
            err6 = np.concatenate([e_p, rot_weight * e_r])
            # Gentle pull toward the natural elbow-up posture keeps joints away from limits.
            posture_grad = posture_weight * (DEFAULT_HOVER_SEED_Q6 - q)
            lhs = J.T @ J + (5e-4 + posture_weight) * self._eye6
            rhs = J.T @ err6 + posture_grad
            dq = np.linalg.solve(lhs, rhs)
            norm = float(np.linalg.norm(dq))
            if norm > 0.25:
                dq *= 0.25 / norm
            q = np.clip(q + dq, self.lo, self.hi)

        # Evaluate exact final position error.
        p_final, _ = self.fk(q)
        pos_err = float(np.linalg.norm(target_pos - p_final))
        return q, pos_err

    def solve_ik(
        self,
        target_pos: np.ndarray,
        q_seed: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        *,
        iters: int = 24,
        rot_weight: float = 0.09,
        posture_weight: float = 2e-4,
    ) -> Tuple[np.ndarray, float]:
        """Solve 6-DOF IK for grasp_site with automatic re-seed fallback; returns (q6, pos_err_m)."""
        rot_goal = self.R_grasp if target_rot is None else target_rot
        q, pos_err = self._solve_ik_once(
            target_pos,
            q_seed,
            rot_goal,
            iters=iters,
            rot_weight=rot_weight,
            posture_weight=posture_weight,
        )
        # If local seed wandered near a joint limit or elbow flip, retry from canonical elbow-up seed.
        if pos_err > 0.003:
            q_fb, err_fb = self._solve_ik_once(
                target_pos,
                DEFAULT_HOVER_SEED_Q6,
                rot_goal,
                iters=max(iters, 32),
                rot_weight=rot_weight,
                posture_weight=posture_weight,
            )
            if err_fb < pos_err:
                q, pos_err = q_fb, err_fb
        return q, pos_err


class HolesawConveyorController:
    """Autonomous CV + velocity-matched IK state machine for the holesaw conveyor task."""

    def __init__(
        self,
        calib: Optional[HolesawBeltCalib] = None,
        *,
        target_count: int = 3,
        lead_time_s: float = 0.025,
        pd_lag_s: float = 0.035,
        grasp_verify_min_aperture: float = GRIPPER_VERIFY_MIN_APERTURE,
        verify_grasp: bool = True,
        urdf_path: Path = DEFAULT_URDF_PATH,
    ) -> None:
        self.calib = calib or HolesawBeltCalib()
        self.target_count = int(target_count)  # 0 = run continuously
        self.lead_time_s = float(lead_time_s)
        self.pd_lag_s = float(pd_lag_s)
        self._adaptive_lead_s = self.lead_time_s + self.pd_lag_s
        self.grasp_verify_min_aperture = float(grasp_verify_min_aperture)
        self.verify_grasp = bool(verify_grasp)

        self.kin = HolesawYamKinematics(urdf_path=urdf_path)
        self.tracker = MultiHolesawTracker(self.calib)

        self.phase = "OBSERVE"
        self.active_track: Optional[HolesawTrack] = None
        self._last_perception_t: Optional[float] = None
        self._staging_track_id: Optional[int] = None
        self.q_des = self.ready_joints()[:6].copy()
        self.grip_cmd = GRIPPER_OPEN
        self.catches = 0
        self.misses = 0

        # Phase transition bookkeeping.
        self.t_phase = 0.0
        self.t_accel_dur = 0.16
        self.belt_speed_est = 0.10
        self._wait_track_id: Optional[int] = None
        self.z_hover_active = self.calib.z_hover
        self.y_wait_active = self.calib.y_wait
        self.y_accel_start = self.calib.y_wait
        self.x_catch = self.calib.x_hover_default
        self.y_lift_start = 0.0
        self.vy_lift_start = 0.0
        self.p_transit_start = np.array(
            [self.calib.x_hover_default, self.calib.y_wait, self.calib.z_lift], dtype=np.float64
        )
        self.p_drop_target = self.calib.drop_pose_for_index(0)
        self.R_grasp_active = self.kin.R_grasp.copy()
        self.last_detections: List[HolesawDetection] = []

    def ready_joints(self, q_seed: Optional[np.ndarray] = None) -> np.ndarray:
        """7-DOF hover pose above the catch entry point with open gripper."""
        seed = DEFAULT_HOVER_SEED_Q6 if q_seed is None else np.asarray(q_seed, dtype=np.float64)[:6]
        if float(np.linalg.norm(seed - DEFAULT_HOVER_SEED_Q6)) > 1.5:
            seed = DEFAULT_HOVER_SEED_Q6
        p_wait = np.array([self.calib.x_hover_default, self.calib.y_wait, self.calib.z_hover], dtype=np.float64)
        q6, _ = self.kin.solve_ik(p_wait, q_seed=seed, target_rot=self.kin.R_grasp, iters=28)
        return np.append(q6, GRIPPER_OPEN).astype(np.float32)

    def reset(self, q_actual: Optional[np.ndarray] = None) -> None:
        self.tracker.reset()
        self.phase = "OBSERVE"
        self.active_track = None
        self._last_perception_t = None
        self._staging_track_id = None
        self.catches = 0
        self.misses = 0
        self._wait_track_id = None
        self.grip_cmd = GRIPPER_OPEN
        self.q_des = (
            self.ready_joints()[:6].copy()
            if q_actual is None
            else np.asarray(q_actual, dtype=np.float64)[:6].copy()
        )

    def _compute_grasp_rotation(self, tr: HolesawTrack) -> np.ndarray:
        """Keep both fingers level above the belt using the calibrated symmetric grasp frame.

        Twisting an angled wrist around tool Z dips one finger into the conveyor belt;
        keeping R_grasp symmetric clears the belt cleanly for both upright and lying saws.
        """
        _ = tr
        return self.kin.R_grasp

    def step(
        self,
        t: float,
        top_img_rgb: Optional[np.ndarray],
        q_actual: np.ndarray,
        *,
        frame_t: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Advance perception, tracking, state machine, and IK; returns (q7_target, info)."""
        # Always keep upstream CV detection + multi-object tracking running so incoming
        # hole saws lock even while the arm is dropping the previous one.
        perception_t = float(t if frame_t is None else frame_t)
        is_new_frame = (
            self._last_perception_t is None
            or perception_t > self._last_perception_t + 1e-6
        )
        if top_img_rgb is not None and is_new_frame:
            dets = detect_holesaws(top_img_rgb, self.calib)
            self.last_detections = dets
            self.tracker.update(perception_t, dets)
            self._last_perception_t = perception_t

        q_act = np.asarray(q_actual, dtype=np.float64)
        p_ee, _ = self.kin.fk(q_act[:6])
        # Keep hover wait pose adapted to the latest observed belt speed so fast belts
        # don't waste time sprinting upstream from y=0.06 -> y=0.25 on each new saw.
        y_idle = float(
            min(self.calib.y_wait_max, self.calib.y_wait + 0.24 * max(0.0, self.belt_speed_est - 0.15))
        )
        z_idle = (
            min(self.calib.z_hover, self.calib.z_grasp + 0.072)
            if self.belt_speed_est > 0.30
            else self.calib.z_hover
        )
        p_des = np.array([self.calib.x_hover_default, y_idle, z_idle], dtype=np.float64)
        R_des = self.kin.R_grasp

        if self.phase == "OBSERVE":
            # Stage on the first usable velocity estimate, before the track's
            # longer lock window elapses. Keep one staging target stable rather
            # than chasing whichever tentative detection is currently closest.
            staging_track = next(
                (
                    tr for tr in self.tracker.tracks
                    if tr.track_id == self._staging_track_id
                    and not tr.consumed
                    and len(tr.history) >= 2
                    and -1.50 <= tr.vy <= -0.025
                ),
                None,
            )
            if staging_track is None:
                staging_track = self.tracker.select_preposition_target(t)
                self._staging_track_id = (
                    staging_track.track_id if staging_track is not None else None
                )
                self._wait_track_id = None
            if staging_track is not None:
                px, _py, _, vy = staging_track.predict_base(t)
                speed = max(0.02, -vy)
                self.belt_speed_est = speed
                self.x_catch = float(np.clip(px, self.calib.x_bounds[0], self.calib.x_bounds[1]))
                if self._wait_track_id != staging_track.track_id:
                    self.y_wait_active = float(
                        min(self.calib.y_wait_max, self.calib.y_wait + 0.24 * max(0.0, speed - 0.15))
                    )
                    self._wait_track_id = staging_track.track_id
                self.z_hover_active = (
                    min(self.calib.z_hover, self.calib.z_grasp + 0.072)
                    if speed > 0.30
                    else self.calib.z_hover
                )
                p_des = np.array(
                    [self.x_catch, self.y_wait_active, self.z_hover_active], dtype=np.float64
                )

        if self.phase == "DONE":
            self.grip_cmd = GRIPPER_OPEN
            seed = self.q_des if np.linalg.norm(self.q_des - q_act[:6]) < 0.7 else q_act[:6]
            self.q_des, pos_err = self.kin.solve_ik(p_des, q_seed=seed, target_rot=R_des)
            return np.append(self.q_des, self.grip_cmd).astype(np.float32), self._diag(pos_err, p_des)

        if self.phase == "OBSERVE":
            self.grip_cmd = GRIPPER_OPEN
            self.active_track = self.tracker.select_target(t)
            if self.active_track is not None:
                self._staging_track_id = self.active_track.track_id
                self.phase = "PRE_POSITION"
                self.t_phase = t

        if self.phase == "PRE_POSITION":
            self.grip_cmd = GRIPPER_OPEN
            # Re-verify target is still active, locked, moving, and not disappeared far upstream.
            still_valid = (
                self.active_track is not None
                and not self.active_track.consumed
                and self.active_track in self.tracker.tracks
                and self.active_track.locked
                and self.active_track.vy <= -0.025
            )
            if still_valid:
                assert self.active_track is not None
                px, py, _, vy = self.active_track.predict_base(t)
                disappeared_upstream = (t - self.active_track.last_seen_t) > 1.5 and py > self.calib.y_wait + 0.14
                if disappeared_upstream:
                    still_valid = False
            if not still_valid:
                self.active_track = None
                self._wait_track_id = None
                self._staging_track_id = None
                self.phase = "OBSERVE"
            else:
                assert self.active_track is not None
                px, py, _, vy = self.active_track.predict_base(t)
                speed = max(0.02, -vy)
                self.belt_speed_est = speed
                self.x_catch = float(np.clip(px, self.calib.x_bounds[0], self.calib.x_bounds[1]))
                self.R_grasp_active = self._compute_grasp_rotation(self.active_track)
                R_des = self.R_grasp_active
                # Choose the wait point once per target. Recomputing it from each
                # velocity update can make the arm hunt back and forth on approach.
                if self._wait_track_id != self.active_track.track_id:
                    self.y_wait_active = float(
                        min(self.calib.y_wait_max, self.calib.y_wait + 0.24 * max(0.0, speed - 0.15))
                    )
                    self._wait_track_id = self.active_track.track_id
                self.z_hover_active = (
                    min(self.calib.z_hover, self.calib.z_grasp + 0.072)
                    if speed > 0.30
                    else self.calib.z_hover
                )
                p_des = np.array([self.x_catch, self.y_wait_active, self.z_hover_active], dtype=np.float64)

                self.t_accel_dur = float(np.clip(0.14 + 0.10 * speed, 0.14, 0.22))
                eff_lead = self._adaptive_lead_s
                # Belt travels toward -Y: trigger so arm reaches velocity match right over the saw.
                y_trigger = self.y_wait_active + (0.5 * self.t_accel_dur + 0.5 * eff_lead) * speed
                min_runway = 0.03 + 0.20 * speed
                if py < self.calib.y_min_catch + min_runway:
                    # Missed window before catching — abandon this track cleanly.
                    self.active_track.consumed = True
                    self.active_track = None
                    self.phase = "OBSERVE"
                elif py <= y_trigger:
                    self.phase = "SYNC_ACCEL"
                    self.t_phase = t
                    self.y_accel_start = float(p_ee[1])

        if self.phase == "SYNC_ACCEL":
            assert self.active_track is not None
            px, py_raw, _, vy = self.active_track.predict_base(t)
            speed = max(0.02, -vy)
            eff_lead = self._adaptive_lead_s
            self.x_catch = float(np.clip(px, self.calib.x_bounds[0], self.calib.x_bounds[1]))
            y_pred = py_raw + vy * eff_lead
            z_target = self.calib.z_grasp - (0.004 if self.active_track.is_lying_down else 0.0)
            tau = (t - self.t_phase) / max(self.t_accel_dur, 1e-3)
            R_des = self.R_grasp_active
            self.grip_cmd = GRIPPER_OPEN
            if tau >= 1.0:
                self.phase = "DESCEND"
                self.t_phase = t
            else:
                s, _ = _smoothstep(tau)
                y_des = (1.0 - s) * self.y_accel_start + s * y_pred
                z_u = max(0.0, (tau - 0.25) / 0.75)
                zs, _ = _smoothstep(z_u)
                z_frac = 0.65 if speed > 0.30 else 0.50
                z_mid = self.z_hover_active - z_frac * (self.z_hover_active - z_target)
                z_des = self.z_hover_active + zs * (z_mid - self.z_hover_active)
                p_des = np.array([self.x_catch, max(y_des, self.calib.y_min_catch), z_des], dtype=np.float64)

        if self.phase == "DESCEND":
            assert self.active_track is not None
            px, py_raw, _, vy = self.active_track.predict_base(t)
            speed = max(0.02, -vy)
            eff_lead = self._adaptive_lead_s
            self.x_catch = float(np.clip(px, self.calib.x_bounds[0], self.calib.x_bounds[1]))
            y_pred = max(py_raw + vy * eff_lead, self.calib.y_min_catch)
            z_target = self.calib.z_grasp - (0.004 if self.active_track.is_lying_down else 0.0)
            desc_dur = float(np.clip(0.22 - 0.14 * max(0.0, speed - 0.10), 0.09, 0.22))
            tau = (t - self.t_phase) / desc_dur
            z_frac = 0.65 if speed > 0.30 else 0.50
            z_mid = self.z_hover_active - z_frac * (self.z_hover_active - z_target)
            R_des = self.R_grasp_active
            self.grip_cmd = GRIPPER_OPEN
            if tau >= 1.0:
                self.phase = "GRASP"
                self.t_phase = t
            else:
                s, _ = _smoothstep(tau)
                z_des = z_mid + s * (z_target - z_mid)
                p_des = np.array([self.x_catch, y_pred, z_des], dtype=np.float64)

        if self.phase == "GRASP":
            assert self.active_track is not None
            px, py_raw, _, vy = self.active_track.predict_base(t)
            speed = max(0.02, -vy)
            eff_lead = self._adaptive_lead_s
            self.x_catch = float(np.clip(px, self.calib.x_bounds[0], self.calib.x_bounds[1]))
            y_pred = max(py_raw + vy * eff_lead, self.calib.y_min_catch)
            z_target = self.calib.z_grasp - (0.004 if self.active_track.is_lying_down else 0.0)
            grasp_dur = float(np.clip(0.24 - 0.10 * max(0.0, speed - 0.10), 0.14, 0.24))
            tau = (t - self.t_phase) / grasp_dur
            p_des = np.array([self.x_catch, y_pred, z_target], dtype=np.float64)
            R_des = self.R_grasp_active
            self.grip_cmd = GRIPPER_CLOSE
            if tau >= 1.0 or y_pred <= self.calib.y_min_catch:
                self.active_track.consumed = True
                self.phase = "LIFT"
                self.t_phase = t
                self.y_lift_start = float(p_des[1])
                self.vy_lift_start = float(vy)

        if self.phase == "LIFT":
            lift_dur = 0.35
            tau = (t - self.t_phase) / lift_dur
            R_des = self.R_grasp_active
            self.grip_cmd = GRIPPER_CLOSE
            u_dec = min(1.0, (t - self.t_phase) / 0.18)
            y_off = self.vy_lift_start * 0.18 * (u_dec - (2.5 * u_dec**4 - 3.0 * u_dec**5 + u_dec**6))
            y_des = max(self.calib.y_min_catch - 0.03, self.y_lift_start + y_off)
            if tau >= 1.0:
                p_des = np.array([self.x_catch, y_des, self.calib.z_lift], dtype=np.float64)
                has_object = (not self.verify_grasp) or (float(q_act[6]) >= self.grasp_verify_min_aperture)
                self.p_transit_start = p_des.copy()
                self.t_phase = t
                if has_object:
                    self.p_drop_target = self.calib.drop_pose_for_index(self.catches)
                    self.phase = "TRANSIT_TO_CASE"
                else:
                    self.misses += 1
                    self.grip_cmd = GRIPPER_OPEN
                    self.active_track = None
                    self.phase = "RETURN_HOVER"
            else:
                s, _ = _smoothstep(tau)
                z_des = self.calib.z_grasp + s * (self.calib.z_lift - self.calib.z_grasp)
                p_des = np.array([self.x_catch, y_des, z_des], dtype=np.float64)

        if self.phase == "TRANSIT_TO_CASE":
            transit_dur = 0.70
            tau = (t - self.t_phase) / transit_dur
            s, _ = _smoothstep(tau)
            p_end = np.array([self.p_drop_target[0], self.p_drop_target[1], self.calib.z_lift], dtype=np.float64)
            p_des = (1.0 - s) * self.p_transit_start + s * p_end
            R_des = _slerp_rotation(self.R_grasp_active, self.kin.R_drop, s)
            self.grip_cmd = GRIPPER_CLOSE
            if tau >= 1.0:
                self.phase = "DROP_DESCEND"
                self.t_phase = t

        if self.phase == "DROP_DESCEND":
            desc_dur = 0.30
            tau = (t - self.t_phase) / desc_dur
            s, _ = _smoothstep(tau)
            z_des = self.calib.z_lift + s * (self.p_drop_target[2] - self.calib.z_lift)
            p_des = np.array([self.p_drop_target[0], self.p_drop_target[1], z_des], dtype=np.float64)
            R_des = self.kin.R_drop
            self.grip_cmd = GRIPPER_CLOSE
            xy_err = float(np.linalg.norm(p_ee[:2] - self.p_drop_target[:2]))
            # Wait until arm is genuinely over the red case before releasing (handles long transits).
            if (tau >= 1.0 and xy_err < 0.032) or (t - self.t_phase) >= 1.2:
                self.phase = "DROP_RELEASE"
                self.t_phase = t

        if self.phase == "DROP_RELEASE":
            rel_dur = 0.28
            tau = (t - self.t_phase) / rel_dur
            p_des = self.p_drop_target.copy()
            R_des = self.kin.R_drop
            self.grip_cmd = GRIPPER_OPEN
            if tau >= 1.0:
                self.catches += 1
                self.active_track = None
                self.p_transit_start = p_des.copy()
                self.phase = "RETURN_HOVER"
                self.t_phase = t

        if self.phase == "RETURN_HOVER":
            ret_dur = 0.75
            tau = (t - self.t_phase) / ret_dur
            s, _ = _smoothstep(tau)
            p_goal = np.array([self.calib.x_hover_default, y_idle, z_idle], dtype=np.float64)
            # Arc upward through z_lift so the arm clears the red case rim cleanly.
            z_arc = max(self.p_transit_start[2], z_idle) + 0.05 * float(np.sin(np.pi * s))
            p_des = (1.0 - s) * self.p_transit_start + s * p_goal
            p_des[2] = max(p_des[2], z_arc)
            R_des = _slerp_rotation(self.kin.R_drop, self.kin.R_grasp, s)
            self.grip_cmd = GRIPPER_OPEN
            if tau >= 1.0:
                if self.target_count > 0 and self.catches >= self.target_count:
                    self.phase = "DONE"
                else:
                    self.phase = "OBSERVE"
                self.t_phase = t

        if self.phase in {"DESCEND", "GRASP"} and self.active_track is not None:
            # Learn physical arm response lag while the desired Y follows the saw.
            # Dividing spatial tracking error by measured belt speed makes the
            # compensation carry across conveyor speeds without per-speed tuning.
            _, py_raw, _, vy = self.active_track.predict_base(t)
            speed = max(0.02, -vy)
            raw_pred_y = py_raw + vy * self._adaptive_lead_s
            if raw_pred_y > self.calib.y_min_catch + 0.02:
                observed_lag = float(np.clip((p_ee[1] - p_des[1]) / speed, 0.0, 0.25))
                base_lead = self.lead_time_s + self.pd_lag_s
                self._adaptive_lead_s = max(
                    base_lead,
                    0.65 * self._adaptive_lead_s + 0.35 * observed_lag,
                )

        seed = self.q_des if float(np.linalg.norm(self.q_des - q_act[:6])) < 0.7 else q_act[:6]
        if float(np.linalg.norm(seed - DEFAULT_HOVER_SEED_Q6)) > 2.2:
            seed = DEFAULT_HOVER_SEED_Q6
        self.q_des, pos_err = self.kin.solve_ik(p_des, q_seed=seed, target_rot=R_des)
        q7 = np.append(self.q_des, self.grip_cmd).astype(np.float32)
        return q7, self._diag(pos_err, p_des)

    def _diag(self, pos_err: float, p_des: np.ndarray) -> Dict[str, Any]:
        tr = self.active_track
        return {
            "phase": self.phase,
            "catches": self.catches,
            "misses": self.misses,
            "num_detections": len(self.last_detections),
            "num_tracks": len(self.tracker.tracks),
            "num_locked": sum(1 for x in self.tracker.tracks if x.locked and not x.consumed),
            "active_track_id": tr.track_id if tr is not None else None,
            "est_vy_mps": tr.vy if tr is not None else 0.0,
            "p_des": p_des.tolist(),
            "ik_pos_err_mm": pos_err * 1000.0,
            "prediction_lead_ms": self._adaptive_lead_s * 1000.0,
        }


def draw_debug_overlay(
    img_rgb: np.ndarray,
    ctrl: HolesawConveyorController,
    t: float,
    info: Dict[str, Any],
) -> np.ndarray:
    """Annotate an overhead RGB frame with belt polygon, detections, tracks, and state."""
    vis = img_rgb.copy()
    h, w = vis.shape[:2]

    # Belt polygon & upstream tracking boundary.
    poly = ctrl.calib.scaled_belt_poly(w, h)
    cv2.polylines(vis, [poly], isClosed=True, color=(80, 200, 255), thickness=1)
    u_lim = int(round(ctrl.calib.u_track_max * w / ctrl.calib.ref_size[0]))
    cv2.line(vis, (u_lim, 0), (u_lim, 240), (180, 180, 60), 1, cv2.LINE_AA)

    # Catch zone (y_wait -> y_min_catch along belt center).
    u_wait, v_wait = ctrl.calib.base_xy_to_pixel(ctrl.calib.x_hover_default, ctrl.calib.y_wait, width=w, height=h)
    cv2.circle(vis, (int(round(u_wait)), int(round(v_wait))), 6, (255, 160, 0), 2)

    # Raw detections.
    for det in ctrl.last_detections:
        x, y, bw, bh = det.bbox
        col = (255, 140, 0) if det.is_lying_down else (0, 255, 120)
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), col, 2)
        cv2.circle(vis, (int(round(det.u)), int(round(det.v_bot))), 3, (255, 0, 255), -1)

    # Tracked predictions + velocity vectors.
    for tr in ctrl.tracker.tracks:
        u_p, v_p = tr.predict_pixel(t)
        px, py, _, vy = tr.predict_base(t)
        is_active = ctrl.active_track is not None and tr.track_id == ctrl.active_track.track_id
        if tr.consumed:
            col = (140, 140, 140)
        elif is_active:
            col = (0, 255, 255)
        elif tr.locked:
            col = (0, 220, 0)
        else:
            col = (200, 200, 255)
        pt = (int(round(u_p)), int(round(v_p)))
        if -40 <= pt[0] <= w + 40 and -40 <= pt[1] <= h + 40:
            cv2.circle(vis, pt, 8 if is_active else 5, col, 2)
            # 0.5s lookahead arrow.
            u_next, v_next = tr.predict_pixel(t + 0.5)
            cv2.arrowedLine(vis, pt, (int(round(u_next)), int(round(v_next))), col, 2, tipLength=0.25)
            lbl = f"#{tr.track_id} vy={vy:+.2f}" + (" [L]" if tr.is_lying_down else "")
            cv2.putText(vis, lbl, (pt[0] - 25, max(16, pt[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)

    # HUD banner.
    hud = (
        f"t={t:5.1f}s | {info['phase']:15s} | catches={info['catches']} "
        f"| trk={info['num_tracks']} (lock={info['num_locked']}) | ik={info['ik_pos_err_mm']:.1f}mm"
    )
    cv2.rectangle(vis, (0, h - 26), (w, h), (15, 15, 15), -1)
    cv2.putText(vis, hud, (8, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (240, 240, 240), 1, cv2.LINE_AA)
    return vis
