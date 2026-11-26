"""Closed-loop MuJoCo physics simulator proving ``HolesawConveyorController`` end-to-end.

Uses the exact ``i2rt`` YAM + ``linear_4310`` MuJoCo robot model, a real moving conveyor
belt with native MuJoCo Coulomb contact friction, free-joint 3D white hole saws (narrow,
medium, wide, and lying-down) interspersed with yellow triangular distractors, an open
red foam-lined case, and a rendered 640x480 overhead RGB camera.

Zero ground-truth object state is passed to the controller: ``HolesawConveyorController``
receives ONLY the rendered overhead RGB frames and simulated 7-DoF joint encoders.

Example:
    PYTHONPATH=examples/yam:../i2rt .venv/bin/python examples/yam/holesaw_sim_eval.py \\
        --belt-speed 0.095 --target-count 3 --out-video /tmp/holesaw_sim.mp4
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")

import cv2
import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
_I2RT_ROOT = Path(__file__).resolve().parents[3] / "i2rt"
if _I2RT_ROOT.is_dir() and str(_I2RT_ROOT) not in sys.path:
    sys.path.insert(0, str(_I2RT_ROOT))

from holesaw_conveyor_controller import (  # noqa: E402
    HolesawBeltCalib,
    HolesawConveyorController,
    YAM_JOINT_MAX,
    YAM_JOINT_MIN,
    draw_debug_overlay,
)

# Scene physical constants matching the real YAM holesaw rig geometry.
BELT_CENTER_X = 0.510
BELT_HALF_WIDTH = 0.115
BELT_TOP_Z = -0.052
BELT_HALF_THICK = 0.010
BELT_HALF_LEN = 2.60

CASE_CENTER_X = 0.285
CASE_CENTER_Y = 0.075
CASE_BOTTOM_Z = -0.052
CASE_HALF_X = 0.095
CASE_HALF_Y = 0.115
CASE_WALL_H = 0.062

FINGER_MAX_Q = 0.0475


def _camera_lookat_xyaxes(cam_pos: np.ndarray, target: np.ndarray, up: np.ndarray = np.array([0.0, 0.0, 1.0])) -> str:
    fwd = target - cam_pos
    fwd = fwd / np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right = right / np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    cam_up = cam_up / np.linalg.norm(cam_up)
    return " ".join(f"{v:.6f}" for v in np.concatenate([right, cam_up]))


def _project_world_point(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cam_name: str,
    pt_world: np.ndarray,
    width: int = 640,
    height: int = 480,
) -> Tuple[float, float]:
    """Project a 3D world point onto a MuJoCo pinhole camera (u, v)."""
    cid = model.camera(cam_name).id
    cam_pos = data.cam_xpos[cid]
    cam_mat = data.cam_xmat[cid].reshape(3, 3)  # cols: right (+x), up (+y), back (+z)
    p_cam = cam_mat.T @ (pt_world - cam_pos)
    # MuJoCo camera looks along -Z.
    fovy_rad = np.deg2rad(float(model.cam_fovy[cid]))
    fy = 0.5 * height / np.tan(0.5 * fovy_rad)
    fx = fy
    z_depth = max(1e-6, -p_cam[2])
    u = 0.5 * width + fx * (p_cam[0] / z_depth)
    v = 0.5 * height - fy * (p_cam[1] / z_depth)
    return float(u), float(v)


def _compute_finger_pad_local_poses(base_xml_path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Compute crisp box finger-pad local pose inside tip_left and tip_right bodies."""
    m = mujoco.MjModel.from_xml_path(base_xml_path)
    d = mujoco.MjData(m)
    j7 = m.jnt_qposadr[m.joint("joint7").id]
    j8 = m.jnt_qposadr[m.joint("joint8").id]
    d.qpos[j7] = FINGER_MAX_Q
    d.qpos[j8] = FINGER_MAX_Q
    mujoco.mj_kinematics(m, d)

    pads: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for bname, sign in (("tip_left", +1.0), ("tip_right", -1.0)):
        bid = m.body(bname).id
        # At q=0.0475, inner pad surface sits at y = sign * 0.0470, x in [0.195, 0.254], z in [0.155, 0.192].
        # Put a 10mm-thick pad centered at y = sign * 0.0520 so its inner face is at y = sign * 0.0470
        # (which closes cleanly to y = 0.000 at q=0!).
        p_world = np.array([0.226, sign * 0.0520, 0.1735], dtype=np.float64)
        R_b = d.xmat[bid].reshape(3, 3)
        p_b = d.xpos[bid]
        p_local = R_b.T @ (p_world - p_b)
        # Express world-axis-aligned box orientation in body local frame.
        R_local = R_b.T
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, R_local.ravel())
        pads[bname] = (p_local, quat)
    return pads


def build_holesaw_sim_model(
    *,
    belt_speed_mps: float = 0.095,
    include_wrist_mount: bool = True,
) -> Tuple[mujoco.MjModel, List[str], List[str]]:
    """Build the complete MuJoCo holesaw conveyor scene using the real i2rt YAM model."""
    from i2rt.robots.utils import ArmType, GripperType, combine_arm_and_gripper_xml

    base_xml = combine_arm_and_gripper_xml(ArmType.YAM, GripperType.LINEAR_4310)
    pads = _compute_finger_pad_local_poses(base_xml)

    tree = ET.parse(base_xml)
    root = tree.getroot()

    # Physics options: crisp contacts, 2ms timestep, implicitfast integrator.
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    opt.set("timestep", "0.002")
    opt.set("integrator", "implicitfast")
    opt.set("cone", "elliptic")
    opt.set("impratio", "5")
    opt.set("noslip_iterations", "3")

    # Visual framebuffer & bright warehouse ambient lighting matching the real rig.
    vis = root.find("visual")
    if vis is None:
        vis = ET.SubElement(root, "visual")
    gl = vis.find("global")
    if gl is None:
        gl = ET.SubElement(vis, "global")
    gl.set("offwidth", "640")
    gl.set("offheight", "480")
    hl = vis.find("headlight")
    if hl is None:
        hl = ET.SubElement(vis, "headlight")
    hl.set("ambient", "0.42 0.42 0.42")
    hl.set("diffuse", "0.65 0.65 0.65")
    hl.set("specular", "0.15 0.15 0.15")

    # Gravity compensation on all YAM links (matching hardware gravcomp=1) + realistic dark finish.
    arm_link_colors = {
        "base": "0.14 0.14 0.16 1",
        "link1": "0.16 0.16 0.18 1",
        "link2": "0.14 0.15 0.17 1",
        "link3": "0.16 0.16 0.18 1",
        "link4": "0.14 0.14 0.16 1",
        "link5": "0.16 0.16 0.18 1",
        "gripper": "0.12 0.12 0.14 1",
        "tip_left": "0.18 0.19 0.21 1",
        "tip_right": "0.18 0.19 0.21 1",
    }
    for body in root.iter("body"):
        bname = body.get("name", "")
        if bname in arm_link_colors or bname == "link6":
            body.set("gravcomp", "1")
        for g in body.findall("geom"):
            if bname in arm_link_colors:
                g.set("rgba", arm_link_colors[bname])
            # Disable raw concave STL collisions on arm & L-brackets; use crisp finger pad boxes below.
            if bname in ("base", "link1", "link2", "link3", "link4", "link5", "gripper", "tip_left", "tip_right"):
                g.set("contype", "0")
                g.set("conaffinity", "0")

        # Add high-friction box finger pads aligned with STL tips.
        if bname in pads:
            p_loc, q_loc = pads[bname]
            ET.SubElement(
                body,
                "geom",
                {
                    "name": f"{bname}_pad",
                    "type": "box",
                    "size": "0.030 0.005 0.020",
                    "pos": " ".join(f"{v:.6f}" for v in p_loc),
                    "quat": " ".join(f"{v:.6f}" for v in q_loc),
                    "rgba": "0.15 0.15 0.16 1",
                    "friction": "2.6 0.015 0.003",
                    "condim": "4",
                    "solref": "0.006 1",
                    "solimp": "0.95 0.99 0.001",
                },
            )

        # Optional white camera housing on link6 (just like the real rig's wrist mount!)
        # to prove our CV tracker ignores the arm's own white hardware.
        if include_wrist_mount and bname == "gripper":
            ET.SubElement(
                body,
                "geom",
                {
                    "name": "wrist_cam_mount",
                    "type": "box",
                    "size": "0.022 0.018 0.016",
                    "pos": "0.0 0.048 -0.020",
                    "rgba": "0.92 0.92 0.92 1",
                    "contype": "0",
                    "conaffinity": "0",
                },
            )

    worldbody = root.find("worldbody")
    assert worldbody is not None

    # Overhead studio lighting matching the bright warehouse rig.
    ET.SubElement(
        worldbody,
        "light",
        {
            "pos": "0.45 0.15 1.6",
            "dir": "0 0 -1",
            "diffuse": "0.95 0.95 0.95",
            "specular": "0.25 0.25 0.25",
            "castshadow": "false",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "pos": "0.0 0.6 1.2",
            "dir": "0.4 -0.4 -0.8",
            "diffuse": "0.45 0.45 0.45",
            "specular": "0.1 0.1 0.1",
            "castshadow": "false",
        },
    )

    # Cameras:
    # 1) overhead_cam: angled D435-like view where belt runs diagonally left-to-right & red case is lower-right.
    ov_pos = np.array([0.08, 0.12, 0.66], dtype=np.float64)
    ov_tgt = np.array([0.50, 0.21, BELT_TOP_Z], dtype=np.float64)
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "overhead_cam",
            "pos": " ".join(f"{v:.4f}" for v in ov_pos),
            "xyaxes": _camera_lookat_xyaxes(ov_pos, ov_tgt, up=np.array([0.0, 0.0, 1.0])),
            "fovy": "48",
        },
    )
    # 2) side_cam: crisp 3/4 view showing robot arm, moving belt, hole saws, yellow triangles, and red case.
    sc_pos = np.array([-0.22, -0.58, 0.50], dtype=np.float64)
    sc_tgt = np.array([0.40, 0.08, 0.02], dtype=np.float64)
    ET.SubElement(
        worldbody,
        "camera",
        {
            "name": "side_cam",
            "pos": " ".join(f"{v:.4f}" for v in sc_pos),
            "xyaxes": _camera_lookat_xyaxes(sc_pos, sc_tgt),
            "fovy": "46",
        },
    )

    # Concrete floor.
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "size": "3 3 0.1",
            "pos": "0 0 -0.45",
            "rgba": "0.48 0.46 0.42 1",
        },
    )

    # Moving matte black conveyor belt body (native MuJoCo slide joint along -Y!).
    # Scale belt slab length with speed so even 10x (~1.0 m/s) runs never run out of belt.
    belt_half_len = max(BELT_HALF_LEN, 16.0 * belt_speed_mps)
    belt_body = ET.SubElement(
        worldbody,
        "body",
        {
            "name": "conveyor_belt",
            "pos": f"{BELT_CENTER_X:.4f} {belt_half_len - 0.80:.4f} {BELT_TOP_Z - BELT_HALF_THICK:.4f}",
        },
    )
    ET.SubElement(
        belt_body,
        "joint",
        {
            "name": "belt_slide",
            "type": "slide",
            "axis": "0 -1 0",
            "limited": "false",
            "damping": "0.0",
        },
    )
    ET.SubElement(
        belt_body,
        "inertial",
        {"pos": "0 0 0", "mass": "50.0", "diaginertia": "10 10 10"},
    )
    ET.SubElement(
        belt_body,
        "geom",
        {
            "name": "belt_surface",
            "type": "box",
            "size": f"{BELT_HALF_WIDTH:.4f} {belt_half_len:.4f} {BELT_HALF_THICK:.4f}",
            "rgba": "0.055 0.058 0.062 1",
            "friction": "1.4 0.008 0.002",
            "condim": "4",
            "solref": "0.008 1",
        },
    )

    # Static brushed silver conveyor rails + frame.
    for side, sx in (("near", BELT_CENTER_X - BELT_HALF_WIDTH - 0.015), ("far", BELT_CENTER_X + BELT_HALF_WIDTH + 0.015)):
        ET.SubElement(
            worldbody,
            "geom",
            {
                "name": f"rail_{side}",
                "type": "box",
                "size": "0.014 1.6 0.016",
                "pos": f"{sx:.4f} 0.45 {BELT_TOP_Z - 0.012:.4f}",
                "rgba": "0.70 0.72 0.75 1",
                "contype": "0",
                "conaffinity": "0",
            },
        )

    # Cardboard box stand + open red foam-lined case at (CASE_CENTER_X, CASE_CENTER_Y).
    # Keep cardboard box clear of the sliding conveyor belt (belt starts at x = 0.395).
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "cardboard_box",
            "type": "box",
            "size": "0.10 0.12 0.19",
            "pos": f"{CASE_CENTER_X:.4f} {CASE_CENTER_Y:.4f} {CASE_BOTTOM_Z - 0.205:.4f}",
            "rgba": "0.62 0.49 0.34 1",
            "contype": "0",
            "conaffinity": "0",
        },
    )
    case_body = ET.SubElement(
        worldbody,
        "body",
        {"name": "red_case", "pos": f"{CASE_CENTER_X:.4f} {CASE_CENTER_Y:.4f} {CASE_BOTTOM_Z:.4f}"},
    )
    # Dark foam floor inside red case.
    ET.SubElement(
        case_body,
        "geom",
        {
            "name": "case_floor",
            "type": "box",
            "size": f"{CASE_HALF_X:.4f} {CASE_HALF_Y:.4f} 0.008",
            "pos": "0 0 -0.008",
            "rgba": "0.07 0.07 0.08 1",
            "friction": "1.2 0.01 0.002",
        },
    )
    wall_t = 0.008
    wall_specs = [
        ("case_wall_px", f"{CASE_HALF_X:.4f} 0 {CASE_WALL_H/2:.4f}", f"{wall_t} {CASE_HALF_Y:.4f} {CASE_WALL_H/2:.4f}"),
        ("case_wall_nx", f"{-CASE_HALF_X:.4f} 0 {CASE_WALL_H/2:.4f}", f"{wall_t} {CASE_HALF_Y:.4f} {CASE_WALL_H/2:.4f}"),
        ("case_wall_py", f"0 {CASE_HALF_Y:.4f} {CASE_WALL_H/2:.4f}", f"{CASE_HALF_X:.4f} {wall_t} {CASE_WALL_H/2:.4f}"),
        ("case_wall_ny", f"0 {-CASE_HALF_Y:.4f} {CASE_WALL_H/2:.4f}", f"{CASE_HALF_X:.4f} {wall_t} {CASE_WALL_H/2:.4f}"),
    ]
    for wname, wpos, wsize in wall_specs:
        ET.SubElement(
            case_body,
            "geom",
            {"name": wname, "type": "box", "pos": wpos, "size": wsize, "rgba": "0.84 0.18 0.08 1"},
        )

    # Spawn 4 diverse white hole saws (medium, narrow, wide, lying-down) + 4 yellow triangle distractors.
    # Spaced in time (~4.5s between saws so the arm finishes each drop-and-return cycle even at 10x speed).
    dt_spacing = 4.6
    y_step = max(0.55, dt_spacing * belt_speed_mps)
    y_first = max(0.62, 1.4 * belt_speed_mps + 0.45)
    holesaw_specs = [
        # (name, x, y, radius, half_height, lying_down)
        ("holesaw_1_med", 0.515, y_first + 0.0 * y_step, 0.021, 0.026, False),
        ("holesaw_2_narrow", 0.485, y_first + 1.0 * y_step, 0.015, 0.026, False),
        ("holesaw_3_wide", 0.540, y_first + 2.0 * y_step, 0.026, 0.025, False),
        ("holesaw_4_lying", 0.505, y_first + 3.0 * y_step, 0.018, 0.026, True),
    ]
    holesaw_names: List[str] = []
    for sname, sx, sy, rad, hh, lying in holesaw_specs:
        holesaw_names.append(sname)
        sz = BELT_TOP_Z + (rad + 0.001 if lying else hh + 0.001)
        quat = "0.707107 0.707107 0 0" if lying else "1 0 0 0"
        b = ET.SubElement(worldbody, "body", {"name": sname, "pos": f"{sx:.4f} {sy:.4f} {sz:.4f}", "quat": quat})
        ET.SubElement(b, "freejoint", {"name": f"{sname}_free"})
        ET.SubElement(
            b,
            "geom",
            {
                "name": f"{sname}_cyl",
                "type": "cylinder",
                "size": f"{rad:.4f} {hh:.4f}",
                "mass": "0.085",
                "rgba": "0.96 0.96 0.97 1",
                "friction": "1.8 0.01 0.002",
                "condim": "4",
                "solref": "0.006 1",
                "solimp": "0.95 0.99 0.001",
            },
        )
        # Visual dark bore cap on top so it looks like a hollow hole saw.
        ET.SubElement(
            b,
            "geom",
            {
                "name": f"{sname}_bore",
                "type": "cylinder",
                "size": f"{rad * 0.65:.4f} 0.001",
                "pos": f"0 0 {hh + 0.0004:.4f}",
                "rgba": "0.18 0.19 0.20 1",
                "contype": "0",
                "conaffinity": "0",
                "mass": "0.0001",
            },
        )

    # Yellow triangular distractors interspersed BEFORE and BETWEEN the hole saws.
    distractor_specs = [
        ("yellow_tri_1", 0.505, max(0.35, y_first - 0.5 * y_step)),
        ("yellow_tri_2", 0.530, y_first + 0.5 * y_step),
        ("yellow_tri_3", 0.490, y_first + 1.5 * y_step),
        ("yellow_tri_4", 0.520, y_first + 2.5 * y_step),
    ]
    distractor_names: List[str] = []
    for dname, dx, dy in distractor_specs:
        distractor_names.append(dname)
        b = ET.SubElement(worldbody, "body", {"name": dname, "pos": f"{dx:.4f} {dy:.4f} {BELT_TOP_Z + 0.016:.4f}"})
        ET.SubElement(b, "freejoint", {"name": f"{dname}_free"})
        # 3-box triangle silhouette in bright yellow.
        ET.SubElement(
            b,
            "geom",
            {
                "name": f"{dname}_base",
                "type": "box",
                "size": "0.022 0.022 0.014",
                "mass": "0.045",
                "rgba": "0.96 0.80 0.05 1",
                "friction": "1.2 0.008 0.002",
            },
        )
        ET.SubElement(
            b,
            "geom",
            {
                "name": f"{dname}_fin",
                "type": "box",
                "size": "0.014 0.026 0.015",
                "euler": "0 0 0.52",
                "rgba": "0.96 0.80 0.05 1",
                "contype": "0",
                "conaffinity": "0",
                "mass": "0.0001",
            },
        )

    # Position actuators for YAM joints 1..6 + finger slides 7..8 + velocity actuator for conveyor belt.
    # Torque limits & damping matched to hardware YAM DM4340 (j1-3) / DM4310 (j4-6) actuators.
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    arm_gains = [
        ("joint1", 150, 4.5, 28),
        ("joint2", 150, 4.5, 28),
        ("joint3", 130, 4.0, 28),
        ("joint4", 80, 2.0, 10),
        ("joint5", 60, 1.5, 10),
        ("joint6", 50, 1.2, 10),
    ]
    for jname, kp, kv, flim in arm_gains:
        ET.SubElement(
            act,
            "position",
            {
                "name": f"act_{jname}",
                "joint": jname,
                "kp": str(kp),
                "kv": str(kv),
                "forcerange": f"-{flim} {flim}",
            },
        )
    for jname in ("joint7", "joint8"):
        ET.SubElement(
            act,
            "position",
            {
                "name": f"act_{jname}",
                "joint": jname,
                "kp": "900",
                "kv": "35",
                "ctrlrange": f"0 {FINGER_MAX_Q}",
                "forcerange": "-30 30",
            },
        )
    ET.SubElement(
        act,
        "velocity",
        {
            "name": "act_belt_vel",
            "joint": "belt_slide",
            "kv": "800",
            "forcerange": "-500 500",
        },
    )

    xml_str = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_str)
    return model, holesaw_names, distractor_names


def calibrate_sim_camera(model: mujoco.MjModel, data: mujoco.MjData) -> HolesawBeltCalib:
    """Calibrate ``HolesawBeltCalib`` using ONLY 4 belt-plane points (exact 4-point rig calib)."""
    mujoco.mj_forward(model, data)

    # 4 belt surface calibration points spanning the belt width and camera FOV.
    calib_xy = np.array(
        [
            [BELT_CENTER_X - 0.08, 0.50],
            [BELT_CENTER_X + 0.08, 0.50],
            [BELT_CENTER_X + 0.08, -0.08],
            [BELT_CENTER_X - 0.08, -0.08],
        ],
        dtype=np.float64,
    )
    calib_uv = np.array(
        [_project_world_point(model, data, "overhead_cam", np.array([x, y, BELT_TOP_Z])) for x, y in calib_xy],
        dtype=np.float64,
    )
    H, _ = cv2.findHomography(calib_uv, calib_xy)
    assert H is not None

    # Project the inset matte black belt polygon corners into overhead_cam pixels.
    inset = BELT_HALF_WIDTH - 0.015
    poly_world = [
        np.array([BELT_CENTER_X + inset, 0.72, BELT_TOP_Z]),
        np.array([BELT_CENTER_X + inset, -0.25, BELT_TOP_Z]),
        np.array([BELT_CENTER_X - inset, -0.25, BELT_TOP_Z]),
        np.array([BELT_CENTER_X - inset, 0.72, BELT_TOP_Z]),
    ]
    poly_px = tuple(
        (
            int(np.clip(round(u), 0, 639)),
            int(np.clip(round(v), 0, 479)),
        )
        for u, v in (_project_world_point(model, data, "overhead_cam", p) for p in poly_world)
    )

    # Compute exact upstream spawn/track pixel limits corresponding to y = +0.26m and y = +0.10m.
    u_spawn, _ = _project_world_point(model, data, "overhead_cam", np.array([BELT_CENTER_X, 0.24, BELT_TOP_Z]))
    u_track, _ = _project_world_point(model, data, "overhead_cam", np.array([BELT_CENTER_X, 0.10, BELT_TOP_Z]))

    return HolesawBeltCalib(
        homography=H,
        belt_poly=poly_px,
        u_spawn_max=float(np.clip(u_spawn, 220.0, 380.0)),
        u_track_max=float(np.clip(u_track, 380.0, 500.0)),
        x_bounds=(BELT_CENTER_X - 0.08, BELT_CENTER_X + 0.08),
        x_hover_default=BELT_CENTER_X,
        y_wait=0.06,
        y_wait_max=0.25,
        y_min_catch=-0.24,
        z_grasp=BELT_TOP_Z + 0.025,  # -0.027 m (equator of 50mm upright saws; lowers -4mm for lying saws)
        z_hover=0.095,
        z_lift=0.145,
        drop_center=(CASE_CENTER_X, CASE_CENTER_Y, 0.055),
    )


def _is_inside_red_case(pos: np.ndarray) -> bool:
    return (
        abs(float(pos[0]) - CASE_CENTER_X) <= CASE_HALF_X - 0.005
        and abs(float(pos[1]) - CASE_CENTER_Y) <= CASE_HALF_Y - 0.005
        and CASE_BOTTOM_Z - 0.015 <= float(pos[2]) <= CASE_BOTTOM_Z + 0.12
    )


def run_simulation(
    *,
    belt_speed_mps: float = 0.095,
    target_count: int = 3,
    max_seconds: float = 34.0,
    control_hz: float = 60.0,
    cam_hz: float = 30.0,
    out_video: Optional[Path] = None,
    video_fps: int = 15,
) -> Dict[str, Any]:
    """Run full closed-loop MuJoCo physics sim using ``HolesawConveyorController``."""
    model, holesaw_names, distractor_names = build_holesaw_sim_model(belt_speed_mps=belt_speed_mps)
    data = mujoco.MjData(model)

    calib = calibrate_sim_camera(model, data)
    ctrl = HolesawConveyorController(
        calib,
        target_count=target_count,
        lead_time_s=0.035,
        verify_grasp=True,
    )

    # Initialize YAM arm at ready hover pose with open gripper.
    ready_q7 = ctrl.ready_joints()
    arm_qids = [model.jnt_qposadr[model.joint(f"joint{i}").id] for i in range(1, 7)]
    j7_qid = model.jnt_qposadr[model.joint("joint7").id]
    j8_qid = model.jnt_qposadr[model.joint("joint8").id]
    for k, qid in enumerate(arm_qids):
        data.qpos[qid] = float(ready_q7[k])
        data.ctrl[k] = float(ready_q7[k])
    data.qpos[j7_qid] = FINGER_MAX_Q * float(ready_q7[6])
    data.qpos[j8_qid] = FINGER_MAX_Q * float(ready_q7[6])
    data.ctrl[6] = FINGER_MAX_Q * float(ready_q7[6])
    data.ctrl[7] = FINGER_MAX_Q * float(ready_q7[6])
    data.ctrl[8] = float(belt_speed_mps)
    # Start the belt and all items already moving steadily at belt_speed_mps (no t=0 startup impulse).
    belt_dof = model.jnt_dofadr[model.joint("belt_slide").id]
    data.qvel[belt_dof] = float(belt_speed_mps)
    for bname in holesaw_names + distractor_names:
        fj_id = model.joint(f"{bname}_free").id
        dof_adr = model.jnt_dofadr[fj_id]
        data.qvel[dof_adr + 1] = -float(belt_speed_mps)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)
    phys_dt = float(model.opt.timestep)
    ctrl_steps = max(1, int(round((1.0 / control_hz) / phys_dt)))
    cam_stride = max(1, int(round(control_hz / cam_hz)))
    vid_stride = max(1, int(round(control_hz / video_fps)))

    n_ticks = int(round(max_seconds * control_hz))
    frames_out: List[np.ndarray] = []
    max_ik_err_mm = 0.0
    distractor_min_dist = 10.0
    grasp_site_id = model.site("grasp_site").id
    distractor_bids = [model.body(n).id for n in distractor_names]
    holesaw_bids = [model.body(n).id for n in holesaw_names]

    last_ov_rgb: Optional[np.ndarray] = None
    done_t: Optional[float] = None

    try:
        for tick in range(n_ticks):
            t_sim = float(data.time)
            q_meas = np.array(
                [data.qpos[qid] for qid in arm_qids] + [0.5 * (data.qpos[j7_qid] + data.qpos[j8_qid]) / FINGER_MAX_Q],
                dtype=np.float64,
            )

            ov_rgb = None
            if tick % cam_stride == 0:
                renderer.update_scene(data, camera="overhead_cam")
                ov_rgb = renderer.render().copy()
                last_ov_rgb = ov_rgb

            q7_cmd, info = ctrl.step(t_sim, ov_rgb, q_meas)
            max_ik_err_mm = max(max_ik_err_mm, float(info["ik_pos_err_mm"]))

            # Apply safety-clamped actuator targets (identical to live rollout).
            delta = np.clip(q7_cmd[:6] - q_meas[:6], -0.14, 0.14)
            q6_safe = np.clip(q_meas[:6] + delta, YAM_JOINT_MIN, YAM_JOINT_MAX)
            g_safe = float(np.clip(q7_cmd[6], 0.0, 1.0))
            for k in range(6):
                data.ctrl[k] = float(q6_safe[k])
            data.ctrl[6] = FINGER_MAX_Q * g_safe
            data.ctrl[7] = FINGER_MAX_Q * g_safe
            data.ctrl[8] = float(belt_speed_mps)

            for _ in range(ctrl_steps):
                mujoco.mj_step(model, data)

            # Monitor clearance from every yellow distractor.
            p_grasp = data.site_xpos[grasp_site_id]
            for bid in distractor_bids:
                d_dist = float(np.linalg.norm(p_grasp - data.xpos[bid]))
                distractor_min_dist = min(distractor_min_dist, d_dist)

            if out_video is not None and tick % vid_stride == 0 and last_ov_rgb is not None:
                left_pane = draw_debug_overlay(last_ov_rgb, ctrl, t_sim, info)
                renderer.update_scene(data, camera="side_cam")
                right_pane = renderer.render().copy()
                cv2.putText(
                    right_pane,
                    f"MuJoCo 3D Physics | belt={belt_speed_mps:.3f}m/s | in_case={sum(_is_inside_red_case(data.xpos[b]) for b in holesaw_bids)}/{target_count}",
                    (12, 26),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (250, 250, 250),
                    2,
                    cv2.LINE_AA,
                )
                frames_out.append(np.hstack([left_pane, right_pane]))

            if ctrl.phase == "DONE":
                if done_t is None:
                    done_t = t_sim
                elif t_sim - done_t >= 1.5:
                    # Settled for 1.5s after DONE.
                    break
    finally:
        renderer.close()

    # Final physical ground-truth check after release & settle.
    holesaws_in_case = [n for n, bid in zip(holesaw_names, holesaw_bids) if _is_inside_red_case(data.xpos[bid])]
    distractors_in_case = [n for n, bid in zip(distractor_names, distractor_bids) if _is_inside_red_case(data.xpos[bid])]
    holesaw_positions = {n: np.round(data.xpos[bid], 3).tolist() for n, bid in zip(holesaw_names, holesaw_bids)}

    if out_video is not None and frames_out:
        out_video.parent.mkdir(parents=True, exist_ok=True)
        h, w = frames_out[0].shape[:2]
        cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-s",
            f"{w}x{h}",
            "-pix_fmt",
            "rgb24",
            "-r",
            str(video_fps),
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "24",
            str(out_video),
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        assert proc.stdin is not None
        for fr in frames_out:
            proc.stdin.write(fr.tobytes())
        proc.stdin.close()
        proc.wait()

    return {
        "belt_speed_mps": belt_speed_mps,
        "sim_time_s": round(float(data.time), 2),
        "controller_phase": ctrl.phase,
        "controller_catches": ctrl.catches,
        "controller_misses": ctrl.misses,
        "physical_holesaws_in_case": holesaws_in_case,
        "physical_holesaws_count": len(holesaws_in_case),
        "physical_distractors_in_case": distractors_in_case,
        "distractor_min_clearance_m": round(distractor_min_dist, 3),
        "max_ik_err_mm": round(max_ik_err_mm, 2),
        "holesaw_final_positions": holesaw_positions,
        "success": len(holesaws_in_case) >= target_count and len(distractors_in_case) == 0,
        "out_video": str(out_video) if out_video else None,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Closed-loop MuJoCo sim evaluation of HolesawConveyorController")
    p.add_argument("--belt-speed", type=float, default=0.095, help="Conveyor belt speed in m/s")
    p.add_argument("--target-count", type=int, default=3, help="Number of hole saws to catch and place")
    p.add_argument("--cam-hz", type=float, default=30.0, help="Simulated overhead camera FPS (30 or 60)")
    p.add_argument("--max-seconds", type=float, default=34.0)
    p.add_argument("--out-video", type=Path, default=None)
    args = p.parse_args()
    res = run_simulation(
        belt_speed_mps=args.belt_speed,
        target_count=args.target_count,
        cam_hz=args.cam_hz,
        max_seconds=args.max_seconds,
        out_video=args.out_video,
    )
    print(res)
    if not res["success"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
