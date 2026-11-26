"""Unit tests for the YAM holesaw CV detector, multi-object belt tracker, and IK state machine."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from holesaw_conveyor_controller import (
    GRIPPER_CLOSE,
    GRIPPER_OPEN,
    HolesawBeltCalib,
    HolesawConveyorController,
    HolesawYamKinematics,
    MultiHolesawTracker,
    detect_holesaws,
)


def _make_belt_frame(
    calib: HolesawBeltCalib,
    *,
    white_saws: list[tuple[int, int, int, int]] = (),
    yellow_triangles: list[tuple[int, int, int]] = (),
    add_cable_x: int | None = None,
) -> np.ndarray:
    """Render a synthetic 640x480 overhead RGB frame matching the real conveyor setup."""
    img = np.full((480, 640, 3), (120, 115, 105), dtype=np.uint8)
    poly = calib.scaled_belt_poly(640, 480)
    cv2.fillPoly(img, [poly], (32, 34, 36))

    for cx, cy, rx, ry in white_saws:
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, (240, 242, 245), -1)
        # Dark hollow center & side slot like a real bi-metal hole saw.
        cv2.circle(img, (cx, cy - 3), max(3, min(rx, ry) // 3), (50, 52, 55), -1)

    for tx, ty, r in yellow_triangles:
        pts = np.array([[tx, ty - r], [tx - r, ty + r], [tx + r, ty + r]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (245, 200, 15))
        # Add a bright specular white highlight inside the yellow triangle to test glare guard.
        cv2.circle(img, (tx, ty + 2), max(4, r // 3), (245, 245, 245), -1)

    if add_cable_x is not None:
        cv2.line(img, (add_cable_x, 0), (add_cable_x - 20, 479), (20, 20, 20), 4)

    return img


def test_detect_holesaws_rejects_yellow_triangles_with_glare() -> None:
    calib = HolesawBeltCalib()
    frame = _make_belt_frame(
        calib,
        white_saws=[(180, 130, 18, 22)],
        yellow_triangles=[(300, 105, 22), (100, 155, 20)],
    )
    dets = detect_holesaws(frame, calib)
    assert len(dets) == 1
    assert abs(dets[0].u - 180.0) < 6.0
    assert not dets[0].is_lying_down


def test_detect_holesaws_handles_lying_down_and_cable_split() -> None:
    calib = HolesawBeltCalib()
    # Elongated lying-down hole saw split right down the middle by a 4px black cable.
    frame = _make_belt_frame(
        calib,
        white_saws=[(240, 120, 32, 14)],
        add_cable_x=245,
    )
    dets = detect_holesaws(frame, calib)
    assert len(dets) == 1
    assert dets[0].is_lying_down
    assert abs(dets[0].u - 240.0) < 8.0


def test_multi_tracker_locks_two_moving_saws_and_ignores_stationary_and_wrist_blobs() -> None:
    calib = HolesawBeltCalib()
    tracker = MultiHolesawTracker(calib)

    for k in range(15):
        t = k * (1.0 / 15.0)
        u1 = int(round(120 + 95.0 * t))
        v1 = int(round(155 - 18.0 * t))
        u2 = int(round(40 + 95.0 * t))
        v2 = int(round(170 - 18.0 * t))
        # A moving white blob entering mid-belt at u=360..430 (like the robot wrist mount)
        # must NEVER spawn a track, and a stationary glint at (260, 115) must never lock.
        u_wrist = int(round(350 + 80.0 * t))
        frame = _make_belt_frame(
            calib,
            white_saws=[(u1, v1, 18, 20), (u2, v2, 16, 18), (u_wrist, 85, 16, 16), (260, 120, 12, 12)],
            yellow_triangles=[(220, 125, 20)],
        )
        dets = detect_holesaws(frame, calib)
        tracker.update(t, dets)

    locked = [tr for tr in tracker.tracks if tr.locked]
    assert len(locked) == 2
    for tr in locked:
        assert -0.13 < tr.vy < -0.06
        assert abs(tr.vx) < 0.025


def test_kinematics_ik_reachable_across_catch_and_drop_zones() -> None:
    kin = HolesawYamKinematics()
    q_seed = np.array([-0.05, 2.15, 1.62, -0.50, -0.72, -1.24], dtype=np.float64)

    # Belt catch grid.
    for x in (0.45, 0.51, 0.56):
        for y in (-0.14, -0.04, 0.06):
            for z in (-0.016, 0.095):
                q, err = kin.solve_ik(np.array([x, y, z]), q_seed=q_seed, target_rot=kin.R_grasp, iters=30)
                assert err < 0.005, f"Catch IK failed at ({x},{y},{z}): err={err*1000:.1f}mm"
                assert np.all(q >= kin.lo - 1e-6) and np.all(q <= kin.hi + 1e-6)

    # Red case drop grid.
    calib = HolesawBeltCalib()
    for idx in range(len(calib.drop_offsets)):
        p_drop = calib.drop_pose_for_index(idx)
        q, err = kin.solve_ik(p_drop, q_seed=q_seed, target_rot=kin.R_drop, iters=30)
        assert err < 0.005, f"Drop IK failed at {p_drop}: err={err*1000:.1f}mm"


@pytest.mark.parametrize("simulate_miss", [False, True])
def test_controller_full_pick_verify_drop_cycle(simulate_miss: bool) -> None:
    calib = HolesawBeltCalib()
    ctrl = HolesawConveyorController(calib, target_count=1, verify_grasp=True)
    q_sim = ctrl.ready_joints().astype(np.float64)

    phases_seen = []
    dt = 1.0 / 30.0
    for step in range(240):
        t = step * dt
        u = int(round(200 + 95.0 * t))
        v = int(round(130 - 18.0 * t))
        saws = [(u, v, 18, 22)] if 15 <= u <= 620 else []
        frame = _make_belt_frame(calib, white_saws=saws, yellow_triangles=[(310, 100, 20)])
        q_cmd, info = ctrl.step(t, frame, q_sim)
        assert info["ik_pos_err_mm"] < 5.0

        # Simulate robot tracking & gripper aperture feedback.
        q_sim[:6] = q_cmd[:6]
        if q_cmd[6] <= GRIPPER_CLOSE + 0.05:
            q_sim[6] = 0.015 if simulate_miss else 0.22
        else:
            q_sim[6] = GRIPPER_OPEN

        if not phases_seen or phases_seen[-1] != info["phase"]:
            phases_seen.append(info["phase"])
        if ctrl.phase == "DONE":
            break

    if simulate_miss:
        assert ctrl.catches == 0
        assert ctrl.misses == 1
        assert "TRANSIT_TO_CASE" not in phases_seen
        assert "RETURN_HOVER" in phases_seen
    else:
        assert ctrl.catches == 1
        assert ctrl.misses == 0
        assert ctrl.phase == "DONE"
        for expected in ("PRE_POSITION", "SYNC_ACCEL", "DESCEND", "GRASP", "LIFT", "TRANSIT_TO_CASE", "DROP_RELEASE"):
            assert expected in phases_seen


def test_controller_does_not_stall_on_stopped_upstream_object() -> None:
    calib = HolesawBeltCalib()
    ctrl = HolesawConveyorController(calib, target_count=1, verify_grasp=False)
    q_sim = ctrl.ready_joints().astype(np.float64)
    dt = 1.0 / 30.0

    # First 2 seconds: an object at the left edge drifts slightly then stops (e.g. belt paused).
    for step in range(60):
        t = step * dt
        u_stall = int(round(30 + min(t, 0.35) * 45.0))
        frame = _make_belt_frame(calib, white_saws=[(u_stall, 180, 18, 20)])
        q_cmd, info = ctrl.step(t, frame, q_sim)
        q_sim[:6] = q_cmd[:6]
        assert info["ik_pos_err_mm"] < 5.0

    # Then a real moving hole saw arrives while the stalled one remains parked at the edge.
    for step in range(60, 300):
        t = step * dt
        t_move = (step - 60) * dt
        u = int(round(120 + 95.0 * t_move))
        v = int(round(145 - 18.0 * t_move))
        saws = [(46, 180, 18, 20)]
        if 15 <= u <= 620:
            saws.append((u, v, 18, 20))
        frame = _make_belt_frame(calib, white_saws=saws)
        q_cmd, info = ctrl.step(t, frame, q_sim)
        q_sim[:6] = q_cmd[:6]
        assert info["ik_pos_err_mm"] < 5.0
        if ctrl.phase == "DONE":
            break

    assert ctrl.catches == 1
    assert ctrl.phase == "DONE"


def test_tracker_locks_at_10x_belt_speed() -> None:
    """Verify 1-point corridor association and fast lock work at ~0.95 m/s (10x nominal)."""
    calib = HolesawBeltCalib()
    tracker = MultiHolesawTracker(calib)
    # 60 Hz camera frames of a saw moving at ~950 px/s (~0.95 m/s).
    for k in range(10):
        t = k * (1.0 / 60.0)
        u = int(round(40 + 950.0 * t))
        v = int(round(170 - 180.0 * t))
        if u > 440:
            break
        frame = _make_belt_frame(calib, white_saws=[(u, v, 18, 20)], yellow_triangles=[(260, 120, 20)])
        tracker.update(t, detect_holesaws(frame, calib))

    locked = [tr for tr in tracker.tracks if tr.locked]
    assert len(locked) == 1
    assert -1.15 < locked[0].vy < -0.75
