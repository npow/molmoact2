"""Drive a tool-pose checkpoint from its declared semantics (Orchestral ``io_spec.semantics``, version 1).

A checkpoint that returns tool poses declares what each value means: a tool position and a tool
rotation (in a stated encoding) of a named tool frame, a gripper with its open and closed values, and
padding. A tool frame is a link of a robot description, pinned by the file's sha256, moved by a fixed
``xyz``/``rpy`` offset and expressed in a base link. This module turns such a declaration into this rig's
joint commands with the MuJoCo kinematics of ``tool_pose_actions.py``, so a checkpoint needs no
family-specific code here. The vocabulary is ``specs/2026-11-22-action-semantics-contract`` in
Orchestral; Orchestral's SDK converts joint-space checkpoints and leaves tool poses to the robot client.

    poses = DeclaredToolPoses(semantics, state_dim=7, action_shape=(32, 10),
                              descriptions=CONFIGS, joint_names=JOINTS, max_joint_step_rad=0.12)
    state = poses.state(measured)              # the model's state for [joint1..6, gripper]
    joints = poses.actions(chunk, measured)    # (<= K, 7) joint targets and gripper, this rig's convention
    prefix = poses.committed(queued_joints)    # queued joint commands as the model's own rows
"""
from __future__ import annotations

import hashlib
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

#: Width of each tool rotation encoding. All describe R_base_tool, the tool frame's orientation in its base.
ROTATION_WIDTHS = {"axis_angle": 3, "matrix_rows": 6, "matrix": 9, "quaternion_wxyz": 4, "quaternion_xyzw": 4}
_TOOL_BODY = "declared_tool"


def _quaternions(rotations):
    """Rotation matrices ``(N, 3, 3)`` to unit quaternions ``(N, 4)`` as ``w, x, y, z`` with ``w >= 0``."""
    r = np.asarray(rotations, dtype=np.float64)
    trace = np.trace(r, axis1=1, axis2=2)
    # The largest of 1 + trace and 1 + 2 R_ii - trace picks the numerically stable branch (Shepperd).
    candidates = np.stack([1 + trace, *(1 + 2 * r[:, i, i] - trace for i in range(3))], axis=1)
    q = np.empty((len(r), 4))
    for n, (m, branch) in enumerate(zip(r, np.argmax(candidates, axis=1))):
        s = 2 * np.sqrt(candidates[n, branch])
        if branch == 0:
            q[n] = [s / 4, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
        elif branch == 1:
            q[n] = [(m[2, 1] - m[1, 2]) / s, s / 4, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
        elif branch == 2:
            q[n] = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, s / 4, (m[1, 2] + m[2, 1]) / s]
        else:
            q[n] = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, s / 4]
    return q * np.where(q[:, :1] < 0, -1.0, 1.0)


def encode_rotations(rotations, encoding):
    """Rotation matrices ``(N, 3, 3)`` as rows of a declared encoding."""
    r = np.asarray(rotations, dtype=np.float64)
    if encoding == "matrix_rows":
        return r[:, :2].reshape(len(r), 6)
    if encoding == "matrix":
        return r.reshape(len(r), 9)
    q = _quaternions(r)
    if encoding == "quaternion_wxyz":
        return q
    if encoding == "quaternion_xyzw":
        return q[:, [1, 2, 3, 0]]
    if encoding == "axis_angle":
        norm = np.linalg.norm(q[:, 1:], axis=1, keepdims=True)
        return q[:, 1:] * np.where(norm > 1e-12, 2 * np.arctan2(norm, q[:, :1]) / np.maximum(norm, 1e-12), 2.0)
    raise ValueError(f"unknown rotation encoding {encoding!r}")


def decode_rotations(values, encoding):
    """Rows of a declared encoding to the nearest rotation matrices ``(N, 3, 3)``."""
    v = np.asarray(values, dtype=np.float64)
    if encoding == "matrix_rows":
        first = v[:, :3] / np.linalg.norm(v[:, :3], axis=1, keepdims=True)
        second = v[:, 3:6] - np.sum(first * v[:, 3:6], axis=1, keepdims=True) * first
        second /= np.linalg.norm(second, axis=1, keepdims=True)
        return np.stack([first, second, np.cross(first, second)], axis=1)
    if encoding == "matrix":
        u, _, vt = np.linalg.svd(v.reshape(-1, 3, 3))
        u[:, :, 2] *= np.where(np.linalg.det(u @ vt) < 0, -1.0, 1.0)[:, None]
        return u @ vt
    if encoding == "axis_angle":
        angle = np.linalg.norm(v, axis=1)
        x, y, z = (v / np.maximum(angle, 1e-12)[:, None]).T
        zero = np.zeros_like(x)
        k = np.stack([np.stack([zero, -z, y], 1), np.stack([z, zero, -x], 1), np.stack([-y, x, zero], 1)], 1)
        return np.eye(3) + np.sin(angle)[:, None, None] * k + (1 - np.cos(angle))[:, None, None] * (k @ k)
    if encoding in ("quaternion_wxyz", "quaternion_xyzw"):
        q = v if encoding == "quaternion_wxyz" else v[:, [3, 0, 1, 2]]
        w, x, y, z = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
        return np.stack([
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], axis=1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], axis=1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], axis=1),
        ], axis=1)
    raise ValueError(f"unknown rotation encoding {encoding!r}")


def _description(digest, descriptions):
    """The robot description file among ``descriptions`` (files or directories of .urdf) with this sha256."""
    candidates = []
    for entry in map(Path, descriptions):
        candidates += sorted(entry.glob("*.urdf")) if entry.is_dir() else [entry]
    for path in candidates:
        if hashlib.sha256(path.read_bytes()).hexdigest() == digest:
            return path
    raise ValueError(f"no robot description with sha256 {digest} among {[str(p) for p in candidates]}")


def tool_kinematics(frame, *, descriptions, joint_names, max_joint_step_rad):
    """``ToolPoseActions`` whose tool body is exactly the declared tool frame.

    The description must be byte-identical to the declared one. Its visual and collision geometry is
    dropped (kinematics only), and the tool frame is added as a link fixed to ``link`` at ``xyz``/``rpy``.
    """
    from tool_pose_actions import ToolPoseActions

    root = ET.fromstring(_description(frame["urdf_sha256"], descriptions).read_bytes())
    links = {link.get("name") for link in root.iter("link")}
    if frame["link"] not in links or frame["base_link"] not in links or _TOOL_BODY in links:
        raise ValueError(f"the robot description has no links {frame['base_link']!r} and {frame['link']!r}")
    for link in root.iter("link"):
        for geometry in [*link.findall("visual"), *link.findall("collision")]:
            link.remove(geometry)
    ET.SubElement(root, "link", name=_TOOL_BODY)
    joint = ET.SubElement(root, "joint", name=f"{frame['link']}_to_{_TOOL_BODY}", type="fixed")
    ET.SubElement(joint, "parent", link=frame["link"])
    ET.SubElement(joint, "child", link=_TOOL_BODY)
    ET.SubElement(joint, "origin", xyz=" ".join(map(repr, map(float, frame["xyz"]))),
                  rpy=" ".join(map(repr, map(float, frame["rpy"]))))
    with tempfile.NamedTemporaryFile("wb", suffix=".urdf", delete=False) as file:
        file.write(ET.tostring(root))
    try:
        return ToolPoseActions(model_path=file.name, base_body=frame["base_link"], tool_body=_TOOL_BODY,
                               joint_names=joint_names, max_joint_step_rad=max_joint_step_rad)
    finally:
        Path(file.name).unlink()


class DeclaredToolPoses:
    """One limb's joints to and from a checkpoint's declared tool-pose state and action rows.

    ``group`` names the limb when the checkpoint declares more than one. ``robot_gripper`` is this rig's
    ``(open, closed)`` aperture values; ``gripper_tolerance`` is how far outside its declared range a
    predicted gripper value may land before it is refused (values within it are clipped). The robot's
    state is ``[joint1..6, gripper]`` in this rig's convention, as ``tool_pose_actions`` expects.
    """

    def __init__(self, semantics, *, state_dim, action_shape, descriptions, joint_names, max_joint_step_rad,
                 group=None, robot_gripper=(1.0, 0.0), gripper_tolerance=0.0):
        if not isinstance(semantics, dict):
            raise ValueError("this deployment declares no semantics; tool poses cannot be converted without them")
        self.state_segments = self._segments(semantics, "state", state_dim)
        self.action_segments = self._segments(semantics, "action", action_shape[1])
        groups = {s.get("group", "single") for s in (*self.state_segments, *self.action_segments)
                  if s["kind"] in ("tool_position", "tool_rotation")}
        if group is None and len(groups) != 1:
            raise ValueError(f"this checkpoint declares tool poses for {sorted(groups)}; choose one with group=")
        self.group = group or next(iter(groups))
        tools = set()
        for label, segments in (("state", self.state_segments), ("action", self.action_segments)):
            driven = [s for s in segments if self._driven(s)]
            kinds = sorted(s["kind"] for s in driven if s["kind"] != "gripper")
            if kinds != ["tool_position", "tool_rotation"] or sum(s["kind"] == "gripper" for s in driven) > 1:
                raise ValueError(f"{label} needs one tool_position, one tool_rotation and at most one gripper "
                                 f"for {self.group!r}; it declares {[s['kind'] for s in driven]}")
            for segment in driven:
                if label == "action" and segment.get("reference") != "absolute":
                    raise ValueError(f"action segment {segment['name']!r} is not absolute")
                if segment["kind"] == "tool_rotation" and ROTATION_WIDTHS.get(segment.get("encoding")) != segment["width"]:
                    raise ValueError(f"segment {segment['name']!r} has encoding {segment.get('encoding')!r}, "
                                     f"which is not {segment['width']} values wide")
                if segment["kind"] == "gripper" and segment.get("open_value") == segment.get("closed_value"):
                    raise ValueError(f"gripper segment {segment['name']!r} needs distinct open and closed values")
            tools |= {s["tool"] for s in driven if s["kind"] != "gripper"}
        if len(tools) != 1 or next(iter(tools)) not in semantics.get("tools", {}):
            raise ValueError(f"state and actions must be poses of one declared tool frame; they name {sorted(tools)}")
        self.tool = next(iter(tools))
        self.tool_frame = semantics["tools"][self.tool]
        self.kinematics = tool_kinematics(self.tool_frame, descriptions=descriptions, joint_names=joint_names,
                                          max_joint_step_rad=max_joint_step_rad)
        self.robot_gripper = tuple(float(v) for v in robot_gripper)
        self.gripper_tolerance = float(gripper_tolerance)
        self.horizon = int(action_shape[0])

    @staticmethod
    def _segments(semantics, label, width):
        segments = semantics.get(label)
        if not isinstance(segments, list) or sum(s.get("width", 0) for s in segments) != width:
            raise ValueError(f"{label} segments must cover exactly {width} values")
        for segment in segments:
            if segment.get("kind") not in ("tool_position", "tool_rotation", "gripper", "unused", "joint_position"):
                raise ValueError(f"{label} segment {segment.get('name')!r} has kind {segment.get('kind')!r}, "
                                 "which this rig cannot convert")
        return segments

    def _driven(self, segment):
        return segment.get("group", "single") == self.group and segment["kind"] != "unused"

    def _to_model_gripper(self, aperture, segment):
        (robot_open, robot_closed), model = self.robot_gripper, (segment["open_value"], segment["closed_value"])
        openness = (np.asarray(aperture, np.float64) - robot_closed) / (robot_open - robot_closed)
        return model[1] + openness * (model[0] - model[1])

    def _to_robot_aperture(self, values, segment):
        low, high = sorted((segment["open_value"], segment["closed_value"]))
        if values.min() < low - self.gripper_tolerance or values.max() > high + self.gripper_tolerance:
            raise ValueError(f"gripper values range from {values.min():.4f} to {values.max():.4f}, outside the "
                             f"declared [{low:g}, {high:g}] by more than {self.gripper_tolerance:g}")
        openness = (np.clip(values, low, high) - segment["closed_value"]) / (segment["open_value"] - segment["closed_value"])
        robot_open, robot_closed = self.robot_gripper
        return robot_closed + openness * (robot_open - robot_closed)

    def _rows(self, segments, positions, rotations, apertures):
        """Rows of ``segments`` from tool positions, rotations and robot gripper apertures."""
        columns = []
        for segment in segments:
            if not self._driven(segment):
                undriven = segment.get("undriven_value")
                if undriven is None and segment["kind"] != "unused":
                    raise ValueError(f"segment {segment['name']!r} is another limb's and declares no undriven_value")
                columns.append(np.tile(np.asarray(undriven or [0.0] * segment["width"], np.float64), (len(positions), 1)))
            elif segment["kind"] == "tool_position":
                columns.append(positions)
            elif segment["kind"] == "tool_rotation":
                columns.append(encode_rotations(rotations, segment["encoding"]))
            else:
                columns.append(self._to_model_gripper(apertures, segment).reshape(-1, 1))
        return np.concatenate(columns, axis=1).astype(np.float32)

    def _poses(self, joint_states):
        """Tool positions and rotations of joint states ``(N, 7)``, from this rig's kinematics."""
        rows = self.kinematics.committed(joint_states, matrix_policy=False)  # position, rotation rows 0-1, gripper
        return rows[:, :3].astype(np.float64), decode_rotations(rows[:, 3:9], "matrix_rows")

    def state(self, joint_state):
        """The model's state for this rig's measured ``[joint1..6, gripper]``."""
        joint_state = np.asarray(joint_state, dtype=np.float64)
        positions, rotations = self._poses(joint_state[None])
        return self._rows(self.state_segments, positions, rotations, joint_state[None, 6])[0]

    def actions(self, rows, joint_state):
        """Joint targets and gripper ``(<= K, 7)`` for the model's rows, solved from the measured joints.

        The chunk ends before the first pose out of reach or of the joint step bound; a first row out of
        reach raises ``UnreachableToolPose``. Without a declared gripper the measured aperture is kept.
        """
        rows = np.asarray(rows, dtype=np.float64)
        joint_state = np.asarray(joint_state, dtype=np.float64)
        width = sum(segment["width"] for segment in self.action_segments)
        if rows.ndim != 2 or rows.shape[1] != width or not np.isfinite(rows).all():
            raise ValueError(f"actions must be finite rows of {width} values")
        canonical = np.empty((len(rows), 10))
        canonical[:, 9] = np.clip(joint_state[6], 0, 1)
        column = 0
        for segment in self.action_segments:
            values = rows[:, column:column + segment["width"]]
            column += segment["width"]
            if not self._driven(segment):
                continue
            if segment["kind"] == "tool_position":
                canonical[:, :3] = values
            elif segment["kind"] == "tool_rotation":
                canonical[:, 3:9] = decode_rotations(values, segment["encoding"])[:, :2].reshape(-1, 6)
            else:
                canonical[:, 9] = self._to_robot_aperture(values[:, 0], segment)
        return self.kinematics.actions(canonical, joint_state, matrix_policy=False)

    def committed(self, joint_targets):
        """Queued joint commands ``(k, 7)`` as the model's own rows: their tool poses and grippers."""
        joint_targets = np.asarray(joint_targets, dtype=np.float64)
        positions, rotations = self._poses(joint_targets)
        return self._rows(self.action_segments, positions, rotations, joint_targets[:, 6])
