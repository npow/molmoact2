"""Orchestral prediction sessions and the local stdio bridge for YAM.

The SDK is imported lazily so the robot can use a separate Python 3.12 runtime.
"""

from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# BimanualYAM contract facts (single source of truth for both processes)
# ---------------------------------------------------------------------------

#: Physical camera roles on the YAM rig. The selected client adapter maps these
#: roles to the deployment's advertised input names.
CAMERA_KEYS: Tuple[str, str, str] = ("top", "left", "right")
STATE_DIM = 14
ACTION_HORIZON = 30
ACTION_SPACE = "joint_position"
NOISE_SCHEME = "numpy-pcg64-15x32-fp32-v1"

#: Environment variable naming a Python >= 3.12 interpreter that can import the
#: official ``orchestral`` package, used when the robot runtime cannot.
ORCHESTRAL_PYTHON_ENV = "ORCHESTRAL_PYTHON"

#: SDK image encoders supported for both managed and direct sessions.
OBSERVATION_ENCODINGS: Tuple[str, ...] = ("jpeg", "h264")

#: Raw bridge frames are always contiguous ``HxWx3`` ``uint8``: the one shape
#: every camera in this rig produces and the only one the SDK's capture path
#: takes without a conversion of its own.
RAW_FRAME_DTYPE = "uint8"

_FRAME_MAGIC = b"SVYB"
_FRAME_PREFIX = struct.Struct("<4sII")
_MAX_BUFFERS = 16


class OrchestralBridgeError(RuntimeError):
    """An Orchestral session/bridge failure with an optional remote error type."""

    def __init__(self, message: str, *, error_type: Optional[str] = None):
        super().__init__(message)
        self.error_type = error_type


# ---------------------------------------------------------------------------
# Local parent/child framing
# ---------------------------------------------------------------------------


def frame_parts(header: Mapping[str, Any], buffers: Sequence[bytes] = ()) -> List[Any]:
    """The pieces of one frame in wire order, ready to be written in sequence.

    Returned as pieces rather than one joined blob because the raw-pixel wire
    moves ~2 MB per act (three 360x640x3 frames). Joining first would copy
    every one of those bytes an extra time, on the act hot path, to produce a
    buffer the pipe then copies again -- so the caller writes the pieces and
    the pixels are read straight out of the capture buffer.
    """
    if len(buffers) > _MAX_BUFFERS:
        raise ValueError(f"a bridge frame carries at most {_MAX_BUFFERS} buffers")
    for buffer in buffers:
        # memoryview is a first-class payload here: the raw-pixel wire hands
        # over a zero-copy view of the capture buffer rather than a copy of it.
        if not isinstance(buffer, (bytes, bytearray, memoryview)) or not len(buffer):
            raise ValueError("bridge frame buffers must be non-empty bytes")
        if len(buffer) > 0xFFFFFFFF:
            raise ValueError("bridge frame buffer exceeds uint32 framing")
    encoded_header = json.dumps(
        dict(header),
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if len(encoded_header) > 0xFFFFFFFF:
        raise ValueError("bridge frame header exceeds uint32 framing")
    lengths = struct.pack(f"<{len(buffers)}I", *(len(buffer) for buffer in buffers))
    return [
        _FRAME_PREFIX.pack(_FRAME_MAGIC, len(encoded_header), len(buffers)),
        lengths,
        encoded_header,
        *buffers,
    ]


def encode_frame(header: Mapping[str, Any], buffers: Sequence[bytes] = ()) -> bytes:
    """Encode one ``(JSON header, raw buffers)`` frame as a single blob."""
    return b"".join(bytes(part) for part in frame_parts(header, buffers))


def raw_frame_spec(buffer_index: int, height: int, width: int) -> Dict[str, Any]:
    """Describe one raw camera buffer in an ``act`` header."""
    return {
        "buffer": int(buffer_index),
        "height": int(height),
        "width": int(width),
        "dtype": RAW_FRAME_DTYPE,
    }


def raw_frame_array(data: bytes, spec: Mapping[str, Any]) -> Any:
    """Rebuild one ``HxWx3`` ``uint8`` frame from a bridge buffer, without a copy.

    ``numpy`` is imported lazily: this module must stay importable on the
    robot runtime with nothing but the standard library at module level.
    """
    import numpy as np

    dtype = str(spec.get("dtype", RAW_FRAME_DTYPE))
    if dtype != RAW_FRAME_DTYPE:
        raise OrchestralBridgeError(
            f"raw bridge frames must be {RAW_FRAME_DTYPE}, got {dtype!r}"
        )
    try:
        height = int(spec["height"])
        width = int(spec["width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise OrchestralBridgeError("raw bridge frame is missing its height/width") from exc
    expected = height * width * 3
    if len(data) != expected:
        # A truncated or mis-shaped buffer must never be silently reinterpreted:
        # a reshape that happens to fit would hand the model a sheared image and
        # nothing downstream could tell.
        raise OrchestralBridgeError(
            f"raw bridge frame carries {len(data)} bytes, expected {expected} "
            f"for {height}x{width}x3 {RAW_FRAME_DTYPE}"
        )
    return np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)


def _frame_is_empty(value: Any) -> bool:
    """True for a camera slot carrying no pixels -- bytes or array alike."""
    if value is None:
        return True
    if isinstance(value, (bytes, bytearray, memoryview)):
        return len(value) == 0
    size = getattr(value, "size", None)
    if size is not None:
        return int(size) == 0
    return not value


def missing_cameras(images: Mapping[str, Any]) -> List[str]:
    """Camera keys with no usable frame, for either payload shape.

    A plain ``not images.get(key)`` cannot do this job any more: a numpy array
    has no truth value, so the check that exists to *report* a stalled camera
    would itself raise instead.
    """
    return [key for key in CAMERA_KEYS if _frame_is_empty(images.get(key))]


def _observation_frame(value: Any) -> Any:
    """Hand the SDK what it must own.

    Pre-encoded bytes ride the stateless JPEG wire verbatim. Raw pixels are
    passed through untouched so ``capture_observation`` fits them and the
    session's own encoder mints the payload -- the only order in which a
    stateful wire is safe.
    """
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    return value


def _pixels(value: Any) -> Any:
    """``HxWx3`` ``uint8`` array; ``Session.predict`` takes arrays, not jpeg bytes."""
    import io

    import numpy as np
    from PIL import Image

    if isinstance(value, (bytes, bytearray, memoryview)):
        return np.asarray(Image.open(io.BytesIO(value)).convert("RGB"))
    return np.asarray(value, dtype=np.uint8)


def _checkpoint_input_names(contract: Any, policy_adapter="identity") -> Tuple[Dict[str, str], str]:
    """Map ``CAMERA_KEYS`` and the state vector onto the contract's input names.

    Cameras match an image input whose name ends in ``.<key>`` (or is ``<key>``).
    The physical overhead camera is called ``top`` locally and ``middle`` by
    native pi0.5 contracts; both names refer to that same camera.
    the state matches the single state input. Anything this embodiment cannot
    fill, or leaves unfilled, is refused here rather than by the SDK's
    validator, whose message would not say which side is wrong.
    """
    features = list(contract.inputs or ())
    if not features or any(f.name is None for f in features):
        raise OrchestralBridgeError(
            f"checkpoint input contract is not fully named (resolution={contract.resolution!r})"
        )
    images = [f.name for f in features if f.modality == "image"]
    states = [f.name for f in features if f.modality == "state"]
    if policy_adapter == "pi05_yam_red_cap":
        cameras = {"top": "base_0_rgb", "left": "left_wrist_0_rgb", "right": "right_wrist_0_rgb"}
        if set(images) != set(cameras.values()) or states != ["state"]:
            raise OrchestralBridgeError("pi05_yam_red_cap requires OpenPI model slots and state; import with policy_config=none")
        return cameras, "state"
    cameras: Dict[str, str] = {}
    for key in CAMERA_KEYS:
        aliases = {"top", "middle"} if key == "top" else {key}
        matches = [name for name in images if name.rsplit(".", 1)[-1] in aliases]
        if len(matches) != 1:
            raise OrchestralBridgeError(
                f"checkpoint input contract has {len(matches)} image inputs for camera "
                f"{key!r} (image inputs: {images})"
            )
        cameras[key] = matches[0]
    unused = sorted(set(images) - set(cameras.values()))
    if unused or len(states) != 1:
        raise OrchestralBridgeError(
            f"checkpoint input contract does not match cameras {list(CAMERA_KEYS)} plus "
            f"one state vector: unmatched image inputs {unused}, state inputs {states}"
        )
    return cameras, states[0]


def _checkpoint_geometry(contract: Any, policy_adapter="identity") -> Dict[str, Any]:
    """Read physical dimensions and camera geometry from the deployed contract."""
    camera_names, state_name = _checkpoint_input_names(contract, policy_adapter)
    features = {feature.name: feature for feature in contract.inputs}
    result: Dict[str, Any] = {"camera_inputs": {}}
    state_shape = getattr(features[state_name], "shape", None)
    outputs = _jsonable(getattr(contract, "outputs", None) or {})
    action_shape = (outputs.get("actions") or {}).get("shape")
    if state_shape is not None or action_shape is not None:
        if (not state_shape or len(state_shape) != 1 or state_shape[0] not in (7, 14)
                or not action_shape or len(action_shape) != 2
                or action_shape[1] != state_shape[0]
                or not isinstance(action_shape[0], int) or action_shape[0] <= 0):
            raise OrchestralBridgeError(
                f"unsupported checkpoint state/action shapes: {state_shape}, {action_shape}"
            )
        result.update(state_dim=state_shape[0], action_dim=action_shape[1],
                      action_horizon=action_shape[0])
    for key, name in camera_names.items():
        feature = features[name]
        shape = getattr(feature, "shape", None)
        if shape is None:
            continue
        layout = getattr(feature, "layout", None) or "HWC"
        if len(shape) != 3 or layout not in ("HWC", "CHW"):
            raise OrchestralBridgeError(f"unsupported checkpoint image shape/layout for {name}: {shape}/{layout}")
        height, width, channels = shape if layout == "HWC" else (shape[1], shape[2], shape[0])
        if channels != 3 or getattr(feature, "dtype", "uint8") not in (None, "uint8"):
            raise OrchestralBridgeError(f"checkpoint image {name} must accept uint8 RGB")
        result["camera_inputs"][key] = dict(height=height, width=width, layout=layout)
    return result


def _checkpoint_pixels(value: Any, geometry: Mapping[str, Any]) -> Any:
    """Fit raw RGB pixels to the checkpoint tensor; H.264 encoding stays in Orchestral."""
    import numpy as np
    from PIL import Image

    pixels = _pixels(value)
    height, width = geometry.get("height"), geometry.get("width")
    if height and width and pixels.shape[:2] != (height, width):
        source_h, source_w = pixels.shape[:2]
        ratio = max(source_w / width, source_h / height)
        size = (max(1, int(source_w / ratio)), max(1, int(source_h / ratio)))
        resized = Image.fromarray(pixels).resize(size, Image.Resampling.BILINEAR)
        canvas = Image.new("RGB", (width, height))
        canvas.paste(resized, ((width - size[0]) // 2, (height - size[1]) // 2))
        pixels = np.asarray(canvas)
    if geometry.get("layout") == "CHW":
        pixels = pixels.transpose(2, 0, 1)
    return np.ascontiguousarray(pixels)


def _read_exactly(stream: Any, size: int) -> bytes:
    """Read exactly ``size`` bytes or raise; pipes may return short reads."""
    chunks: List[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError("Orchestral bridge stream closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: Any) -> Tuple[Dict[str, Any], List[bytes]]:
    """Read one frame written by :func:`encode_frame`. Raises ``EOFError`` at end."""
    prefix = stream.read(_FRAME_PREFIX.size)
    if not prefix:
        raise EOFError("Orchestral bridge stream closed")
    if len(prefix) < _FRAME_PREFIX.size:
        prefix += _read_exactly(stream, _FRAME_PREFIX.size - len(prefix))
    magic, header_len, buffer_count = _FRAME_PREFIX.unpack(prefix)
    if magic != _FRAME_MAGIC:
        raise ValueError("Orchestral bridge frame magic does not match")
    if buffer_count > _MAX_BUFFERS:
        raise ValueError("Orchestral bridge frame declares too many buffers")
    lengths = (
        struct.unpack(f"<{buffer_count}I", _read_exactly(stream, 4 * buffer_count))
        if buffer_count
        else ()
    )
    header = json.loads(_read_exactly(stream, header_len).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("Orchestral bridge frame header must be a JSON object")
    return header, [_read_exactly(stream, length) for length in lengths]


def write_frame(stream: Any, header: Mapping[str, Any], buffers: Sequence[bytes] = ()) -> None:
    for part in frame_parts(header, buffers):
        stream.write(part)
    stream.flush()


# ---------------------------------------------------------------------------
# The official-SDK half
# ---------------------------------------------------------------------------


class OrchestralSessionHost:
    """One Orchestral prediction session for the entire evaluation."""

    def __init__(self, *, deployment_id: str, instruction=None, observation_encoding="h264", policy_adapter="identity"):
        if not deployment_id:
            raise OrchestralBridgeError("a managed Orchestral deployment id is required")
        if observation_encoding not in OBSERVATION_ENCODINGS:
            raise OrchestralBridgeError(f"unsupported observation encoding: {observation_encoding!r}")
        if policy_adapter not in {"identity", "pi05_yam_red_cap"}:
            raise OrchestralBridgeError(f"unsupported policy_adapter: {policy_adapter!r}")
        self.policy_adapter = policy_adapter
        self.deployment_id = deployment_id
        self.instruction = instruction
        self.observation_encoding = observation_encoding
        self._client = self._session = None
        self._episode_start = True
        self.identity = {}

    def _configure(self, spec):
        from types import SimpleNamespace
        contract = SimpleNamespace(
            inputs=[SimpleNamespace(name=name, modality="image", shape=(h, w, 3),
                                    dtype="uint8", layout="HWC")
                    for name, (h, w) in spec.cameras.items()] +
                   [SimpleNamespace(name="state", modality="state", shape=(spec.state_dim,))],
            outputs={"actions": {"shape": list(spec.action_shape)}},
            declared={}, resolution="resolved",
        )
        if getattr(spec, "modalities", {}):
            raise OrchestralBridgeError("YAM cannot supply additional deployment modalities")
        geometry = _checkpoint_geometry(contract, self.policy_adapter)
        self._camera_names, _ = _checkpoint_input_names(contract, self.policy_adapter)
        if self.policy_adapter == "pi05_yam_red_cap" and (
            geometry["state_dim"] != 7 or geometry["action_dim"] != 7
            or geometry["action_horizon"] != 15
        ):
            raise OrchestralBridgeError("pi05_yam_red_cap requires 7 state values and a 15x7 action chunk")
        self.identity.update(geometry, policy_adapter=self.policy_adapter)

    def open(self):
        if self._session is not None:
            return dict(self.identity)
        from orchestral import Orchestral
        client = Orchestral()
        try:
            deployment = client.deployments.get(self.deployment_id)
            self._configure(deployment.io_spec)
            session = deployment.session(instruction=self.instruction or "", encoder="mjpeg" if self.observation_encoding == "jpeg" else "h264")
        except BaseException:
            client._http.close()
            raise
        self._client, self._session = client, session
        self.identity.update(deployment_id=self.deployment_id, session_id=session.session_id,
                             model_ref=deployment.model_id, base_url=client.base_url,
                             observation_encoding=self.observation_encoding)
        return dict(self.identity)

    def act(self, images, state, instruction=None, noise_seed=None):
        import numpy as np
        from dataclasses import asdict
        if self._session is None:
            raise OrchestralBridgeError("Orchestral session is not open")
        missing = missing_cameras(images)
        if missing:
            raise OrchestralBridgeError(f"Orchestral observation is missing camera bytes: {missing}")
        if len(state) != self.identity["state_dim"]:
            raise OrchestralBridgeError(f"Orchestral endpoint state must be {self.identity['state_dim']} floats, got {len(state)}")
        inputs = {self._camera_names[key]: _checkpoint_pixels(images[key], self.identity["camera_inputs"][key])
                  for key in CAMERA_KEYS}
        query_state = np.asarray(state, dtype=np.float32).copy()
        if query_state.shape != (self.identity["state_dim"],) or not np.isfinite(query_state).all():
            raise OrchestralBridgeError("state must be a finite vector matching the endpoint")
        inputs["state"] = query_state.copy()
        if self.policy_adapter == "pi05_yam_red_cap":
            inputs["state"][6] = 1.0 - np.clip(query_state[6], 0.0, 1.0)
        result = self._session.predict(inputs=inputs, instruction=instruction or self.instruction,
                                       episode_start=self._episode_start, noise_seed=noise_seed)
        self._episode_start = False
        actions = np.asarray(result.actions, dtype=np.float32).copy()
        if actions.shape != (self.identity["action_horizon"], self.identity["action_dim"]) or not np.isfinite(actions).all():
            raise OrchestralBridgeError("actions must be finite and match the endpoint shape")
        if self.policy_adapter == "pi05_yam_red_cap":
            actions[:, :6] += query_state[:6]
            actions[:, 6] = np.clip(1.0 - actions[:, 6], 0.0, 1.0)
        timings = asdict(result.timings)
        return {"actions": actions.tolist(), "horizon": len(actions),
                "action_space": ACTION_SPACE,
                "telemetry": {**timings, "server_infer_ms": result.timings.accel_exec_us / 1000,
                              "trace_id": result.trace_id,
                              "noise_scheme": NOISE_SCHEME if noise_seed is not None else None,
                              "noise_seed_acknowledged": noise_seed is not None},
                "binding": None, "safety_signal": None}

    def begin_episode(self):
        if self._session is None:
            raise OrchestralBridgeError("begin_episode requested before open")
        self._episode_start = True
        return True

    def close(self, success=True):
        session, self._session = self._session, None
        client, self._client = self._client, None
        try:
            if session is not None:
                session.close()
        finally:
            if client is not None:
                client._http.close()


class OrchestralDirectHost(OrchestralSessionHost):
    """A standalone Orchestral ConnectGrant session."""

    def __init__(self, *, grant, instruction=None, timeout_sec=None,
                 observation_encoding="h264", h264_crf=None, policy_adapter="identity"):
        super().__init__(deployment_id="self-hosted", instruction=instruction,
                         observation_encoding=observation_encoding, policy_adapter=policy_adapter)
        if h264_crf is not None:
            raise OrchestralBridgeError("Orchestral does not expose h264_crf; unset this option")
        self._grant_path = grant

    def open(self):
        if self._session is not None:
            return dict(self.identity)
        from orchestral import ConnectGrant, Session
        path = Path(self._grant_path).expanduser()
        if path.stat().st_mode & 0o077:
            raise OrchestralBridgeError(f"Orchestral grant {path} is group/world readable; chmod 600 it")
        grant = ConnectGrant.from_token(path.read_text().strip())
        self._configure(grant.io_spec)
        session = Session(grant, instruction=self.instruction or "", encoder="mjpeg" if self.observation_encoding == "jpeg" else "h264")
        self._session = session
        self.identity.update(deployment_id=grant.deployment_id, session_id=session.session_id,
                             observation_encoding=self.observation_encoding)
        return dict(self.identity)


def _jsonable(value: Any) -> Any:
    """Reduce SDK payloads to JSON the parent process can read back verbatim.

    Non-finite floats (e.g. a runtime-reported ``inf`` rate) are flattened to
    ``None`` rather than left for ``json.dumps(allow_nan=False)`` to reject —
    telemetry must never be able to kill the bridge mid-run.
    """
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return str(value)


# ---------------------------------------------------------------------------
# stdio server (child process)
# ---------------------------------------------------------------------------


def _handle(host_ref: Dict[str, Any], header: Dict[str, Any], buffers: List[bytes]) -> Dict[str, Any]:
    op = header.get("op")
    host: Optional[OrchestralSessionHost] = host_ref.get("host")
    if op == "open":
        if host is not None:
            raise OrchestralBridgeError("a Orchestral session is already open on this bridge")
        observation_encoding = header.get("observation_encoding") or "h264"
        if header.get("grant"):
            host = OrchestralDirectHost(
                grant=header["grant"],
                instruction=header.get("instruction"),
                timeout_sec=header.get("timeout_sec"),
                observation_encoding=observation_encoding,
                policy_adapter=header.get("policy_adapter", "identity"),
                h264_crf=header.get("h264_crf"),
            )
        else:
            host = OrchestralSessionHost(
                deployment_id=header["deployment_id"],
                instruction=header.get("instruction"),
                observation_encoding=observation_encoding,
                policy_adapter=header.get("policy_adapter", "identity"),
            )
        identity = host.open()
        host_ref["host"] = host
        return {"ok": True, "identity": identity}
    if op == "act":
        if host is None:
            raise OrchestralBridgeError("act requested before open")
        frame_index = header.get("frames")
        if frame_index:
            images = {
                key: raw_frame_array(buffers[int(spec["buffer"])], spec)
                for key, spec in frame_index.items()
            }
        else:
            image_index = header.get("images") or {}
            images = {key: buffers[int(index)] for key, index in image_index.items()}
        return {
            "ok": True,
            "prediction": host.act(
                images,
                header.get("state") or [],
                instruction=header.get("instruction"),
                noise_seed=header.get("noise_seed"),
            ),
        }
    if op == "begin_episode":
        if host is None:
            raise OrchestralBridgeError("begin_episode requested before open")
        return {"ok": True, "honoured": host.begin_episode()}
    if op == "close":
        if host is not None:
            host.close(success=bool(header.get("success", True)))
            host_ref["host"] = None
        return {"ok": True}
    if op == "ping":
        try:
            import orchestral  # noqa: F401
            import PIL.Image  # noqa: F401

            orchestral_importable = True
        except ImportError:
            orchestral_importable = False
        return {
            "ok": True,
            "executable": sys.executable,
            "python_version": sys.version.split()[0],
            "orchestral_importable": orchestral_importable,
        }
    raise OrchestralBridgeError(f"unknown Orchestral bridge op {op!r}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Serve bridge requests on stdin/stdout until the parent closes the pipe."""
    del argv
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer
    host_ref: Dict[str, Any] = {"host": None}
    try:
        while True:
            try:
                header, buffers = read_frame(stdin)
            except EOFError:
                break
            except KeyboardInterrupt:
                # Ctrl-C reaches this subprocess directly (same process
                # group as the parent launcher). Exit quietly through the
                # same path as a closed pipe instead of an unhandled
                # traceback; ``finally`` below still closes the host.
                break
            try:
                response = _handle(host_ref, header, buffers)
            except KeyboardInterrupt:
                # Same as the idle-read case above: Ctrl-C can also land
                # mid-request (e.g. inside an in-flight network act()).
                # Exit quietly rather than propagate a raw traceback; the
                # parent already treats an interrupted request as failed.
                break
            except Exception as exc:  # noqa: BLE001 - reported to the parent
                response = {
                    "ok": False,
                    "error": str(exc) or exc.__class__.__name__,
                    "error_type": exc.__class__.__name__,
                }
            # Echo the request id so the parent can never pair a late reply
            # with a later observation.
            response["request_id"] = header.get("request_id")
            try:
                write_frame(stdout, response)
            except (TypeError, ValueError) as exc:
                # A response payload that cannot be JSON-encoded (e.g. a
                # non-finite float that slipped past ``_jsonable``) must not
                # kill this process — the caller degrades to a failed request,
                # not a dead session.
                write_frame(
                    stdout,
                    {
                        "ok": False,
                        "error": f"Orchestral bridge could not encode its response: {exc}",
                        "error_type": exc.__class__.__name__,
                        "request_id": header.get("request_id"),
                    },
                )
    finally:
        host = host_ref.get("host")
        if host is not None:
            try:
                host.close(success=False)
            except Exception:  # noqa: BLE001 - teardown is best effort
                pass
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
