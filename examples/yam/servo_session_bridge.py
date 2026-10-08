"""Official Servo action-session bridge for the BimanualYAM eval client.

Everything that leaves this machine is official Servo SDK transport: one
control-plane action session (signed offer + lease) opened once and reused for
every action chunk of the run.  There is no custom ``/act`` endpoint, no
endpoint JWT, and no bespoke wire format on the network path.

The official ``servo`` SDK requires Python >= 3.12. This module works either
way, so a robot runtime that cannot import it is still able to run ``server``
mode:

* :class:`ServoSessionHost` — the actual SDK usage (``Servo`` ->
  ``deployments.get`` -> ``deployment.policy`` -> ``sv.session(policy)``).
  Imported directly when the running interpreter can already import ``servo``.
* :func:`main` — a stdio request/response server, launched by
  :mod:`molmoact_client` under a Python >= 3.12 interpreter when it cannot.

The parent/child framing below is a local implementation detail (a pipe on this
host), deliberately kept trivial: a JSON header plus binary buffers so camera
frames never take a base64 round trip between the two local processes.

Two payload shapes ride that framing, chosen by the wire the action session
negotiated:

* ``images`` -- one pre-encoded JPEG buffer per camera. The JPEG wire is
  stateless, so bytes minted anywhere are valid on any session.
* ``frames`` -- one raw ``HxWx3`` ``uint8`` buffer per camera plus its shape.
  The h264 wire is session-stateful: an access unit is only valid against the
  decoder state its predecessor left, so it is minted by the transport at SEND
  time inside the SDK and a pre-encoded payload is refused outright. Pixels are
  therefore what has to cross this pipe -- see
  ``servo.execution.action_session_transport._encode_observation``.

This module must stay importable with no ``servo`` installed: keep the module
level to the standard library and import the SDK lazily.
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

#: Servo's stable camera names for the YAM embodiment, in the order the
#: released checkpoint consumes them. Servo maps these to the exact LeRobot
#: runtime keys declared by the deployment's immutable manifest.
CAMERA_KEYS: Tuple[str, str, str] = ("top", "left", "right")
STATE_DIM = 14
ACTION_HORIZON = 30
ACTION_SPACE = "joint_position"

#: Environment variable naming a Python >= 3.12 interpreter that can import the
#: official ``servo`` package, used when the robot runtime cannot.
SERVO_PYTHON_ENV = "SERVO_PYTHON"

#: Observation wires an action session can negotiate. ``h264`` is the direct
#: endpoint default; ``jpeg`` remains available for older direct endpoints.
#: H.264 is available only on a direct
#: (``servo serve`` grant) session because the encoder is owned by that
#: session's transport.
OBSERVATION_ENCODINGS: Tuple[str, ...] = ("jpeg", "h264")

#: Raw bridge frames are always contiguous ``HxWx3`` ``uint8``: the one shape
#: every camera in this rig produces and the only one the SDK's capture path
#: takes without a conversion of its own.
RAW_FRAME_DTYPE = "uint8"

_FRAME_MAGIC = b"SVYB"
_FRAME_PREFIX = struct.Struct("<4sII")
_MAX_BUFFERS = 16


class ServoBridgeError(RuntimeError):
    """A Servo session/bridge failure with an optional remote error type."""

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
        raise ServoBridgeError(
            f"raw bridge frames must be {RAW_FRAME_DTYPE}, got {dtype!r}"
        )
    try:
        height = int(spec["height"])
        width = int(spec["width"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ServoBridgeError("raw bridge frame is missing its height/width") from exc
    expected = height * width * 3
    if len(data) != expected:
        # A truncated or mis-shaped buffer must never be silently reinterpreted:
        # a reshape that happens to fit would hand the model a sheared image and
        # nothing downstream could tell.
        raise ServoBridgeError(
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


def _checkpoint_input_contract(policy: Any) -> Any:
    """The contract ``Session.predict`` validates against, or ``None`` for ``act``.

    Servo exposes no public "takes checkpoint-named inputs" predicate, so this
    mirrors the rule in servo's ``src/servo/execution/session.py``
    (``Session.predict``): a policy with no physical observation contract but a
    binding carrying a checkpoint input contract wants checkpoint-named arrays
    instead of an images/state dict. Keep the two in step.
    """
    if policy is None or policy.observation_contract is not None:
        return None
    binding = policy.active_binding
    return binding.checkpoint_input_contract if binding is not None else None


def _checkpoint_input_names(contract: Any) -> Tuple[Dict[str, str], str]:
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
        raise ServoBridgeError(
            f"checkpoint input contract is not fully named (resolution={contract.resolution!r})"
        )
    images = [f.name for f in features if f.modality == "image"]
    states = [f.name for f in features if f.modality == "state"]
    cameras: Dict[str, str] = {}
    for key in CAMERA_KEYS:
        aliases = {"top", "middle"} if key == "top" else {key}
        matches = [name for name in images if name.rsplit(".", 1)[-1] in aliases]
        if len(matches) != 1:
            raise ServoBridgeError(
                f"checkpoint input contract has {len(matches)} image inputs for camera "
                f"{key!r} (image inputs: {images})"
            )
        cameras[key] = matches[0]
    unused = sorted(set(images) - set(cameras.values()))
    if unused or len(states) != 1:
        raise ServoBridgeError(
            f"checkpoint input contract does not match cameras {list(CAMERA_KEYS)} plus "
            f"one state vector: unmatched image inputs {unused}, state inputs {states}"
        )
    return cameras, states[0]


def _checkpoint_geometry(contract: Any) -> Dict[str, Any]:
    """Read physical dimensions and camera geometry from the deployed contract."""
    camera_names, state_name = _checkpoint_input_names(contract)
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
            raise ServoBridgeError(
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
            raise ServoBridgeError(f"unsupported checkpoint image shape/layout for {name}: {shape}/{layout}")
        height, width, channels = shape if layout == "HWC" else (shape[1], shape[2], shape[0])
        if channels != 3 or getattr(feature, "dtype", "uint8") not in (None, "uint8"):
            raise ServoBridgeError(f"checkpoint image {name} must accept uint8 RGB")
        result["camera_inputs"][key] = dict(height=height, width=width, layout=layout)
    declared = _jsonable(getattr(contract, "declared", None) or {})
    if declared.get("control_rate_hz") is not None:
        result["control_rate_hz"] = declared["control_rate_hz"]
    return result


def _checkpoint_pixels(value: Any, geometry: Mapping[str, Any]) -> Any:
    """Fit raw RGB pixels to the checkpoint tensor; H.264 encoding stays in Servo."""
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
            raise EOFError("Servo bridge stream closed mid-frame")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def read_frame(stream: Any) -> Tuple[Dict[str, Any], List[bytes]]:
    """Read one frame written by :func:`encode_frame`. Raises ``EOFError`` at end."""
    prefix = stream.read(_FRAME_PREFIX.size)
    if not prefix:
        raise EOFError("Servo bridge stream closed")
    if len(prefix) < _FRAME_PREFIX.size:
        prefix += _read_exactly(stream, _FRAME_PREFIX.size - len(prefix))
    magic, header_len, buffer_count = _FRAME_PREFIX.unpack(prefix)
    if magic != _FRAME_MAGIC:
        raise ValueError("Servo bridge frame magic does not match")
    if buffer_count > _MAX_BUFFERS:
        raise ValueError("Servo bridge frame declares too many buffers")
    lengths = (
        struct.unpack(f"<{buffer_count}I", _read_exactly(stream, 4 * buffer_count))
        if buffer_count
        else ()
    )
    header = json.loads(_read_exactly(stream, header_len).decode("utf-8"))
    if not isinstance(header, dict):
        raise ValueError("Servo bridge frame header must be a JSON object")
    return header, [_read_exactly(stream, length) for length in lengths]


def write_frame(stream: Any, header: Mapping[str, Any], buffers: Sequence[bytes] = ()) -> None:
    for part in frame_parts(header, buffers):
        stream.write(part)
    stream.flush()


# ---------------------------------------------------------------------------
# The official-SDK half
# ---------------------------------------------------------------------------


class ServoSessionHost:
    """One official Servo action session, reused for every action chunk."""

    def __init__(
        self,
        *,
        deployment_id: str,
        instruction: Optional[str] = None,
        observation_encoding: str = "h264",
    ):
        if not deployment_id:
            raise ServoBridgeError("a managed Servo deployment id is required")
        if observation_encoding not in OBSERVATION_ENCODINGS:
            raise ServoBridgeError(f"unsupported observation encoding: {observation_encoding!r}")
        self.observation_encoding = observation_encoding
        self.deployment_id = str(deployment_id)
        self.instruction = instruction
        self._client: Any = None
        self._session: Any = None
        self._policy: Any = None
        self.identity: Dict[str, Any] = {}

    def open(self) -> Dict[str, Any]:
        """Resolve the managed deployment and open one action session."""
        if self._session is not None:
            return dict(self.identity)
        try:
            from servo import Servo
            import PIL.Image  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ServoBridgeError(
                "the official servo SDK or its image dependency (Pillow/PIL) is not importable in this interpreter "
                f"({sys.executable}); install servo-client and pillow or point "
                f"{SERVO_PYTHON_ENV} at a Python >= 3.12 that has it"
            ) from exc

        client = Servo()
        deployment = client.deployments.get(self.deployment_id)
        policy = deployment.policy(instruction=self.instruction)
        binding = policy.active_binding
        # Validate the camera/state mapping before a session or robot is opened.
        # A deployment contract is available now; discovering an incompatible
        # name on the first predict would be too late for the launcher's preflight.
        try:
            contract = _checkpoint_input_contract(policy)
            geometry = _checkpoint_geometry(contract) if contract is not None else {}
        except Exception:
            client._http.close()
            raise
        session = client.session(policy, observation_encoding=self.observation_encoding)
        session.open()
        self._client = client
        self._session = session
        self._policy = policy
        # ``_jsonable`` so the identity survives the local frame header even if
        # a future SDK returns a model object for one of these fields.
        self.identity = _jsonable({
            **geometry,
            "deployment_id": self.deployment_id,
            "session_id": session.session_id,
            "eval_run_id": session.eval_run_id,
            "model_ref": policy.model_ref,
            "embodiment_id": policy.embodiment_id,
            "generation_id": getattr(binding, "generation_id", None),
            "checkpoint_digest": getattr(binding, "checkpoint_digest", None),
            "manifest_hash": getattr(binding, "manifest_hash", None),
            "binding_revision": getattr(binding, "binding_revision", None),
            "base_url": str(client._http.base_url),
            "observation_encoding": self.observation_encoding,
            "advisory": getattr(deployment, "advisory", None),
        })
        return dict(self.identity)

    def act(
        self,
        images: Mapping[str, Any],
        state: Sequence[float],
        instruction: Optional[str] = None,
        noise_seed: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run one action chunk over the open session.

        ``images`` carries pre-encoded JPEG bytes (jpeg wire) or raw ``HxWx3``
        ``uint8`` arrays (codec wire); the SDK decides what to do with each.

        Optional per-query seeds are forwarded to the SDK.
        """
        if self._session is None:
            raise ServoBridgeError("Servo session is not open")
        missing = missing_cameras(images)
        if missing:
            raise ServoBridgeError(f"Servo observation is missing camera bytes: {missing}")
        state_values = [float(value) for value in state]
        expected_state_dim = self._expected_state_dim()
        if len(state_values) != expected_state_dim:
            raise ServoBridgeError(
                f"Servo endpoint state must be {expected_state_dim} floats, "
                f"got {len(state_values)}"
            )
        contract = _checkpoint_input_contract(self._policy)
        if contract is not None:
            import numpy as np

            camera_names, state_name = _checkpoint_input_names(contract)
            geometry = _checkpoint_geometry(contract).get("camera_inputs", {})
            inputs = {camera_names[key]: _checkpoint_pixels(images[key], geometry.get(key, {}))
                      for key in CAMERA_KEYS}
            inputs[state_name] = np.asarray(state_values, dtype=np.float32)
            options = {} if noise_seed is None else {"noise_seed": int(noise_seed)}
            prediction = self._session.predict(
                inputs=inputs, instruction=instruction or self.instruction, **options
            )
            return self._prediction_result(prediction)
        observation = {
            "images": {key: _observation_frame(images[key]) for key in CAMERA_KEYS},
            "state": state_values,
            "instruction": instruction or self.instruction,
        }
        options = {} if noise_seed is None else {"noise_seed": int(noise_seed)}
        prediction = self._session.act(observation, instruction=instruction, **options)
        return self._prediction_result(prediction)

    def begin_episode(self) -> bool:
        """Invoke the optional episode-boundary hook on the SDK session."""
        session = self._session
        if session is None:
            raise ServoBridgeError("begin_episode requested before open")
        hook = getattr(session, "begin_episode", None)
        if not callable(hook):
            return False
        hook()
        return True

    def _expected_state_dim(self) -> int:
        """Return the endpoint-advertised state width, with the legacy default."""
        return int(self.identity.get("state_dim") or STATE_DIM)

    @staticmethod
    def _prediction_result(prediction: Any) -> Dict[str, Any]:
        """Serialize the SDK result for the local bridge without revalidating it."""
        return {
            "actions": _jsonable(prediction.actions),
            "horizon": int(prediction.horizon),
            "action_space": prediction.action_space,
            "telemetry": _jsonable(prediction.telemetry),
            "binding": _jsonable(prediction.binding),
            "safety_signal": _jsonable(prediction.safety_signal),
        }

    def close(self, success: bool = True) -> None:
        session, self._session = self._session, None
        if session is None:
            return
        try:
            session.close(success=success, suppress_completion_error=True)
        finally:
            client, self._client = self._client, None
            closer = getattr(getattr(client, "_http", None), "close", None)
            if closer is not None:
                try:
                    closer()
                except Exception:  # noqa: BLE001 - teardown is best effort
                    pass


class ServoDirectHost(ServoSessionHost):
    """One self-hosted ``servo serve`` endpoint over the native action session.

    Grant-only and fallback-free by construction: the grant names exactly one
    endpoint, there is no control plane to consult, no SDK credentials, and no
    fallback URL — any failure surfaces as an error instead of a reroute.
    ``act``/``_validated_prediction`` are inherited unchanged: the direct
    policy's prediction carries the same fields the hosted session returns.
    """

    def __init__(
        self,
        *,
        grant: str,
        instruction: Optional[str] = None,
        timeout_sec: Optional[float] = 600.0,
        observation_encoding: str = "h264",
        h264_crf: Optional[int] = None,
    ):
        # Deliberately NOT calling super().__init__: this host has no
        # credentials file and no managed deployment id to require.
        if observation_encoding not in OBSERVATION_ENCODINGS:
            raise ServoBridgeError(
                f"observation_encoding must be one of {list(OBSERVATION_ENCODINGS)}, "
                f"got {observation_encoding!r}"
            )
        if h264_crf is not None and observation_encoding != "h264":
            raise ServoBridgeError(
                "h264_crf only applies to observation_encoding='h264'"
            )
        self._grant_path = grant
        self.deployment_id = "self-hosted"
        self.instruction = instruction
        self._timeout_sec = timeout_sec
        self.observation_encoding = observation_encoding
        self.h264_crf = int(h264_crf) if h264_crf is not None else None
        self._client = None
        self._session = None
        self._policy = None
        self.identity: Dict[str, Any] = {}

    def open(self) -> Dict[str, Any]:
        """Attach to the granted endpoint; the grant is the entire identity."""
        if self._session is not None:
            return dict(self.identity)
        try:
            from servo.direct import attach
            import PIL.Image  # noqa: F401
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise ServoBridgeError(
                "the official servo SDK or its image dependency (Pillow/PIL) is not importable in this interpreter "
                f"({sys.executable}); install servo-client and pillow or point "
                f"{SERVO_PYTHON_ENV} at a Python >= 3.12 that has it"
            ) from exc
        grant_path = Path(self._grant_path).expanduser()
        try:
            mode = grant_path.stat().st_mode
        except OSError as exc:
            raise ServoBridgeError(f"Servo grant cannot be read: {grant_path}") from exc
        if mode & 0o077:
            raise ServoBridgeError(
                f"Servo grant {grant_path} is group/world readable; it holds a "
                "private key — chmod 600 it"
            )
        policy = attach(
            grant=grant_path.read_text().strip(),
            instruction=self.instruction,
            timeout=float(self._timeout_sec or 600.0),
            observation_encoding=self.observation_encoding,
            h264_crf=self.h264_crf,
        )
        # DirectPolicy is LAZY: constructing it touches no network at all, so
        # without the two round trips below a dead serve, a stale grant or a
        # revoked key opens "successfully" here and fails on the FIRST ACT --
        # which in a real run is AFTER the arms are live. The launcher's
        # contract is that a bad credential, deployment or lease fails with the
        # arms still cold, so this is the only place it can be honoured.
        served, expected, metadata = self._verify_endpoint(policy)
        transport = self._open_session(policy)
        lease = getattr(transport, "lease", None)
        # DirectPolicy.act(observation, instruction=...) matches the hosted
        # Session.act call shape, so the inherited act() drives it unchanged.
        self._session = policy
        identity_fields = dict(getattr(policy.grant, "identity", None) or {})
        self.identity = _jsonable({
            "deployment_id": identity_fields.get("deployment_id") or "self-hosted",
            "backend": "servo-direct-action-session",
            "grant": str(grant_path),
            # Both of these read ``None`` on the operator's banner until they
            # are populated here; the launcher prints them immediately before
            # the arms are enabled, so "session None (generation None)" is the
            # last thing read before motion.
            "session_id": getattr(lease, "session_id", None),
            "generation_id": identity_fields.get("deploy_generation"),
            "checkpoint_digest": identity_fields.get("checkpoint_digest"),
            "manifest_hash": served or expected,
            "base_url": getattr(policy.grant, "endpoint_url", None),
            "cameras": sorted(dict(getattr(policy, "camera_inputs", None) or {})),
            "camera_inputs": dict(getattr(policy, "camera_inputs", None) or {}),
            # ``camera_inputs`` is the model-side fit (224x224 here), not the
            # local hardware source.  Keep the signed native declarations so
            # the parent can configure V4L2 before enabling either arm.
            "camera_sources": _camera_sources(policy),
            "observation_encoding": self.observation_encoding,
            "h264_crf": self.h264_crf,
            "control_profile": dict(getattr(policy.grant, "control_profile", None) or {}),
            "state_dim": (metadata.get("observation") or {}).get("state_dim"),
            "action_dim": (metadata.get("action") or {}).get("dim"),
            "action_horizon": (metadata.get("action") or {}).get("horizon"),
            # Where each control number came from. A generic bracket and a
            # measured one are indistinguishable from the profile alone, and
            # inheriting another model family's numbers silently is exactly the
            # failure this field exists to make visible.
            "control_provenance": dict(
                getattr(policy.grant, "control_provenance", None) or {}
            ),
        })
        return dict(self.identity)

    def _verify_endpoint(
        self, policy: Any
    ) -> Tuple[Optional[str], Optional[str], Dict[str, Any]]:
        """Prove the endpoint answers, authenticates, and serves THIS manifest."""
        endpoint = getattr(policy.grant, "endpoint_url", "the endpoint")
        try:
            metadata = dict(policy.metadata() or {})
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            self._abandon(policy)
            raise ServoBridgeError(
                f"Servo endpoint {endpoint} refused the grant at open: {exc}. "
                "A `servo serve` restart invalidates every grant it issued -- "
                "re-copy the current one from the serve host."
            ) from exc
        served = metadata.get("manifest_hash")
        expected = dict(getattr(policy.grant, "identity", None) or {}).get("manifest_hash")
        if served and expected and served != expected:
            self._abandon(policy)
            raise ServoBridgeError(
                f"Servo endpoint {endpoint} serves manifest {served} but the grant "
                f"names {expected}; this grant belongs to a previous serve"
            )
        return served, expected, metadata

    def _open_session(self, policy: Any) -> Any:
        """Open the action session itself, with the arms cold.

        This is the leg a stale grant fails on (``session control call failed
        (401)``), and it is also where an instruction-specific capture is paid
        -- off the act path, where a multi-second stall would freeze an arm.
        """
        try:
            return policy.open_action_session_transport(instruction=self.instruction)
        except Exception as exc:  # noqa: BLE001 - reported, not swallowed
            self._abandon(policy)
            raise ServoBridgeError(
                "Servo action session refused to open on "
                f"{getattr(policy.grant, 'endpoint_url', 'the endpoint')}: {exc}"
            ) from exc

    @staticmethod
    def _abandon(policy: Any) -> None:
        """Release a policy that failed open-time validation, then report."""
        try:
            policy.close()
        except Exception:  # noqa: BLE001 - teardown must not mask the real fault
            pass

    def close(self, success: bool = True) -> None:
        del success  # a self-hosted endpoint keeps no central rollout record
        session, self._session = self._session, None
        if session is None:
            return
        try:
            session.close()
        except Exception:  # noqa: BLE001 - teardown is best effort
            pass


def _camera_sources(policy: Any) -> Dict[str, Any]:
    """Project signed native camera declarations from a direct policy route."""
    route = getattr(policy, "observation_route", None)
    if route is None:
        return {}
    sources: Dict[str, Any] = {}
    for camera_id in tuple(getattr(route, "camera_ids", ()) or ()):
        resolved = route.by_camera_id[camera_id]
        sources[str(camera_id)] = _jsonable(resolved.sensor)
    return sources


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
    host: Optional[ServoSessionHost] = host_ref.get("host")
    if op == "open":
        if host is not None:
            raise ServoBridgeError("a Servo session is already open on this bridge")
        observation_encoding = header.get("observation_encoding") or "h264"
        if header.get("grant"):
            host = ServoDirectHost(
                grant=header["grant"],
                instruction=header.get("instruction"),
                timeout_sec=header.get("timeout_sec"),
                observation_encoding=observation_encoding,
                h264_crf=header.get("h264_crf"),
            )
        else:
            host = ServoSessionHost(
                deployment_id=header["deployment_id"],
                instruction=header.get("instruction"),
                observation_encoding=observation_encoding,
            )
        identity = host.open()
        host_ref["host"] = host
        return {"ok": True, "identity": identity}
    if op == "act":
        if host is None:
            raise ServoBridgeError("act requested before open")
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
            raise ServoBridgeError("begin_episode requested before open")
        return {"ok": True, "honoured": host.begin_episode()}
    if op == "close":
        if host is not None:
            host.close(success=bool(header.get("success", True)))
            host_ref["host"] = None
        return {"ok": True}
    if op == "ping":
        try:
            import servo  # noqa: F401
            import PIL.Image  # noqa: F401

            servo_importable = True
        except ImportError:
            servo_importable = False
        return {
            "ok": True,
            "executable": sys.executable,
            "python_version": sys.version.split()[0],
            "servo_importable": servo_importable,
        }
    raise ServoBridgeError(f"unknown Servo bridge op {op!r}")


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
                        "error": f"Servo bridge could not encode its response: {exc}",
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
