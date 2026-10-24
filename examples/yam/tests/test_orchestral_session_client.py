"""Hardware-free coverage for the official Orchestral action-session client path.

Nothing here touches the network, a GPU, the robot, or a real ``orchestral``
install: the SDK is replaced by an injected fake module (in-process) and by a
fake ``orchestral`` package on ``PYTHONPATH`` (subprocess bridge). The point of the
suite is to fence the parts of the contract that a live run cannot re-check:

* the local parent/child framing survives pipe fragmentation and refuses
  malformed frames;
* managed sessions construct the SDK client from Orchestral's documented environment;
* exactly ONE action session is opened and reused for every action chunk, and
  camera frames reach ``session.predict`` as RGB arrays;
* SDK predictions and metadata pass through unchanged; the robot-facing
  policy still checks action shape and finiteness before execution.
"""

from __future__ import annotations

import base64
import hashlib
import importlib.machinery
import importlib.util
import io
import json
import logging
import os
import struct
import sys
import tempfile
import time
import unittest
import unittest.mock
from pathlib import Path
from types import ModuleType, SimpleNamespace

from rollout_manifest import SEED_STRATEGY, RolloutSeedPlan
from orchestral_session_bridge import (
    ACTION_HORIZON,
    CAMERA_KEYS,
    OBSERVATION_ENCODINGS,
    ORCHESTRAL_PYTHON_ENV,
    STATE_DIM,
    OrchestralBridgeError,
    OrchestralDirectHost,
    OrchestralSessionHost,
    encode_frame,
    frame_parts,
    missing_cameras,
    raw_frame_array,
    raw_frame_spec,
    read_frame,
    write_frame,
)

try:  # numpy is present in both runtimes; the bridge only needs it for pixels.
    import numpy as np

    _HAVE_NUMPY = True
except Exception:  # pragma: no cover - numpy is a hard dep of the robot runtime
    np = None  # type: ignore[assignment]
    _HAVE_NUMPY = False

try:  # ``molmoact_client`` pulls torch/transformers; the bridge itself does not.
    from molmoact_client import ARM_DIM, MolmoActOrchestral, OrchestralSessionTransport

    _CLIENT_IMPORT_ERROR = None
except Exception as exc:  # noqa: BLE001 - reported through skipUnless below
    ARM_DIM = 7
    MolmoActOrchestral = None  # type: ignore[assignment]
    OrchestralSessionTransport = None  # type: ignore[assignment]
    _CLIENT_IMPORT_ERROR = exc

_HAVE_CLIENT = _CLIENT_IMPORT_ERROR is None
_CLIENT_SKIP = f"molmoact_client is not importable here: {_CLIENT_IMPORT_ERROR!r}"

# Keep the policy logger quiet; ``setup_logger`` never resets ``disabled``.
logging.getLogger("molmoact").disabled = True
_QUIET_LOGGER = logging.getLogger("orchestral_session_client_tests")
_QUIET_LOGGER.addHandler(logging.NullHandler())
_QUIET_LOGGER.propagate = False
_QUIET_LOGGER.setLevel(logging.CRITICAL)

# A syntactically real (tiny) JPEG payload: SOI ... EOI. The fakes never decode
# it, so the bridge tests stay stdlib-only.
_JPEG = b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + bytes(
    range(256)
) + b"\xff\xd9"

_FAKE_BASE_URL = "https://orchestral.invalid"


def _camera_blobs() -> dict:
    """Three distinguishable JPEG payloads, one per Orchestral camera key."""
    return {key: _JPEG + key.encode("ascii") * 7 for key in CAMERA_KEYS}


def _iter_strings(value):
    """Yield every string (keys included) reachable inside ``value``."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(key, str):
                yield key
            yield from _iter_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_strings(item)


def _assert_no_base64_images(test, observation, blobs) -> None:
    """No camera frame may appear as base64/data-URI text anywhere."""
    encoded = [base64.b64encode(blob).decode("ascii") for blob in blobs]
    for text in _iter_strings(observation):
        test.assertFalse(
            text.startswith("data:"),
            msg=f"observation carries a data URI: {text[:40]!r}",
        )
        for candidate in encoded:
            test.assertNotIn(candidate[:48], text)
            test.assertNotIn(candidate, text)


# ---------------------------------------------------------------------------
# 1. Local parent/child framing
# ---------------------------------------------------------------------------


class _DribbleStream:
    """A stream that hands back exactly one byte per ``read`` (pipe worst case)."""

    def __init__(self, payload: bytes):
        self._payload = payload
        self._offset = 0
        self.reads = 0

    def read(self, size: int) -> bytes:
        self.reads += 1
        if size <= 0 or self._offset >= len(self._payload):
            return b""
        chunk = self._payload[self._offset : self._offset + 1]
        self._offset += 1
        return chunk


class FrameCodecTests(unittest.TestCase):
    def test_round_trip_without_buffers(self):
        header = {"op": "ping", "nested": {"list": [1, 2, 3]}, "unicode": "café"}
        decoded_header, buffers = read_frame(io.BytesIO(encode_frame(header)))
        self.assertEqual(decoded_header, header)
        self.assertEqual(buffers, [])

    def test_round_trip_with_three_binary_buffers(self):
        blobs = list(_camera_blobs().values())
        frame = encode_frame({"op": "act", "images": {"top": 0, "left": 1, "right": 2}}, blobs)
        decoded_header, decoded_buffers = read_frame(io.BytesIO(frame))
        self.assertEqual(decoded_header["op"], "act")
        self.assertEqual(decoded_buffers, blobs)
        for buffer in decoded_buffers:
            self.assertIsInstance(buffer, bytes)
            self.assertTrue(buffer.startswith(b"\xff\xd8"))
        # The frame is byte-transparent: the raw JPEGs are embedded verbatim.
        for blob in blobs:
            self.assertIn(blob, frame)
            self.assertNotIn(base64.b64encode(blob), frame)

    def test_short_reads_are_reassembled(self):
        blobs = list(_camera_blobs().values())
        stream = _DribbleStream(encode_frame({"op": "act", "n": 3}, blobs))
        decoded_header, decoded_buffers = read_frame(stream)
        self.assertEqual(decoded_header, {"op": "act", "n": 3})
        self.assertEqual(decoded_buffers, blobs)
        # One byte per read: the codec cannot have relied on a single read().
        self.assertGreater(stream.reads, len(blobs[0]))

    def test_encode_rejects_empty_and_non_bytes_buffers(self):
        with self.assertRaises(ValueError):
            encode_frame({"op": "act"}, [b""])
        with self.assertRaises(ValueError):
            encode_frame({"op": "act"}, ["not-bytes"])
        with self.assertRaises(ValueError):
            encode_frame({"op": "act"}, [_JPEG, b"", _JPEG])

    def test_encode_rejects_too_many_buffers(self):
        with self.assertRaises(ValueError):
            encode_frame({"op": "act"}, [b"x"] * 17)
        # 16 is the documented ceiling and must still encode.
        self.assertTrue(encode_frame({"op": "act"}, [b"x"] * 16))

    def test_encode_rejects_non_finite_header_values(self):
        with self.assertRaises(ValueError):
            encode_frame({"op": "act", "state": [float("nan")] * STATE_DIM})

    def test_read_rejects_bad_magic_and_oversized_buffer_count(self):
        with self.assertRaisesRegex(ValueError, "magic"):
            read_frame(io.BytesIO(struct.pack("<4sII", b"XXXX", 2, 0) + b"{}"))
        with self.assertRaisesRegex(ValueError, "too many buffers"):
            read_frame(io.BytesIO(struct.pack("<4sII", b"SVYB", 2, 17) + b"{}"))

    def test_read_rejects_a_non_object_header(self):
        with self.assertRaisesRegex(ValueError, "JSON object"):
            read_frame(io.BytesIO(struct.pack("<4sII", b"SVYB", 2, 0) + b"[]"))

    def test_read_reports_eof_and_truncation(self):
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(b""))
        truncated = encode_frame({"op": "act"}, [_JPEG])[:-1]
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(truncated))
        with self.assertRaises(EOFError):
            read_frame(io.BytesIO(encode_frame({"op": "act"})[:5]))

    def test_write_frame_streams_a_readable_frame(self):
        buffer = io.BytesIO()
        write_frame(buffer, {"ok": True, "identity": {"session_id": "sess_1"}}, [_JPEG])
        buffer.seek(0)
        header, buffers = read_frame(buffer)
        self.assertTrue(header["ok"])
        self.assertEqual(buffers, [_JPEG])


# ---------------------------------------------------------------------------
# 3. OrchestralSessionHost against a fake official SDK
# ---------------------------------------------------------------------------










# ---------------------------------------------------------------------------
# 4. The subprocess bridge, end to end
# ---------------------------------------------------------------------------

_FAKE_ORCHESTRAL_PACKAGE = 'from dataclasses import dataclass\nfrom types import SimpleNamespace\nimport numpy as np\n\n@dataclass\nclass Timings:\n    accel_exec_us: int = 3250\n\nclass Orchestral:\n    def __init__(self):\n        self.base_url = "https://orchestral.invalid"\n        self._http = SimpleNamespace(close=lambda: None)\n        self.deployments = SimpleNamespace(get=self.get)\n    def get(self, deployment_id):\n        return SimpleNamespace(model_id="mdl_test", io_spec=SPEC, session=Session)\n\nSPEC = SimpleNamespace(cameras={"middle": (8, 8), "left": (8, 8), "right": (8, 8)},\n                       state_dim=14, action_shape=(30, 14), control_hz=30, modalities={})\nclass Session:\n    def __init__(self, grant=None, *, instruction="", encoder="h264"):\n        self.session_id = "ses_test"\n        self.calls = []\n        self.closed = False\n        self.encoder = encoder\n    def predict(self, *, inputs, instruction=None, episode_start=False, noise_seed=None):\n        self.calls.append((inputs, instruction, episode_start, noise_seed))\n        actions = np.zeros((30, 14), dtype=np.float32)\n        actions[0, 0] = inputs["middle"][0, 0, 0]\n        return SimpleNamespace(actions=actions, timings=Timings(), trace_id="trace_test")\n    def close(self):\n        self.closed = True\n\nclass ConnectGrant:\n    @staticmethod\n    def from_token(token):\n        return SimpleNamespace(io_spec=SPEC, deployment_id="dep_direct")\n'

@unittest.skipUnless(_HAVE_CLIENT, _CLIENT_SKIP)
class OrchestralSessionTransportTests(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.root = self._dir.name
        package = Path(self.root) / "fakepkg" / "orchestral"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(_FAKE_ORCHESTRAL_PACKAGE, encoding="utf-8")
        self.package_root = str(package.parent)
        self.trace = str(Path(self.root) / "trace.jsonl")

    def _transport(self, **kwargs):
        kwargs.setdefault("deployment_id", "dep_test")
        kwargs.setdefault("observation_encoding", "jpeg")  # Legacy framing coverage.
        kwargs.setdefault("instruction", "pick up the red cap")
        kwargs.setdefault("orchestral_python", sys.executable)
        kwargs.setdefault("logger", _QUIET_LOGGER)
        env = {
            "PYTHONPATH": self.package_root,
            "FAKE_ORCHESTRAL_TRACE": self.trace,
        }
        env.update(kwargs.pop("extra_env", {}))
        kwargs.setdefault("bridge_env", env)
        transport = OrchestralSessionTransport(**kwargs)
        self.addCleanup(self._safe_close, transport)
        return transport

    @staticmethod
    def _safe_close(transport):
        try:
            transport.close(success=False)
        except Exception:  # noqa: BLE001 - cleanup only
            pass

    def _trace_events(self):
        path = Path(self.trace)
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    def test_direct_bridge_preserves_the_configured_open_timeout(self):
        from orchestral_session_bridge import _handle

        transport = self._transport(
            deployment_id=None, grant="/unused/grant.json", open_timeout_sec=45.0
        )
        host_ref = {}
        with unittest.mock.patch("orchestral_session_bridge.OrchestralDirectHost") as host_class:
            host_class.return_value.open.return_value = {"session_id": "direct_test"}
            with unittest.mock.patch.object(transport, "_spawn"), unittest.mock.patch.object(
                transport, "_request",
                side_effect=lambda header, **kwargs: _handle(host_ref, header, []),
            ) as request:
                transport.open()

            self.assertEqual(request.call_args.kwargs["timeout"], 45.0)
            self.assertEqual(host_class.call_args.kwargs["timeout_sec"], 35.0)





    def test_a_missing_interpreter_is_refused_up_front(self):
        with self.assertRaisesRegex(OrchestralBridgeError, "interpreter not found"):
            OrchestralSessionTransport(
                deployment_id="dep_test",
                orchestral_python=str(Path(self.root) / "no-such-python"),
                logger=_QUIET_LOGGER,
            )

    def test_a_non_positive_timeout_is_refused(self):
        for bad in (0, -1.0):
            for kwarg in ("open_timeout_sec", "act_timeout_sec"):
                with self.assertRaises(ValueError):
                    OrchestralSessionTransport(
                        deployment_id="dep_test",
                        orchestral_python=sys.executable,
                        logger=_QUIET_LOGGER,
                        **{kwarg: bad},
                    )

    def test_the_interpreter_choice_is_explicit(self):
        environment = {key: value for key, value in os.environ.items()}
        environment.pop(ORCHESTRAL_PYTHON_ENV, None)
        with unittest.mock.patch.dict(os.environ, environment, clear=True):
            if importlib.util.find_spec("orchestral") is not None:  # pragma: no cover
                transport = OrchestralSessionTransport(
                    deployment_id="dep_test",
                    logger=_QUIET_LOGGER,
                )
                self.assertEqual(transport.mode, "in-process")
            else:
                with self.assertRaisesRegex(OrchestralBridgeError, "not importable"):
                    OrchestralSessionTransport(
                        deployment_id="dep_test",
                        logger=_QUIET_LOGGER,
                    )

    def test_requests_before_spawn_are_refused(self):
        transport = self._transport()
        with self.assertRaisesRegex(OrchestralBridgeError, "not running"):
            transport._request({"op": "ping"}, timeout=1.0)


# ---------------------------------------------------------------------------
# 5. MolmoActOrchestral over a stub transport
# ---------------------------------------------------------------------------


class _StubTransport:
    """Stands in for :class:`OrchestralSessionTransport` (no SDK, no subprocess)."""

    def __init__(self, prediction=None, deployment_id="dep_test"):
        self.deployment_id = deployment_id
        self.identity = {
            "deployment_id": deployment_id,
            "session_id": "sess_stub_1",
            "generation_id": "gen_stub_1",
            "checkpoint_digest": "sha256:" + "e" * 64,
            "manifest_hash": "sha256:" + "f" * 64,
            "base_url": _FAKE_BASE_URL,
        }
        self.calls = []
        self.opens = 0
        self.closes = []
        self._prediction = prediction

    def open(self):
        self.opens += 1
        return dict(self.identity)

    def act(self, images, state, instruction=None, noise_seed=None):
        # Named explicitly rather than absorbed by **kwargs: a stand-in that
        # silently accepts a seed it never records is how a dropped seed goes
        # unnoticed, which is the one failure this whole path exists to catch.
        self.calls.append(
            {
                "images": dict(images),
                "state": list(state),
                "instruction": instruction,
                "noise_seed": noise_seed,
            }
        )
        if self._prediction is not None:
            return self._prediction
        return {
            "actions": [
                [float(row) + column / 100.0 for column in range(STATE_DIM)]
                for row in range(ACTION_HORIZON)
            ],
            "horizon": ACTION_HORIZON,
            "action_space": "joint_position",
            "telemetry": {"server_infer_ms": 42.0},
            "binding": {"deploy_generation": "gen_stub_1"},
            "safety_signal": None,
        }

    def begin_episode(self):
        return True

    def close(self, success=True):
        self.closes.append(success)


@unittest.skipUnless(_HAVE_CLIENT, _CLIENT_SKIP)
class MolmoActOrchestralTests(unittest.TestCase):
    def setUp(self):
        import numpy as np

        self.np = np

    def _policy(self, *, prediction=None, **kwargs):
        kwargs.setdefault("observation_encoding", "jpeg")  # Legacy encoder coverage.
        kwargs.setdefault("orchestral_python", sys.executable)
        kwargs.setdefault("instruction", "pick up the red cap")
        policy = MolmoActOrchestral("dep_test", **kwargs)
        policy._transport = _StubTransport(prediction=prediction)
        return policy

    def _frame(self, height, width, color):
        frame = self.np.zeros((height, width, 3), dtype=self.np.uint8)
        frame[:, :, :] = color
        return frame

    def _observation(self):
        return {
            "front_camera_rgb": self._frame(48, 64, (220, 20, 20)),
            "left_camera_rgb": self._frame(24, 32, (20, 220, 20)),
            "right_camera_rgb": self._frame(16, 16, (20, 20, 220)),
            "joint_positions": self.np.arange(STATE_DIM, dtype=self.np.float32),
        }

    def test_a_deployment_id_is_mandatory(self):
        for missing in (None, ""):
            with self.assertRaisesRegex(ValueError, "deployment id"):
                MolmoActOrchestral(missing)


    def test_unseeded_h264_rollout_does_not_warn_about_seed_replay(self):
        policy = self._policy(observation_encoding="h264")
        policy._transport.begin_episode = unittest.mock.Mock(return_value=False)
        with unittest.mock.patch.object(policy.logger, "warning") as warning:
            policy.begin_rollout(None)
        warning.assert_not_called()

    def test_encoder_settings_are_validated(self):
        with self.assertRaisesRegex(ValueError, "jpeg_quality"):
            MolmoActOrchestral("dep_test", jpeg_quality=0)
        with self.assertRaisesRegex(ValueError, "jpeg_quality"):
            MolmoActOrchestral("dep_test", jpeg_quality=101)
        with self.assertRaisesRegex(ValueError, "image_size"):
            MolmoActOrchestral(
                "dep_test",
                orchestral_python=sys.executable,
                image_size=0,
            )

    def test_prepare_input_and_inference_produce_a_30x14_float32_chunk(self):
        policy = self._policy()
        self.assertEqual(policy.get_action_horizon(), ACTION_HORIZON)
        prepared = policy.prepare_input(self._observation(), "pick up the red cap")
        self.assertEqual(prepared["instruction"], "pick up the red cap")
        self.assertEqual(prepared["state"].shape, (STATE_DIM,))

        result = policy.inference(prepared)
        actions = result["actions"]
        self.assertEqual(actions.shape, (ACTION_HORIZON, STATE_DIM))
        self.assertEqual(actions.dtype, self.np.float32)
        self.assertTrue(self.np.isfinite(actions).all())
        self.assertEqual(result["transport"]["protocol"], "orchestral-action-session")
        self.assertEqual(result["transport"]["session_id"], "sess_stub_1")
        self.assertEqual(result["transport"]["action_space"], "joint_position")
        self.assertEqual(result["transport"]["server_inference_ms"], 42.0)
        self.assertGreater(result["transport"]["image_bytes"], 0)
        self.assertEqual(result["reproducibility"]["backend"], "orchestral-action-session")
        self.assertEqual(result["orchestral"]["action_space"], "joint_position")
        self.assertEqual(result["orchestral"]["telemetry"], {"server_infer_ms": 42.0})

        call = policy._transport.calls[0]
        self.assertEqual(call["instruction"], "pick up the red cap")
        self.assertEqual(call["state"], [float(i) for i in range(STATE_DIM)])
        self.assertEqual(
            result["transport"]["image_bytes"],
            sum(len(blob) for blob in call["images"].values()),
        )

    def test_single_arm_endpoint_is_repacked_into_the_live_bimanual_state(self):
        horizon = 15
        endpoint_actions = [
            [float(row) + column / 100.0 for column in range(ARM_DIM)]
            for row in range(horizon)
        ]
        prediction = {
            "actions": endpoint_actions,
            "horizon": horizon,
            "action_space": "joint_position",
            "telemetry": {"server_infer_ms": 42.0},
            "binding": {"deploy_generation": "gen_stub_1"},
            "safety_signal": None,
        }

        for side, endpoint_state, active, held in (
            ("left", range(ARM_DIM), slice(0, ARM_DIM), slice(ARM_DIM, STATE_DIM)),
            ("right", range(ARM_DIM, STATE_DIM), slice(ARM_DIM, STATE_DIM), slice(0, ARM_DIM)),
        ):
            with self.subTest(side=side):
                policy = self._policy(prediction=prediction, single_arm_side=side)
                policy._transport.identity.update(
                    {
                        "state_dim": ARM_DIM,
                        "action_dim": ARM_DIM,
                        "action_horizon": horizon,
                        "control_profile": {"exec_steps": horizon, "fps": 15.0},
                    }
                )
                policy.open()
                prepared = policy.prepare_input(self._observation(), "pick up the red cap")
                self.assertEqual(prepared["state"].tolist(), [float(i) for i in endpoint_state])

                result = policy.inference(prepared)
                actions = result["actions"]
                self.assertEqual(actions.shape, (horizon, STATE_DIM))
                self.assertTrue(
                    self.np.array_equal(
                        actions[:, active], self.np.asarray(endpoint_actions, dtype=self.np.float32)
                    )
                )
                expected_hold = self.np.repeat(
                    prepared["bimanual_state"][None, held], horizon, axis=0
                )
                self.assertTrue(self.np.array_equal(actions[:, held], expected_hold))
                self.assertEqual(
                    policy._transport.calls[0]["state"],
                    [float(i) for i in endpoint_state],
                )

    def test_single_arm_endpoint_contract_mismatch_fails_closed(self):
        policy = self._policy(single_arm_side="left")
        policy._transport.identity.update({"state_dim": STATE_DIM, "action_dim": ARM_DIM})

        with self.assertRaisesRegex(OrchestralBridgeError, "state_dim=14"):
            policy.open()

        self.assertEqual(policy._transport.closes, [False])

    def test_active_arm_hint_does_not_slice_a_native_bimanual_endpoint(self):
        policy = self._policy(single_arm_side="left")
        policy._transport.identity.update(
            {
                "state_dim": STATE_DIM,
                "action_dim": STATE_DIM,
                "action_horizon": ACTION_HORIZON,
                "control_profile": {"exec_steps": ACTION_HORIZON, "fps": 30.0},
            }
        )

        policy.open()
        prepared = policy.prepare_input(self._observation(), "pick up the red cap")
        result = policy.inference(prepared)

        self.assertIsNone(policy._endpoint_single_arm_side)
        self.assertEqual(prepared["state"].shape, (STATE_DIM,))
        self.assertEqual(result["actions"].shape, (ACTION_HORIZON, STATE_DIM))
        self.assertEqual(policy._transport.calls[0]["state"], list(map(float, range(STATE_DIM))))

    def test_cameras_map_onto_the_orchestral_keys(self):
        from PIL import Image

        policy = self._policy()
        observation = self._observation()
        policy.inference(policy.prepare_input(observation, "pick up the red cap"))
        images = policy._transport.calls[0]["images"]

        self.assertEqual(set(images), set(CAMERA_KEYS))
        expected = {
            "top": ((64, 48), 0),  # front camera, red
            "left": ((32, 24), 1),  # left wrist, green
            "right": ((16, 16), 2),  # right wrist, blue
        }
        for key, (size, channel) in expected.items():
            blob = images[key]
            self.assertIsInstance(blob, bytes)
            self.assertTrue(blob.startswith(b"\xff\xd8"), f"{key} is not a JPEG")
            with Image.open(io.BytesIO(blob)) as image:
                self.assertEqual(image.format, "JPEG")
                self.assertEqual(image.size, size, f"{key} came from the wrong camera")
                pixels = self.np.asarray(image.convert("RGB"))
            self.assertEqual(
                int(pixels.mean(axis=(0, 1)).argmax()),
                channel,
                f"{key} came from the wrong camera",
            )
        _assert_no_base64_images(self, policy._transport.calls[0], images.values())

    def test_jpeg_quality_and_image_size_are_honoured(self):
        from PIL import Image

        rng = self.np.random.default_rng(1234)
        noisy = rng.integers(0, 256, size=(96, 128, 3), dtype=self.np.uint8)
        observation = {
            "front_camera_rgb": noisy,
            "left_camera_rgb": noisy,
            "right_camera_rgb": noisy,
            "joint_positions": self.np.zeros(STATE_DIM, dtype=self.np.float32),
        }

        low = self._policy(jpeg_quality=40)
        high = self._policy(jpeg_quality=95)
        low.inference(low.prepare_input(observation, "x"))
        high.inference(high.prepare_input(observation, "x"))
        low_bytes = len(low._transport.calls[0]["images"]["top"])
        high_bytes = len(high._transport.calls[0]["images"]["top"])
        self.assertLess(low_bytes, high_bytes)

        sized = self._policy(image_size=224)
        sized.inference(sized.prepare_input(observation, "x"))
        for blob in sized._transport.calls[0]["images"].values():
            with Image.open(io.BytesIO(blob)) as image:
                self.assertEqual(image.size, (224, 224))

        self.assertEqual(
            low.reproducibility_metadata()["image_encoding"], "jpeg-quality-40-size-source"
        )
        self.assertEqual(
            sized.reproducibility_metadata()["image_encoding"], "jpeg-quality-85-size-224"
        )

    def test_reproducibility_metadata_reports_the_session_identity(self):
        policy = self._policy()
        metadata = policy.reproducibility_metadata()
        self.assertEqual(metadata["backend"], "orchestral-action-session")
        self.assertEqual(metadata["deployment_id"], "dep_test")
        self.assertEqual(metadata["session_id"], "sess_stub_1")
        self.assertEqual(metadata["generation_id"], "gen_stub_1")
        self.assertEqual(metadata["checkpoint_digest"], "sha256:" + "e" * 64)
        self.assertEqual(metadata["manifest_hash"], "sha256:" + "f" * 64)
        self.assertEqual(metadata["base_url"], _FAKE_BASE_URL)
        # Unproven before any act: the metadata reports what the serve has
        # actually demonstrated, not what was asked for.
        self.assertFalse(metadata["deterministic_generator"])
        self.assertFalse(metadata["seed_requested"])
        # A seeded rollout now genuinely arms a deterministic stream -- this
        # used to assert False because the hosted path could not carry a seed.
        armed = policy.begin_rollout(7)
        self.assertTrue(armed["deterministic_generator"])
        self.assertEqual(armed["query_seed_strategy"], SEED_STRATEGY)
        self.assertTrue(policy.reproducibility_metadata()["seed_requested"])

    def test_open_and_close_delegate_to_the_single_session(self):
        policy = self._policy()
        self.assertEqual(policy.open()["session_id"], "sess_stub_1")
        self.assertEqual(policy.identity["session_id"], "sess_stub_1")
        policy.close(success=False)
        policy.close()
        self.assertEqual(policy._transport.opens, 1)
        self.assertEqual(policy._transport.closes, [False, True])

    def test_a_malformed_chunk_from_the_transport_is_refused(self):
        prepared_observation = self._observation()
        broken = [
            {
                "actions": [[0.0] * STATE_DIM for _ in range(29)],
                "action_space": "joint_position",
            },
            {
                "actions": [[0.0] * 7 for _ in range(ACTION_HORIZON)],
                "action_space": "joint_position",
            },
            {
                "actions": [
                    [float("nan")] * STATE_DIM for _ in range(ACTION_HORIZON)
                ],
                "action_space": "joint_position",
            },
        ]
        for prediction in broken:
            policy = self._policy(prediction=prediction)
            prepared = policy.prepare_input(prepared_observation, "x")
            with self.assertRaisesRegex(RuntimeError, "malformed action chunk"):
                policy.inference(prepared)

    def test_a_non_bimanual_state_is_refused_before_the_wire(self):
        policy = self._policy()
        observation = self._observation()
        observation["joint_positions"] = self.np.zeros(7, dtype=self.np.float32)
        with self.assertRaisesRegex(ValueError, "bimanual state"):
            policy.prepare_input(observation, "x")
        self.assertEqual(policy._transport.calls, [])

        prepared = policy.prepare_input(self._observation(), "x")
        prepared["state"] = self.np.full(STATE_DIM, self.np.nan, dtype=self.np.float32)
        with self.assertRaisesRegex(ValueError, "non-finite"):
            policy.inference(prepared)
        self.assertEqual(policy._transport.calls, [])

    def test_a_non_rgb_camera_frame_is_refused(self):
        policy = self._policy()
        observation = self._observation()
        observation["front_camera_rgb"] = self.np.zeros((8, 8), dtype=self.np.uint8)
        with self.assertRaisesRegex(ValueError, "HxWx3"):
            policy.inference(policy.prepare_input(observation, "x"))
        self.assertEqual(policy._transport.calls, [])




# ---------------------------------------------------------------------------
# 6. The codec wire: raw pixels across the bridge, and a cold-failing open
# ---------------------------------------------------------------------------




def _write_grant(directory: str) -> str:
    path = Path(directory) / "grant.txt"
    path.write_text("fake-grant-blob")
    path.chmod(0o600)
    return str(path)


@unittest.skipUnless(_HAVE_NUMPY, "numpy is required for the raw-pixel wire")
class RawFrameFramingTests(unittest.TestCase):
    """The pipe must carry pixels intact -- a wrong shape is undetectable later."""

    def test_a_raw_frame_survives_the_pipe_byte_for_byte(self):
        image = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        buffer = memoryview(np.ascontiguousarray(image)).cast("B")
        header = {"op": "act", "frames": {"top": raw_frame_spec(0, 4, 5)}}
        stream = io.BytesIO(b"".join(bytes(p) for p in frame_parts(header, [buffer])))
        decoded_header, buffers = read_frame(stream)
        restored = raw_frame_array(buffers[0], decoded_header["frames"]["top"])
        self.assertEqual(restored.shape, (4, 5, 3))
        self.assertEqual(restored.dtype, np.uint8)
        self.assertTrue(np.array_equal(restored, image))

    def test_a_memoryview_is_framed_by_its_byte_count_not_its_row_count(self):
        # ``len(memoryview(hxwx3))`` is the ROW count. Framing that would
        # declare a 4-byte buffer for a 60-byte frame and truncate silently.
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        buffer = memoryview(image).cast("B")
        self.assertEqual(len(buffer), image.nbytes)
        parts = frame_parts({"op": "act"}, [buffer])
        self.assertEqual(sum(len(bytes(p)) for p in parts[3:]), image.nbytes)

    def test_a_truncated_raw_frame_is_refused_not_reshaped(self):
        with self.assertRaisesRegex(OrchestralBridgeError, "expected 60"):
            raw_frame_array(b"\x00" * 59, raw_frame_spec(0, 4, 5))

    def test_a_foreign_dtype_is_refused(self):
        spec = {**raw_frame_spec(0, 4, 5), "dtype": "float32"}
        with self.assertRaisesRegex(OrchestralBridgeError, "must be uint8"):
            raw_frame_array(b"\x00" * 60, spec)

    def test_missing_cameras_reports_arrays_without_raising(self):
        full = {key: np.zeros((2, 2, 3), dtype=np.uint8) for key in CAMERA_KEYS}
        self.assertEqual(missing_cameras(full), [])
        empty = {**full, "left": np.zeros((0, 2, 3), dtype=np.uint8)}
        self.assertEqual(missing_cameras(empty), ["left"])
        self.assertEqual(missing_cameras({}), list(CAMERA_KEYS))




@unittest.skipUnless(_HAVE_CLIENT, _CLIENT_SKIP)
@unittest.skipUnless(_HAVE_NUMPY, "numpy is required for the raw-pixel wire")
class MolmoActOrchestralWireTests(unittest.TestCase):
    """Which payload the policy builds is decided by the negotiated wire."""

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.grant = _write_grant(self._dir.name)

    def _policy(self, **kwargs):
        return MolmoActOrchestral(
            grant=self.grant,
            orchestral_python=sys.executable,
            instruction="pick up the red lid",
            **kwargs,
        )

    def _observation(self):
        return {
            "left_camera_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "front_camera_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "right_camera_rgb": np.zeros((8, 8, 3), dtype=np.uint8),
            "joint_positions": np.zeros(STATE_DIM, dtype=np.float32),
        }

    def test_the_jpeg_wire_still_mints_jpeg_bytes(self):
        policy = self._policy(observation_encoding="jpeg")
        prepared = policy.prepare_input(self._observation(), "pick up the red lid")
        captured = {}
        with unittest.mock.patch.object(
            policy._transport, "act",
            side_effect=lambda images, state, instruction=None, noise_seed=None: (
                captured.update(images=images) or _fake_prediction_dict()
            ),
        ):
            policy.inference(prepared)
        for key in CAMERA_KEYS:
            self.assertIsInstance(captured["images"][key], bytes)
            self.assertTrue(captured["images"][key].startswith(b"\xff\xd8"))

    def test_the_codec_wire_sends_raw_contiguous_uint8_pixels(self):
        policy = self._policy(observation_encoding="h264")
        prepared = policy.prepare_input(self._observation(), "pick up the red lid")
        captured = {}
        with unittest.mock.patch.object(
            policy._transport, "act",
            side_effect=lambda images, state, instruction=None, noise_seed=None: (
                captured.update(images=images) or _fake_prediction_dict()
            ),
        ):
            policy.inference(prepared)
        for key in CAMERA_KEYS:
            frame = captured["images"][key]
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.dtype, np.uint8)
            self.assertEqual(frame.shape, (8, 8, 3))
            self.assertTrue(frame.flags["C_CONTIGUOUS"])

    def test_managed_sessions_default_to_h264_and_never_encode_jpeg(self):
        policy = MolmoActOrchestral(deployment="dep_hosted", orchestral_python=sys.executable)
        self.assertEqual(policy.observation_encoding, "h264")
        policy._transport = _StubTransport()
        with unittest.mock.patch.object(policy, "_jpeg_bytes", side_effect=AssertionError("JPEG")):
            policy.inference(policy.prepare_input(self._observation(), "pick up the red lid"))
        for frame in policy._transport.calls[0]["images"].values():
            self.assertIsInstance(frame, np.ndarray)
            self.assertEqual(frame.dtype, np.uint8)

    def test_every_declared_encoding_is_constructible(self):
        for encoding in OBSERVATION_ENCODINGS:
            self._policy(observation_encoding=encoding)


def _fake_prediction_dict():
    """The bridge-shaped prediction mapping ``MolmoActOrchestral.inference`` consumes."""
    return {
        "actions": [[0.0] * STATE_DIM for _ in range(ACTION_HORIZON)],
        "action_space": "joint_position",
        "telemetry": {"server_infer_ms": 12.0, "request_bytes": 4096},
    }


@unittest.skipUnless(_HAVE_CLIENT, _CLIENT_SKIP)
@unittest.skipUnless(_HAVE_NUMPY, "numpy is required for the raw-pixel wire")
class RemoteNoiseSeedTests(unittest.TestCase):
    """The remote policy must draw the SAME noise a local one would.

    A thor-local run and a thor->odin run are only comparable as the *same*
    rollout if both sample from the same stream. MolmoAct2's flow-matching head
    redraws noise per query, and a single draw moves commanded excursion by
    0.03-0.68 rad on identical input -- so an unseeded A/B measures the sampler,
    not the deployment. These tests pin the seed's whole path, including every
    way it can be silently lost.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.grant = _write_grant(self._dir.name)

    def _policy(self, **kwargs):
        policy = MolmoActOrchestral(
            grant=self.grant,
            orchestral_python=sys.executable,
            instruction="pick up the red lid",
            **kwargs,
        )
        policy._transport = _StubTransport(prediction=_seeded_prediction_dict())
        return policy

    def _observation(self):
        import numpy as np

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        return {
            "front_camera_rgb": frame,
            "left_camera_rgb": frame,
            "right_camera_rgb": frame,
            "joint_positions": [0.0] * STATE_DIM,
        }

    def test_query_seeds_match_the_local_policy_exactly(self):
        """The whole point: same base seed => same per-query seeds, both modes.

        Compared against RolloutSeedPlan directly rather than against a live
        MolmoActLocal, which would need a GPU and a checkpoint; the plan IS the
        shared derivation, so agreeing with it is agreeing with local.
        """
        base_seed = 1234
        policy = self._policy()
        rollout_seed = RolloutSeedPlan(base_seed).rollout_seed(3)
        policy.begin_rollout(rollout_seed)

        for _ in range(4):
            policy.inference(policy.prepare_input(self._observation(), "pick up the red lid"))

        sent = [call["noise_seed"] for call in policy._transport.calls]
        expected = [
            RolloutSeedPlan(None).query_seed(rollout_seed, index) for index in range(4)
        ]
        self.assertEqual(sent, expected)
        self.assertTrue(all(seed is not None for seed in sent))
        # Distinct per query -- a constant seed would make every chunk of a
        # rollout replay one noise draw, which is not what local does.
        self.assertEqual(len(set(sent)), 4)

    def test_unseeded_rollout_sends_no_seed(self):
        policy = self._policy()
        policy.begin_rollout(None)
        policy.inference(policy.prepare_input(self._observation(), "pick up the red lid"))
        self.assertIsNone(policy._transport.calls[0]["noise_seed"])
        self.assertFalse(policy.reproducibility_metadata()["seed_requested"])

    def test_begin_rollout_restarts_the_query_stream(self):
        """Two rollouts on the same seed must replay, not continue."""
        policy = self._policy()
        rollout_seed = RolloutSeedPlan(99).rollout_seed(0)

        policy.begin_rollout(rollout_seed)
        policy.inference(policy.prepare_input(self._observation(), "pick up the red lid"))
        first = policy._transport.calls[-1]["noise_seed"]

        policy.begin_rollout(rollout_seed)
        policy.inference(policy.prepare_input(self._observation(), "pick up the red lid"))
        self.assertEqual(policy._transport.calls[-1]["noise_seed"], first)

    def test_a_failed_act_does_not_consume_its_seed(self):
        """A retried act replays its own seed, matching MolmoActLocal."""
        policy = self._policy()
        policy.begin_rollout(RolloutSeedPlan(7).rollout_seed(0))
        observation = policy.prepare_input(self._observation(), "pick up the red lid")

        boom = RuntimeError("transient serve failure")
        original = policy._transport.act
        calls = {"n": 0}

        def failing(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise boom
            return original(*args, **kwargs)

        policy._transport.act = failing
        with self.assertRaises(RuntimeError):
            policy.inference(observation)
        policy.inference(observation)

        replayed = policy._transport.calls[-1]["noise_seed"]
        self.assertEqual(
            replayed, RolloutSeedPlan(None).query_seed(policy._rollout_seed, 0)
        )

    def test_reproducibility_reports_the_seed_as_proven_not_intended(self):
        policy = self._policy()
        policy.begin_rollout(RolloutSeedPlan(5).rollout_seed(0))
        self.assertFalse(policy.reproducibility_metadata()["deterministic_generator"])
        policy.inference(policy.prepare_input(self._observation(), "pick up the red lid"))
        metadata = policy.reproducibility_metadata()
        self.assertTrue(metadata["deterministic_generator"])
        self.assertTrue(metadata["seed_requested"])

    def test_a_serve_that_reports_no_scheme_is_not_counted_as_proven(self):
        """Belt to the SDK's braces.

        DirectPolicy already refuses a seeded act whose response carries no
        noise scheme. If that refusal is ever relaxed, the harness must still
        not claim a reproducibility it cannot demonstrate.
        """
        policy = self._policy()
        policy._transport = _StubTransport(prediction=_fake_prediction_dict())
        policy.begin_rollout(RolloutSeedPlan(5).rollout_seed(0))
        policy.inference(policy.prepare_input(self._observation(), "pick up the red lid"))
        self.assertFalse(policy.reproducibility_metadata()["deterministic_generator"])




def _seeded_prediction_dict():
    """A prediction from a serve that reports the scheme it sampled under."""
    prediction = _fake_prediction_dict()
    prediction["telemetry"] = dict(prediction["telemetry"])
    prediction["telemetry"]["noise_scheme"] = "numpy-pcg64-15x32-fp32-v1"
    return prediction











class OrchestralSDKTests(unittest.TestCase):
    def setUp(self):
        self.module = ModuleType("orchestral")
        patcher = unittest.mock.patch.dict(sys.modules, orchestral=self.module)
        patcher.start()
        self.addCleanup(patcher.stop)
        exec(_FAKE_ORCHESTRAL_PACKAGE, self.module.__dict__)
        self.host = OrchestralSessionHost(deployment_id="dep_test", instruction="pick")
        self.addCleanup(self.host.close)
        self.images = {key: np.full((8, 8, 3), 42, dtype=np.uint8) for key in CAMERA_KEYS}

    def test_session_reuse_mapping_timings_and_episode_boundary(self):
        identity = self.host.open()
        session = self.host._session
        self.assertEqual(identity["action_horizon"], 30)
        self.assertEqual(session.encoder, "h264")
        self.assertEqual(self.host.open(), identity)
        for _ in range(2):
            result = self.host.act(self.images, [0] * 14)
        self.assertEqual(result["actions"][0][0], 42)
        self.assertEqual(result["telemetry"]["server_infer_ms"], 3.25)
        self.assertEqual(result["telemetry"]["trace_id"], "trace_test")
        self.assertEqual([c[2] for c in session.calls], [True, False])
        self.assertEqual(set(session.calls[0][0]), {"middle", "left", "right", "state"})
        self.host.begin_episode()
        self.host.act(self.images, [0] * 14)
        self.assertTrue(session.calls[-1][2])
        self.host.close(success=False)
        self.host.close()
        self.assertTrue(session.closed)

    def test_invalid_contract_is_refused_before_session_creation(self):
        self.module.SPEC.cameras["wrist"] = (8, 8)
        with self.assertRaisesRegex(OrchestralBridgeError, "unmatched image"):
            self.host.open()
        self.assertIsNone(self.host._session)

    def test_single_arm_contract(self):
        self.module.SPEC.state_dim = 7
        self.module.SPEC.action_shape = (15, 7)
        identity = self.host.open()
        self.assertEqual((identity["state_dim"], identity["action_dim"], identity["action_horizon"]), (7, 7, 15))

    def test_missing_inputs_and_invalid_state_are_refused(self):
        self.host.open()
        for images, state, message in [({}, [0]*14, "missing camera"),
                                       (self.images, [0]*7, "14 floats")]:
            with self.assertRaisesRegex(OrchestralBridgeError, message):
                self.host.act(images, state)
        self.assertEqual(self.host._session.calls, [])

    def test_per_query_noise_seed_reaches_the_sdk_session(self):
        self.host.open()
        result = self.host.act(self.images, [0]*14, noise_seed=12)
        self.assertEqual(self.host._session.calls[-1][3], 12)
        self.assertEqual(result["telemetry"]["noise_scheme"], "numpy-pcg64-15x32-fp32-v1")
        self.assertTrue(result["telemetry"]["noise_seed_acknowledged"])

    def test_jpeg_is_decoded_before_predict(self):
        from PIL import Image
        buf = io.BytesIO()
        Image.fromarray(self.images["top"]).save(buf, format="JPEG")
        self.host.open()
        self.host.act({key: buf.getvalue() for key in CAMERA_KEYS}, [0]*14)
        pixels = self.host._session.calls[0][0]["middle"]
        self.assertEqual(pixels.shape, (8, 8, 3))
        self.assertEqual(pixels.dtype, np.uint8)

    def test_direct_grant_and_permissions(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"grant"
            path.write_text("token")
            path.chmod(0o600)
            host = OrchestralDirectHost(grant=str(path))
            self.addCleanup(host.close)
            self.assertEqual(host.open()["deployment_id"], "dep_direct")
            host.act(self.images, [0]*14)
            host.close()
            path.chmod(0o644)
            with self.assertRaisesRegex(OrchestralBridgeError, "chmod 600"):
                host.open()

    @unittest.skipUnless(_HAVE_CLIENT, _CLIENT_SKIP)
    def test_pixels_and_errors_cross_subprocess_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)/"orchestral"
            package.mkdir()
            (package/"__init__.py").write_text(_FAKE_ORCHESTRAL_PACKAGE)
            transport = OrchestralSessionTransport(deployment_id="dep_test", orchestral_python=sys.executable,
                bridge_env={"PYTHONPATH": directory}, logger=_QUIET_LOGGER)
            try:
                transport.open()
                for _ in range(2):
                    result = transport.act(self.images, [0]*14)
                    self.assertEqual(result["actions"][0][0], 42)
                with self.assertRaisesRegex(OrchestralBridgeError, "14 floats"):
                    transport.act(self.images, [0]*7)
            finally:
                transport.close()

    @unittest.skipUnless(_HAVE_CLIENT, _CLIENT_SKIP)
    def test_red_cap_adapter_crosses_subprocess_bridge(self):
        package_source = _FAKE_ORCHESTRAL_PACKAGE.replace(
            'cameras={"middle": (8, 8), "left": (8, 8), "right": (8, 8)}',
            'cameras={"base_0_rgb": (8, 8), "left_wrist_0_rgb": (8, 8), "right_wrist_0_rgb": (8, 8)}',
        ).replace('state_dim=14, action_shape=(30, 14)', 'state_dim=7, action_shape=(15, 7)')
        package_source = package_source.replace('np.zeros((30, 14)', 'np.zeros((15, 7)').replace('inputs["middle"]', 'inputs["base_0_rgb"]')
        package_source = package_source.replace('        return SimpleNamespace(actions=actions,',
            '        actions[:, 6] = inputs["state"][6]\n        return SimpleNamespace(actions=actions,')
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "orchestral"
            package.mkdir()
            (package / "__init__.py").write_text(package_source)
            transport = OrchestralSessionTransport(deployment_id="dep_test", orchestral_python=sys.executable,
                bridge_env={"PYTHONPATH": directory}, logger=_QUIET_LOGGER, policy_adapter="pi05_yam_red_cap")
            try:
                self.assertEqual(transport.open()["policy_adapter"], "pi05_yam_red_cap")
                result = transport.act(self.images, [1, 2, 3, 4, 5, 6, 0.2])
                np.testing.assert_allclose(result["actions"][0], [43, 2, 3, 4, 5, 6, 0.2], atol=1e-7)
            finally:
                transport.close()

class RedCapAdapterTests(unittest.TestCase):
    def make_host(self, adapter="pi05_yam_red_cap"):
        from dataclasses import make_dataclass
        host = OrchestralSessionHost(deployment_id="dep_test", policy_adapter=adapter)
        host._configure(SimpleNamespace(
            cameras={"base_0_rgb": (8, 8), "left_wrist_0_rgb": (8, 8), "right_wrist_0_rgb": (8, 8)},
            state_dim=7, action_shape=(15, 7), modalities={},
        ))
        self.calls = []
        self.actions = np.full((15, 7), 0.25, dtype=np.float32)
        self.actions[0, 6] = -0.5
        self.actions[1, 6] = 1.5
        timings = make_dataclass("Timings", [("accel_exec_us", int)])(1000)

        def predict(**kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(actions=self.actions, timings=timings, trace_id="trace")
        host._session = SimpleNamespace(predict=predict)
        return host

    def test_query_state_and_actions_use_checkpoint_conversion(self):
        host = self.make_host()
        images = {key: np.full((8, 8, 3), i * 20, np.uint8) for i, key in enumerate(CAMERA_KEYS)}
        for shift, grip in ((0, 0.2), (10, 1.2), (-10, -0.2)):
            state = np.array([shift + i for i in range(6)] + [grip], dtype=np.float32)
            original = state.copy()
            result = host.act(images, state)
            sent = self.calls[-1]["inputs"]
            np.testing.assert_array_equal(state, original)
            np.testing.assert_array_equal(sent["state"][:6], original[:6])
            self.assertAlmostEqual(sent["state"][6], 1 - np.clip(grip, 0, 1))
            np.testing.assert_allclose(np.asarray(result["actions"])[:, :6],
                                       np.broadcast_to(original[:6] + 0.25, (15, 6)))
            np.testing.assert_allclose(np.asarray(result["actions"])[:3, 6], [1, 0, 0.75])
            for name, i in (("base_0_rgb", 0), ("left_wrist_0_rgb", 1), ("right_wrist_0_rgb", 2)):
                self.assertEqual(int(sent[name][0, 0, 0]), i * 20)
        self.assertEqual(self.actions[0, 0], 0.25)
        self.assertEqual(self.actions[0, 6], -0.5)

    def test_adapter_rejects_a_complete_policy_contract(self):
        host = OrchestralSessionHost(deployment_id="dep", policy_adapter="pi05_yam_red_cap")
        with self.assertRaisesRegex(OrchestralBridgeError, "policy_config=none"):
            host._configure(SimpleNamespace(cameras={"middle": (8, 8), "left": (8, 8), "right": (8, 8)},
                state_dim=7, action_shape=(15, 7), modalities={}))

    def test_adapter_rejects_other_checkpoint_dimensions(self):
        host = OrchestralSessionHost(deployment_id="dep", policy_adapter="pi05_yam_red_cap")
        with self.assertRaisesRegex(OrchestralBridgeError, "15x7"):
            host._configure(SimpleNamespace(cameras={"base_0_rgb": (8, 8), "left_wrist_0_rgb": (8, 8), "right_wrist_0_rgb": (8, 8)},
                state_dim=14, action_shape=(15, 14), modalities={}))

    def test_invalid_adapter_is_rejected(self):
        with self.assertRaisesRegex(OrchestralBridgeError, "unsupported policy_adapter"):
            OrchestralSessionHost(deployment_id="dep", policy_adapter="guessed")


class PublishedSDKTests(unittest.TestCase):
    def test_real_sdk_encodes_predicts_and_closes_without_network(self):
        try:
            from orchestral import ConnectGrant, IOSpec
            from orchestral.core.wire import pack_step_response
        except ImportError:
            self.skipTest("orchestral-client is not installed")
        spec = IOSpec(cameras={"middle": (32, 32), "left": (32, 32), "right": (32, 32)},
                      state_dim=7, action_shape=(15, 7), control_hz=15)
        grant = ConnectGrant(deployment_id="dep_test", org_id="org_test", model_id="mdl_test",
            endpoint_host="127.0.0.1", http_port=8000, quic_port=8001,
            io_spec=spec, expires_at=time.time()+3600)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"grant"
            path.write_text(grant.to_token())
            path.chmod(0o600)
            for encoder in OBSERVATION_ENCODINGS:
                with self.subTest(encoder=encoder):
                    transport = unittest.mock.Mock()
                    transport.supports_stateful_h264 = True
                    transport.supports_connection_grant_cache = True
                    transport.request.return_value = pack_step_response(
                        np.ones((15, 7), dtype=np.float32), decode_us=1, queue_us=2, accel_exec_us=3250)
                    with unittest.mock.patch("orchestral.client.sdk.create_transport", return_value=transport), \
                         unittest.mock.patch.dict(os.environ, ORC_EXPORT_OTLP="0"):
                        host = OrchestralDirectHost(grant=str(path), observation_encoding=encoder)
                        try:
                            self.assertEqual(host.open()["state_dim"], 7)
                            for _ in range(2):
                                result = host.act({k: np.zeros((32, 32, 3), dtype=np.uint8) for k in CAMERA_KEYS}, [0]*7)
                            self.assertEqual(np.asarray(result["actions"]).shape, (15, 7))
                            self.assertEqual(result["telemetry"]["server_infer_ms"], 3.25)
                            self.assertEqual(transport.request.call_count, 2)
                        finally:
                            host.close()
                        transport.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
