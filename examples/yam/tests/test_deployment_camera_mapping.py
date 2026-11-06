"""Deployment metadata maps named views without model-specific robot settings.

These tests use the installed SDK's real IOSpec and the pure bridge configuration
path. They do not open a session or substitute a wire, codec, or auth component.
"""

import unittest
from pathlib import Path

import numpy as np
import yaml
from orchestral import IOSpec
from orchestral_session_bridge import (
    OrchestralBridgeError,
    OrchestralSessionHost,
    _checkpoint_pixels,
)


class DeploymentCameraMappingTests(unittest.TestCase):
    def test_deployment_names_map_the_same_physical_views(self):
        layouts = [
            ("top", "left", "right"),
            ("observation.images.top", "observation.images.left", "observation.images.right"),
            ("middle", "left", "right"),
            ("base_view", "left_wrist_view", "right_wrist_view"),
            ("video.base_view", "video.left_wrist_view", "video.right_wrist_view"),
            ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"),
            ("observation/images/top_cam", "observation/images/left_cam", "observation/images/right_cam"),
            ("top_camera-images-rgb_320_240", "left_camera-images-rgb_320_240", "right_camera-images-rgb_320_240"),
        ]
        views = {"top": 17, "left": 83, "right": 191}
        for names in layouts:
            with self.subTest(names=names):
                # Ordering cannot determine the physical mapping.
                spec = IOSpec(cameras={name: (8, 12) for name in reversed(names)},
                              state_dim=14, action_shape=(16, 14))
                host = OrchestralSessionHost(deployment_id="metadata-only")
                host._configure(spec)
                self.assertEqual(host.identity["camera_mapping"], dict(zip(views, names)))
                self.assertEqual(host.identity["action_horizon"], 16)
                for role, value in views.items():
                    pixels = np.full((24, 32, 3), value, dtype=np.uint8)
                    fitted = _checkpoint_pixels(pixels, host.identity["camera_inputs"][role])
                    self.assertEqual(fitted.shape, (8, 12, 3))
                    # Padding may add zeros, but the physical view stays intact.
                    self.assertEqual(int(fitted.max()), value)

    def test_shared_hardware_configs_do_not_select_red_cap_adapter(self):
        root = Path(__file__).resolve().parents[1] / "configs"
        for name in ("yam_deployment_physical.yaml", "yam_deployment_right.yaml"):
            config = yaml.safe_load((root / name).read_text())
            self.assertEqual(config["hz"], 30)
            self.assertEqual(config["eval"]["mode"], "server")
            self.assertIsNone(config["eval"]["reproducibility"]["seed"])
            self.assertEqual(config["eval"]["camera_preflight"]["max_clipped_white_fraction"], 1.0)
            for mode in ("server", "direct"):
                self.assertEqual((config["eval"].get(mode) or {}).get("policy_adapter", "identity"), "identity")

    def test_current_sdk_does_not_require_control_rate_metadata(self):
        host = OrchestralSessionHost(deployment_id="metadata-only")
        host._configure(IOSpec(cameras={name: (8, 8) for name in ("top", "left", "right")},
                               state_dim=7, action_shape=(15, 7)))
        self.assertEqual(host.identity["state_dim"], 7)
        self.assertEqual(host.identity["action_horizon"], 15)

    def test_ambiguous_unknown_and_extra_views_are_rejected(self):
        layouts = [
            ("top", "base_view", "left", "right"),
            ("camera_0", "camera_1", "camera_2"),
            ("top", "exterior_image_0_left", "right"),
            ("top", "top_left_wrist", "right"),
            ("top", "left", "right", "depth_view"),
        ]
        for names in layouts:
            with self.subTest(names=names):
                host = OrchestralSessionHost(deployment_id="metadata-only")
                with self.assertRaises(OrchestralBridgeError):
                    host._configure(IOSpec(cameras={name: (8, 8) for name in names}))
                self.assertIsNone(host._session)

    def test_temporal_views_are_rejected_before_session_creation(self):
        spec = IOSpec(cameras={name: (8, 8) for name in ("top", "left", "right")})
        # Older installed SDKs omit this optional field; the bridge accepts
        # their single-frame contracts and guards history when advertised.
        spec.camera_history = {"top": (-1, 0)}
        host = OrchestralSessionHost(deployment_id="metadata-only")
        with self.assertRaisesRegex(OrchestralBridgeError, "requires camera history"):
            host._configure(spec)
        self.assertIsNone(host._session)


if __name__ == "__main__":
    unittest.main()
