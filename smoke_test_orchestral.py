"""Predict from saved YAM images using a managed deployment; never moves hardware."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "examples" / "yam"))
from orchestral_session_bridge import OrchestralSessionHost


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deployment_id")
    parser.add_argument("--image-dir", type=Path, required=True,
                        help="Rollout directory containing front_rgb, left_rgb, right_rgb/000000.png")
    parser.add_argument("--state", type=Path, required=True, help="JSON array of 7 or 14 joint values")
    parser.add_argument("--instruction", default="pick up the red lid")
    args = parser.parse_args()
    images = {key: np.asarray(Image.open(args.image_dir / folder / "000000.png").convert("RGB"))
              for key, folder in (("top", "front_rgb"), ("left", "left_rgb"), ("right", "right_rgb"))}
    host = OrchestralSessionHost(deployment_id=args.deployment_id, instruction=args.instruction)
    try:
        identity = host.open()
        prediction = host.act(images, json.loads(args.state.read_text()))
        actions = np.asarray(prediction["actions"])
        expected = (identity["action_horizon"], identity["action_dim"])
        if actions.shape != expected or not np.isfinite(actions).all():
            raise RuntimeError(f"Invalid actions: {actions.shape}, expected {expected}")
        print(json.dumps({"action_shape": list(actions.shape), "telemetry": prediction["telemetry"]}, indent=2))
    finally:
        host.close()


if __name__ == "__main__":
    main()
