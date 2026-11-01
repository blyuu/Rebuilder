"""Build a versioned scene manifest from an extracted TUM RGB-D sequence."""

import argparse
import json
from pathlib import Path

from .tum import make_tum_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rgb-depth-tolerance-s", type=float, default=0.02)
    parser.add_argument("--pose-tolerance-s", type=float, default=0.02)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    args = parser.parse_args()
    manifest = make_tum_manifest(args.sequence, args.output,
        rgb_depth_tolerance_s=args.rgb_depth_tolerance_s,
        pose_tolerance_s=args.pose_tolerance_s,
        stride=args.stride, max_frames=args.max_frames)
    print(json.dumps({"output": str(args.output), "source": manifest["source"],
                      "selected_frames": len(manifest["frames"])}, indent=2))


if __name__ == "__main__":
    main()
