"""Run the known-pose RGB-D point cloud baseline."""

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np

from . import __version__
from .exports import write_trajectory
from .geometry import backproject, voxel_fuse
from .io import load_rgbd, load_scene, write_ply
from .measure import PeakRSS


def run(manifest: Path, output: Path, voxel_size_m: float,
        min_depth_m: float, max_depth_m: float) -> dict:
    with PeakRSS() as memory:
        start = time.perf_counter()
        scene = load_scene(manifest)
        points_by_frame = []
        colors_by_frame = []
        for index, frame in enumerate(scene.frames):
            rgb, depth = load_rgbd(frame, scene)
            points, colors = backproject(depth, rgb, scene.intrinsics, frame.t_world_camera,
                                         min_depth_m, max_depth_m)
            logging.info("frame=%d timestamp_s=%.6f valid_points=%d", index, frame.timestamp_s, len(points))
            points_by_frame.append(points)
            colors_by_frame.append(colors)
        points = np.concatenate(points_by_frame)
        colors = np.concatenate(colors_by_frame)
        fused, fused_colors, counts = voxel_fuse(points, colors, voxel_size_m)
        if not len(fused):
            raise ValueError("No valid points remain after depth filtering")
        output.mkdir(parents=True, exist_ok=True)
        write_ply(output / "cloud.ply", fused, fused_colors)
        write_trajectory(output / "trajectory.csv", scene)
        elapsed = time.perf_counter() - start
    report = {
        "method": "known-pose-rgbd-naive-voxel-fusion",
        "version": __version__, "input_manifest": str(manifest.resolve()),
        "frames": len(scene.frames), "raw_points": len(points), "fused_points": len(fused),
        "valid_input_pixels": len(points), "output_points": len(fused), "mesh_triangles": 0,
        "observations_per_voxel_mean": float(counts.mean()),
        "voxel_size_m": voxel_size_m, "min_depth_m": min_depth_m, "max_depth_m": max_depth_m,
        "elapsed_s": elapsed, "peak_rss_bytes": memory.peak_bytes,
        "peak_memory_method": "Windows process peak working set; sampled RSS elsewhere",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "platform": platform.platform()},
        "limitations": ["known metric depth and poses", "no TSDF", "no mesh", "no real-dataset quality metrics"],
    }
    (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline"))
    parser.add_argument("--voxel-size-m", type=float, default=0.02)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(run(args.manifest, args.output, args.voxel_size_m,
                         args.min_depth_m, args.max_depth_m), indent=2))


if __name__ == "__main__":
    main()
