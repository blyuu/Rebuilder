"""Open3D known-pose RGB-D TSDF integration using a bounded uniform volume."""

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path

import numpy as np

from .exports import write_trajectory
from .io import load_rgbd, load_scene
from .measure import PeakRSS


def run_tsdf(manifest: Path, output: Path, voxel_size_m: float = 0.02,
             sdf_trunc_m: float = 0.08, min_depth_m: float = 0.1,
             max_depth_m: float = 5.0) -> dict:
    import open3d as o3d

    if not 0 < voxel_size_m < sdf_trunc_m or not 0 < min_depth_m < max_depth_m:
        raise ValueError("Expected 0 < voxel size < truncation and 0 < min depth < max depth")
    with PeakRSS() as memory:
        start = time.perf_counter()
        scene = load_scene(manifest)
        k = scene.intrinsics
        intrinsic = o3d.camera.PinholeCameraIntrinsic(k.width, k.height, k.fx, k.fy, k.cx, k.cy)
        # Probe observed world points to bound a uniform volume. Open3D 0.20.0's
        # ScalableTSDFVolume returned an empty surface on this audited Windows host.
        low = np.full(3, np.inf)
        high = np.full(3, -np.inf)
        for frame in scene.frames:
            _, depth = load_rgbd(frame, scene)
            sampled = depth[::8, ::8]
            valid = np.isfinite(sampled) & (sampled >= min_depth_m) & (sampled <= max_depth_m)
            v, u = np.nonzero(valid)
            if not len(u):
                continue
            u, v = u * 8, v * 8
            z = depth[v, u]
            camera = np.column_stack(((u - k.cx) * z / k.fx,
                                      (v - k.cy) * z / k.fy, z))
            world = camera @ frame.t_world_camera[:3, :3].T + frame.t_world_camera[:3, 3]
            low = np.minimum(low, world.min(axis=0))
            high = np.maximum(high, world.max(axis=0))
        if not np.isfinite(low).all():
            raise ValueError("No valid depth samples for TSDF bounds")
        margin = max(2 * sdf_trunc_m, 2 * voxel_size_m)
        length = float(np.max(high - low) + 2 * margin)
        resolution = int(np.ceil(length / voxel_size_m))
        if resolution > 256:
            raise ValueError("Uniform TSDF resolution exceeds 256; increase voxel size")
        origin = (low + high) / 2 - length / 2
        volume = o3d.pipelines.integration.UniformTSDFVolume(
            length=length, resolution=resolution, sdf_trunc=sdf_trunc_m,
            color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
            origin=origin)
        valid_pixels = 0
        for index, frame in enumerate(scene.frames):
            rgb, depth = load_rgbd(frame, scene)
            valid = np.isfinite(depth) & (depth >= min_depth_m) & (depth <= max_depth_m)
            valid_pixels += int(valid.sum())
            # Legacy Open3D TSDF integration expects the conventional uint16 depth image.
            # Re-encode the validated metric values at the manifest's integer scale.
            raw_depth = np.rint(np.where(valid, depth, 0) * scene.depth_scale)
            if raw_depth.max() > np.iinfo(np.uint16).max:
                raise ValueError("Depth values exceed uint16 range after scaling")
            raw_depth = raw_depth.astype(np.uint16)
            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
                o3d.geometry.Image(np.ascontiguousarray(rgb)),
                o3d.geometry.Image(np.ascontiguousarray(raw_depth)),
                depth_scale=scene.depth_scale, depth_trunc=max_depth_m,
                convert_rgb_to_intensity=False)
            # Open3D integrate() expects extrinsic T_camera_world, inverse of manifest pose.
            volume.integrate(rgbd, intrinsic, np.linalg.inv(frame.t_world_camera))
            logging.info("tsdf frame=%d timestamp_s=%.6f valid_pixels=%d",
                         index, frame.timestamp_s, int(valid.sum()))
        mesh = volume.extract_triangle_mesh()
        mesh.compute_vertex_normals()
        cloud = volume.extract_point_cloud()
        logging.info("tsdf extracted vertices=%d triangles=%d points=%d",
                     len(mesh.vertices), len(mesh.triangles), len(cloud.points))
        if not len(mesh.triangles) or not len(cloud.points):
            raise ValueError("TSDF extraction returned an empty mesh or point cloud")
        output.mkdir(parents=True, exist_ok=True)
        if not o3d.io.write_triangle_mesh(str(output / "mesh.ply"), mesh, write_ascii=False):
            raise OSError("Failed to export mesh.ply")
        if not o3d.io.write_point_cloud(str(output / "cloud.ply"), cloud, write_ascii=False):
            raise OSError("Failed to export cloud.ply")
        write_trajectory(output / "trajectory.csv", scene)
        elapsed = time.perf_counter() - start
    report = {
        "method": "pose-provided-rgbd-open3d-uniform-tsdf",
        "input_manifest": str(manifest.resolve()), "frames": len(scene.frames),
        "valid_input_pixels": valid_pixels, "output_points": len(cloud.points),
        "mesh_vertices": len(mesh.vertices), "mesh_triangles": len(mesh.triangles),
        "voxel_size_m_requested": voxel_size_m,
        "voxel_size_m_effective": length / resolution,
        "sdf_trunc_m": sdf_trunc_m, "volume_length_m": length,
        "volume_resolution": resolution, "volume_origin_m": origin.tolist(),
        "min_depth_m": min_depth_m, "max_depth_m": max_depth_m,
        "elapsed_s": elapsed, "peak_rss_bytes": memory.peak_bytes,
        "peak_memory_method": "Windows process peak working set; sampled RSS elsewhere",
        "environment": {"python": sys.version.split()[0], "numpy": np.__version__,
                        "open3d": o3d.__version__, "platform": platform.platform()},
    }
    (output / "run.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/tsdf"))
    parser.add_argument("--voxel-size-m", type=float, default=0.02)
    parser.add_argument("--sdf-trunc-m", type=float, default=0.08)
    parser.add_argument("--min-depth-m", type=float, default=0.1)
    parser.add_argument("--max-depth-m", type=float, default=5.0)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    print(json.dumps(run_tsdf(args.manifest, args.output, args.voxel_size_m,
                              args.sdf_trunc_m, args.min_depth_m, args.max_depth_m), indent=2))


if __name__ == "__main__":
    main()
