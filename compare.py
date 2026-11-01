"""Compare known-pose voxel and TSDF outputs with observed-surface support."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .io import load_rgbd, load_scene


def _observed_sample(manifest: Path, min_depth_m: float, max_depth_m: float,
                     pixel_stride: int = 8) -> np.ndarray:
    scene = load_scene(manifest)
    k = scene.intrinsics
    points = []
    for frame in scene.frames:
        _, depth = load_rgbd(frame, scene)
        sample = depth[::pixel_stride, ::pixel_stride]
        valid = np.isfinite(sample) & (sample >= min_depth_m) & (sample <= max_depth_m)
        v, u = np.nonzero(valid)
        u, v = u * pixel_stride, v * pixel_stride
        z = depth[v, u]
        camera = np.column_stack(((u-k.cx)*z/k.fx, (v-k.cy)*z/k.fy, z))
        points.append(camera @ frame.t_world_camera[:3, :3].T + frame.t_world_camera[:3, 3])
    return np.concatenate(points)


def _project(points: np.ndarray, pose: np.ndarray, k) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = (points - pose[:3, 3]) @ pose[:3, :3]
    z = camera[:, 2]
    valid = z > 0
    u = np.rint(k.fx * camera[:, 0] / np.maximum(z, 1e-9) + k.cx).astype(int)
    v = np.rint(k.fy * camera[:, 1] / np.maximum(z, 1e-9) + k.cy).astype(int)
    valid &= (u >= 0) & (u < k.width) & (v >= 0) & (v < k.height)
    return np.column_stack((u, v)), z, valid


def _draw_cloud(cloud, pose, k, title: str) -> Image.Image:
    points = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)
    uv, z, valid = _project(points, pose, k)
    image = Image.new("RGB", (k.width, k.height), (16, 20, 30))
    draw = ImageDraw.Draw(image)
    for index in np.flatnonzero(valid)[np.argsort(z[valid])[::-1]]:
        u, v = uv[index]
        color = tuple(int(x) for x in np.clip(colors[index] * 255, 0, 255))
        draw.ellipse((u-1, v-1, u+1, v+1), fill=color)
    draw.text((12, 12), title, fill="white")
    return image


def _draw_mesh(mesh, pose, k) -> Image.Image:
    points = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)
    uv, z, valid = _project(points, pose, k)
    image = Image.new("RGB", (k.width, k.height), (16, 20, 30))
    draw = ImageDraw.Draw(image)
    visible = valid[faces].all(axis=1)
    face_ids = np.flatnonzero(visible)
    for index in face_ids[np.argsort(z[faces[face_ids]].mean(axis=1))[::-1]]:
        face = faces[index]
        color = tuple(int(x) for x in np.clip(colors[face].mean(axis=0) * 255, 0, 255))
        draw.polygon([tuple(uv[vertex]) for vertex in face], fill=color)
    draw.text((12, 12), "Open3D TSDF mesh", fill="white")
    return image


def compare(manifest: Path, voxel_dir: Path, tsdf_dir: Path, output: Path,
            support_threshold_m: float = 0.06) -> dict:
    import open3d as o3d
    from scipy.spatial import cKDTree

    voxel = json.loads((voxel_dir / "run.json").read_text(encoding="utf-8"))
    tsdf = json.loads((tsdf_dir / "run.json").read_text(encoding="utf-8"))
    if not 0 < support_threshold_m or voxel["frames"] != tsdf["frames"] or \
       voxel["valid_input_pixels"] != tsdf["valid_input_pixels"] or \
       Path(voxel["input_manifest"]).resolve() != manifest.resolve() or \
       Path(tsdf["input_manifest"]).resolve() != manifest.resolve() or \
       voxel["voxel_size_m"] != tsdf["voxel_size_m_requested"] or \
       voxel["min_depth_m"] != tsdf["min_depth_m"] or \
       voxel["max_depth_m"] != tsdf["max_depth_m"]:
        raise ValueError("Comparison requires identical manifest, frame/depth settings and voxel size")
    scene = load_scene(manifest)
    observed = _observed_sample(manifest, voxel["min_depth_m"], voxel["max_depth_m"])
    voxel_cloud = o3d.io.read_point_cloud(str(voxel_dir / "cloud.ply"))
    tsdf_cloud = o3d.io.read_point_cloud(str(tsdf_dir / "cloud.ply"))
    mesh = o3d.io.read_triangle_mesh(str(tsdf_dir / "mesh.ply"))
    support = {}
    for name, cloud in (("voxel", voxel_cloud), ("tsdf", tsdf_cloud)):
        nearest, _ = cKDTree(np.asarray(cloud.points)).query(observed, workers=-1)
        support[name] = {"fraction_within_threshold": float(np.mean(nearest <= support_threshold_m)),
                         "median_distance_m": float(np.median(nearest))}
    output.mkdir(parents=True, exist_ok=True)
    left = _draw_cloud(voxel_cloud, scene.frames[0].t_world_camera,
                       scene.intrinsics, "Naive voxel cloud")
    right = _draw_mesh(mesh, scene.frames[0].t_world_camera, scene.intrinsics)
    composite = Image.new("RGB", (left.width + right.width, left.height))
    composite.paste(left, (0, 0))
    composite.paste(right, (left.width, 0))
    composite.save(output / "comparison.png")
    report = {
        "manifest": str(manifest.resolve()), "frames": len(scene.frames),
        "observed_sample_points": len(observed),
        "coverage_proxy_definition": "Fraction of stride-8 observed RGB-D world points within nearest-output-point threshold; same input used for reconstruction, so this is support, not ground-truth accuracy or completeness.",
        "support_threshold_m": support_threshold_m,
        "voxel": {"elapsed_s": voxel["elapsed_s"], "peak_rss_bytes": voxel["peak_rss_bytes"],
                  "output_points": voxel["output_points"], "mesh_triangles": 0, **support["voxel"]},
        "tsdf": {"elapsed_s": tsdf["elapsed_s"], "peak_rss_bytes": tsdf["peak_rss_bytes"],
                 "output_points": tsdf["output_points"],
                 "mesh_triangles": tsdf["mesh_triangles"], **support["tsdf"]},
        "accuracy_claim": "No reference mesh; no geometric accuracy improvement measured.",
    }
    (output / "comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("voxel_dir", type=Path)
    parser.add_argument("tsdf_dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/comparison"))
    parser.add_argument("--support-threshold-m", type=float, default=0.06)
    args = parser.parse_args()
    print(json.dumps(compare(args.manifest, args.voxel_dir, args.tsdf_dir,
                             args.output, args.support_threshold_m), indent=2))


if __name__ == "__main__":
    main()
