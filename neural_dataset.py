"""Convert a metric TUM manifest into a fixed-pose gsplat image dataset."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .io import load_scene


def world_to_camera(t_world_camera: np.ndarray) -> np.ndarray:
    pose = np.asarray(t_world_camera, dtype=np.float64)
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError("Pose must be finite 4x4")
    rotation = pose[:3, :3]
    if (not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1, atol=1e-4)
            or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-6)):
        raise ValueError("Pose must be right-handed rigid T_world_camera")
    inverse = np.eye(4)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ pose[:3, 3]
    return inverse


def split_indices(count: int) -> dict[str, list[int]]:
    if count < 8:
        raise ValueError("At least eight views needed for train/validation/test")
    train_end = int(count * .75)
    val_end = train_end + max(1, (count - train_end) // 2)
    groups = {"train": list(range(train_end)),
              "validation": list(range(train_end, val_end)),
              "test": list(range(val_end, count))}
    if min(map(len, groups.values())) == 0 or sorted(sum(groups.values(), [])) != list(range(count)):
        raise ValueError("Invalid disjoint split")
    return groups


def convert(manifest: Path, output: Path, width: int = 160,
            pose_source: str = "tum_ground_truth") -> dict:
    scene = load_scene(manifest)
    if width <= 0:
        raise ValueError("Positive output width required")
    base_width, base_height = scene.intrinsics.width, scene.intrinsics.height
    height = round(base_height * width / base_width)
    sx, sy = width / base_width, height / base_height
    splits = split_indices(len(scene.frames))
    image_dir, depth_dir = output / "images", output / "depth"
    image_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for index, frame in enumerate(scene.frames):
        pose = np.asarray(frame.t_world_camera)
        view = world_to_camera(pose)
        rgb = Image.open(frame.rgb_path).convert("RGB")
        depth = Image.open(frame.depth_path)
        if rgb.size != (base_width, base_height) or depth.size != rgb.size:
            raise ValueError("Source image dimensions disagree with intrinsics")
        rgb_name, depth_name = f"{index:04d}.png", f"{index:04d}.png"
        rgb.resize((width, height), Image.Resampling.BILINEAR).save(image_dir / rgb_name)
        depth.resize((width, height), Image.Resampling.NEAREST).save(depth_dir / depth_name)
        group = next(name for name, indices in splits.items() if index in indices)
        frames.append({"index": index, "timestamp_s": frame.timestamp_s,
                       "image": f"images/{rgb_name}", "depth": f"depth/{depth_name}",
                       "split": group, "T_world_camera": pose.tolist(),
                       "T_camera_world": view.tolist()})
    result = {"schema_version": 1, "source_manifest": str(manifest.resolve()),
              "pose_source": pose_source, "pose_direction": "T_world_camera",
              "viewmat_direction": "T_camera_world", "units": "meters",
              "camera_convention": "OpenCV optical: +x right, +y down, +z forward",
              "depth_usage": "GT sensor depth only for Gaussian initialization; no depth loss",
              "split_protocol": "chronological first 75% train, next validation, final test",
              "width": width, "height": height, "original_width": base_width,
              "original_height": base_height, "depth_scale": scene.depth_scale,
              "intrinsics": {"fx": scene.intrinsics.fx * sx, "fy": scene.intrinsics.fy * sy,
                             "cx": scene.intrinsics.cx * sx, "cy": scene.intrinsics.cy * sy},
              "splits": splits, "frames": frames}
    (output / "dataset.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument("--pose-source", default="tum_ground_truth")
    args = parser.parse_args()
    print(json.dumps({key: value for key, value in convert(
        args.manifest, args.output, args.width, args.pose_source).items()
                      if key != "frames"}, indent=2))


if __name__ == "__main__":
    main()
