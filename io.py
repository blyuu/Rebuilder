"""Scene manifest loading and portable PLY output."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .geometry import Intrinsics, validate_pose


@dataclass(frozen=True)
class Frame:
    timestamp_s: float
    rgb_path: Path
    depth_path: Path
    t_world_camera: NDArray[np.float64]


@dataclass(frozen=True)
class Scene:
    intrinsics: Intrinsics
    depth_scale: float
    frames: list[Frame]


def load_scene(manifest: Path) -> Scene:
    data = json.loads(manifest.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("depth_unit") != "integer_scaled_meters":
        raise ValueError("Expected schema_version=1 and depth_unit=integer_scaled_meters")
    intrinsics = Intrinsics(**data["intrinsics"])
    intrinsics.validate()
    scale = float(data["depth_scale"])
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("depth_scale must be positive and finite")
    frames = []
    previous_time = -np.inf
    for entry in data["frames"]:
        timestamp = float(entry["timestamp_s"])
        if not np.isfinite(timestamp) or timestamp <= previous_time:
            raise ValueError("Frame timestamps must be finite and strictly increasing")
        previous_time = timestamp
        pose = np.asarray(entry["T_world_camera"], dtype=np.float64)
        validate_pose(pose)
        rgb_path = manifest.parent / entry["rgb"]
        depth_path = manifest.parent / entry["depth"]
        if not rgb_path.is_file() or not depth_path.is_file():
            raise FileNotFoundError(f"Missing RGB/depth pair: {rgb_path}, {depth_path}")
        frames.append(Frame(timestamp, rgb_path, depth_path, pose))
    if not frames:
        raise ValueError("Scene has no frames")
    return Scene(intrinsics, scale, frames)


def load_rgbd(frame: Frame, scene: Scene) -> tuple[NDArray[np.uint8], NDArray[np.float32]]:
    with Image.open(frame.rgb_path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    with Image.open(frame.depth_path) as image:
        raw = np.asarray(image)
    if rgb.shape != (scene.intrinsics.height, scene.intrinsics.width, 3):
        raise ValueError(f"RGB dimensions do not match intrinsics: {frame.rgb_path}")
    if raw.shape != (scene.intrinsics.height, scene.intrinsics.width) or raw.dtype != np.uint16:
        raise ValueError(f"Depth must be aligned uint16 PNG matching intrinsics: {frame.depth_path}")
    depth = raw.astype(np.float32) / scene.depth_scale
    return rgb, depth


def write_ply(path: Path, points: NDArray[np.float64], colors: NDArray[np.uint8]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write("ply\nformat ascii 1.0\n")
        stream.write(f"element vertex {len(points)}\n")
        stream.write("property float x\nproperty float y\nproperty float z\n")
        stream.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
        for xyz, rgb in zip(points, colors):
            stream.write(f"{xyz[0]:.7f} {xyz[1]:.7f} {xyz[2]:.7f} "
                         f"{int(rgb[0])} {int(rgb[1])} {int(rgb[2])}\n")
