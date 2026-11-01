"""Metric camera-optical T_world_camera adapters for trajectory formats."""

from pathlib import Path
import numpy as np

from .geometry import validate_pose
from .tum import quaternion_to_pose


def tum_trajectory(path: Path) -> dict[float, np.ndarray]:
    """TUM/RTAB-Map format 1/ORB-SLAM3: stamp tx ty tz qx qy qz qw."""
    poses = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != 8:
            raise ValueError(f"{path}:{number}: expected 8 values")
        values = list(map(float, fields))
        stamp = values[0]
        if not np.isfinite(stamp) or stamp in poses:
            raise ValueError(f"{path}:{number}: invalid or duplicate timestamp")
        poses[stamp] = quaternion_to_pose(values[1:4], values[4:8])
    if not poses:
        raise ValueError(f"Empty trajectory: {path}")
    return dict(sorted(poses.items()))


def colmap_world_camera(qwxyz: list[float], t_world_to_camera: list[float],
                        metric_scale: float | None = None) -> np.ndarray:
    """Invert COLMAP images.txt world-to-camera; scale must be externally known."""
    if metric_scale is None or not np.isfinite(metric_scale) or metric_scale <= 0:
        raise ValueError("COLMAP SfM requires an external metric scale")
    world_to_camera = quaternion_to_pose(t_world_to_camera,
                                         [*qwxyz[1:], qwxyz[0]])
    camera_to_world = np.linalg.inv(world_to_camera)
    camera_to_world[:3, 3] *= metric_scale
    validate_pose(camera_to_world)
    return camera_to_world


def load_estimated_poses(source: str, path: Path, *, metric_scale: float | None = None,
                         image_timestamps: dict[str, float] | None = None) -> dict[float, np.ndarray]:
    if source in {"tum_ground_truth", "rtabmap", "orb_slam3"}:
        return tum_trajectory(path)
    if source != "colmap":
        raise ValueError(f"Unsupported pose source: {source}")
    if image_timestamps is None:
        raise ValueError("COLMAP requires an image-name timestamp mapping")
    poses = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines:
        parts = line.split()
        if not parts or line.lstrip().startswith("#") or len(parts) < 10:
            continue
        try:
            int(parts[0])
            values = list(map(float, parts[1:8]))
        except ValueError:
            continue
        name = parts[9]
        if name not in image_timestamps:
            continue
        poses[image_timestamps[name]] = colmap_world_camera(values[:4], values[4:7], metric_scale)
    if not poses:
        raise ValueError("No COLMAP poses with mapped image timestamps")
    return dict(sorted(poses.items()))
