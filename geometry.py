"""Validated metric camera geometry. Camera axes: right, down, forward."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Intrinsics:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0 or self.fx <= 0 or self.fy <= 0:
            raise ValueError("Camera dimensions and focal lengths must be positive")
        if not np.isfinite([self.fx, self.fy, self.cx, self.cy]).all():
            raise ValueError("Camera intrinsics must be finite")
        if not (0 <= self.cx < self.width and 0 <= self.cy < self.height):
            raise ValueError("Principal point must lie within the image")


def validate_pose(t_world_camera: NDArray[np.float64]) -> None:
    if t_world_camera.shape != (4, 4) or not np.isfinite(t_world_camera).all():
        raise ValueError("T_world_camera must be a finite 4x4 matrix")
    if not np.allclose(t_world_camera[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous pose row")
    rotation = t_world_camera[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4) or not np.isclose(
        np.linalg.det(rotation), 1, atol=1e-4
    ):
        raise ValueError("Pose rotation must be a proper orthonormal matrix")


def backproject(
    depth_m: NDArray[np.float32],
    rgb: NDArray[np.uint8],
    intrinsics: Intrinsics,
    t_world_camera: NDArray[np.float64],
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    intrinsics.validate()
    validate_pose(t_world_camera)
    if depth_m.shape != (intrinsics.height, intrinsics.width):
        raise ValueError("Depth shape does not match camera intrinsics")
    if rgb.shape != (intrinsics.height, intrinsics.width, 3):
        raise ValueError("RGB shape does not match camera intrinsics")
    if not 0 < min_depth_m < max_depth_m:
        raise ValueError("Depth bounds must satisfy 0 < min < max")
    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    v, u = np.nonzero(valid)
    z = depth_m[valid].astype(np.float64)
    camera_points = np.column_stack(((u - intrinsics.cx) * z / intrinsics.fx,
                                     (v - intrinsics.cy) * z / intrinsics.fy, z))
    world_points = camera_points @ t_world_camera[:3, :3].T + t_world_camera[:3, 3]
    return world_points, rgb[valid]


def voxel_fuse(
    points: NDArray[np.float64], colors: NDArray[np.uint8], voxel_size_m: float
) -> tuple[NDArray[np.float64], NDArray[np.uint8], NDArray[np.int64]]:
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError("Voxel size must be positive and finite")
    if points.ndim != 2 or points.shape[1] != 3 or colors.shape != points.shape:
        raise ValueError("Points and colors must both have shape (N, 3)")
    if not np.isfinite(points).all():
        raise ValueError("Point cloud contains nonfinite coordinates")
    if len(points) == 0:
        return points.copy(), colors.copy(), np.empty(0, dtype=np.int64)
    keys = np.floor(points / voxel_size_m).astype(np.int64)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(counts), 3), dtype=np.float64)
    color_sums = np.zeros_like(sums)
    np.add.at(sums, inverse, points)
    np.add.at(color_sums, inverse, colors)
    return sums / counts[:, None], np.rint(color_sums / counts[:, None]).astype(np.uint8), counts
