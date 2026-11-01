"""TUM RGB-D PNG sequence to the RECON-X metric manifest contract.

Source format: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats
"""

import bisect
import json
import re
from pathlib import Path

import numpy as np

from .geometry import Intrinsics, validate_pose

TUM_DEPTH_SCALE = 5000.0
TUM_INTRINSICS = {
    # TUM recommends the ROS default for pre-registered depth in fr1/fr2.
    "fr1": Intrinsics(640, 480, 525.0, 525.0, 319.5, 239.5),
    "fr2": Intrinsics(640, 480, 525.0, 525.0, 319.5, 239.5),
    # Freiburg 3 images are already undistorted; RGB calibration is applicable.
    "fr3": Intrinsics(640, 480, 535.4, 539.2, 320.1, 247.6),
}


def _rows(path: Path, columns: int) -> list[tuple[float, list[str]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        fields = line.split("#", 1)[0].split()
        if not fields:
            continue
        if len(fields) != columns:
            raise ValueError(f"{path}:{line_number}: expected {columns} fields")
        timestamp = float(fields[0])
        if not np.isfinite(timestamp):
            raise ValueError(f"{path}:{line_number}: nonfinite timestamp")
        rows.append((timestamp, fields[1:]))
    if not rows or any(a[0] >= b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError(f"{path}: timestamps must be nonempty and strictly increasing")
    return rows


def associate_nearest(source: list[float], target: list[float], tolerance_s: float,
                      unique: bool = True) -> list[tuple[int, int, float]]:
    """Find nearest timestamp within tolerance, optionally using each target once."""
    if not np.isfinite(tolerance_s) or tolerance_s < 0:
        raise ValueError("Timestamp tolerance must be finite and nonnegative")
    if any(a >= b for a, b in zip(source, source[1:])) or any(
        a >= b for a, b in zip(target, target[1:])
    ):
        raise ValueError("Timestamps must be strictly increasing")
    matches = []
    last_target = -1
    for source_index, stamp in enumerate(source):
        insertion = bisect.bisect_left(target, stamp)
        candidates = (insertion - 1, insertion)
        valid = [index for index in candidates if 0 <= index < len(target)
                 and (not unique or index > last_target)]
        if not valid:
            continue
        target_index = min(valid, key=lambda index: (abs(target[index] - stamp), index))
        delta = target[target_index] - stamp
        if abs(delta) <= tolerance_s:
            matches.append((source_index, target_index, delta))
            if unique:
                last_target = target_index
    return matches


def quaternion_to_pose(translation: list[float], quaternion_xyzw: list[float]) -> np.ndarray:
    t = np.asarray(translation, dtype=np.float64)
    q = np.asarray(quaternion_xyzw, dtype=np.float64)
    if t.shape != (3,) or q.shape != (4,) or not np.isfinite(t).all() or not np.isfinite(q).all():
        raise ValueError("Invalid TUM translation/quaternion")
    norm = np.linalg.norm(q)
    if abs(norm - 1) > 1e-2:
        raise ValueError("TUM quaternion is not near unit length")
    x, y, z, w = q / norm
    rotation = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    pose = np.eye(4)
    pose[:3, :3] = rotation
    pose[:3, 3] = t
    validate_pose(pose)
    return pose


def _safe_image_path(root: Path, name: str) -> str:
    relative = Path(name)
    absolute = (root / relative).resolve()
    if absolute.suffix.lower() != ".png" or not absolute.is_relative_to(root.resolve()) or not absolute.is_file():
        raise ValueError(f"Invalid or missing TUM PNG: {name}")
    return relative.as_posix()


def make_tum_manifest(sequence: Path, output: Path, *, rgb_depth_tolerance_s: float = 0.02,
                      pose_tolerance_s: float = 0.02, stride: int = 1,
                      max_frames: int | None = None) -> dict:
    sequence = sequence.resolve()
    match = re.search(r"freiburg([123])", sequence.name)
    if not match:
        raise ValueError("Sequence folder must identify freiburg1, freiburg2 or freiburg3")
    family = f"fr{match.group(1)}"
    if stride < 1 or max_frames is not None and max_frames < 1:
        raise ValueError("stride and max_frames must be positive")
    intrinsics = TUM_INTRINSICS[family]
    intrinsics.validate()
    rgb = _rows(sequence / "rgb.txt", 2)
    depth = _rows(sequence / "depth.txt", 2)
    groundtruth = _rows(sequence / "groundtruth.txt", 8)
    image_pairs = associate_nearest([row[0] for row in rgb], [row[0] for row in depth],
                                    rgb_depth_tolerance_s)
    pose_matches = associate_nearest([rgb[i][0] for i, _, _ in image_pairs],
                                     [row[0] for row in groundtruth], pose_tolerance_s, unique=False)
    selected = pose_matches[::stride]
    if max_frames is not None:
        selected = selected[:max_frames]
    if not selected:
        raise ValueError("No associated RGB/depth/pose frames within tolerances")
    output.parent.mkdir(parents=True, exist_ok=True)
    frames = []
    for pair_index, pose_index, pose_delta in selected:
        rgb_index, depth_index, depth_delta = image_pairs[pair_index]
        pose_fields = groundtruth[pose_index][1]
        pose = quaternion_to_pose([float(value) for value in pose_fields[:3]],
                                  [float(value) for value in pose_fields[3:]])
        frames.append({
            "timestamp_s": rgb[rgb_index][0],
            "rgb": (sequence / _safe_image_path(sequence, rgb[rgb_index][1][0])).as_posix(),
            "depth": (sequence / _safe_image_path(sequence, depth[depth_index][1][0])).as_posix(),
            "depth_timestamp_s": depth[depth_index][0],
            "pose_timestamp_s": groundtruth[pose_index][0],
            "rgb_depth_delta_s": depth_delta,
            "rgb_pose_delta_s": pose_delta,
            "T_world_camera": pose.tolist(),
        })
    manifest = {
        "schema_version": 1,
        "source": {"dataset": "TUM RGB-D", "sequence": sequence.name,
                   "url": "https://cvg.cit.tum.de/data/datasets/rgbd-dataset/file_formats",
                   "rgb_depth_tolerance_s": rgb_depth_tolerance_s,
                   "pose_tolerance_s": pose_tolerance_s, "stride": stride,
                   "max_frames": max_frames, "full_rgb_count": len(rgb),
                   "associated_image_pairs": len(image_pairs),
                   "associated_triplets": len(pose_matches),
                   "camera_calibration": family,
                   "pose_direction": "T_world_camera"},
        "intrinsics": vars(intrinsics),
        "depth_unit": "integer_scaled_meters", "depth_scale": TUM_DEPTH_SCALE,
        "frames": frames,
    }
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
