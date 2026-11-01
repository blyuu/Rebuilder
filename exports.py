"""Shared camera path export."""

import csv
from pathlib import Path

from .io import Scene


def write_trajectory(path: Path, scene: Scene) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["timestamp_s", "world_x_m", "world_y_m", "world_z_m"])
        for frame in scene.frames:
            writer.writerow([frame.timestamp_s, *frame.t_world_camera[:3, 3]])
