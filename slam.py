"""Evaluate a real SLAM trajectory and build matched TSDF manifests."""

import argparse
import json
import re
from pathlib import Path

from .poses import load_estimated_poses
from .trajectory import evaluate_trajectory
from .tum import _rows, associate_nearest, quaternion_to_pose


def run(input_metadata: Path, pose_file: Path, base_manifest: Path,
        output: Path, source: str = "rtabmap", tolerance_s: float = 0.02,
        metric_scale: float | None = None,
        image_timestamps: dict[str, float] | None = None) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    meta = json.loads(input_metadata.read_text(encoding="utf-8"))
    frames = meta["frames"]
    timestamps = [frame["timestamp_s"] for frame in frames]
    estimated = load_estimated_poses(source, pose_file, metric_scale=metric_scale,
                                    image_timestamps=image_timestamps)
    gt_rows = _rows(Path(meta["source"]) / "groundtruth.txt", 8)
    ground_truth = {stamp: quaternion_to_pose([float(v) for v in fields[:3]],
                                                [float(v) for v in fields[3:]])
                    for stamp, fields in gt_rows}
    evaluation = evaluate_trajectory(timestamps, estimated, ground_truth, tolerance_s)
    base = json.loads(base_manifest.read_text(encoding="utf-8"))
    est_stamps = list(estimated)
    matches = associate_nearest([f["timestamp_s"] for f in base["frames"]],
                                est_stamps, tolerance_s)
    gt_frames, slam_frames = [], []
    for index, pose_index, delta in matches:
        original = base["frames"][index].copy()
        for image_key in ("rgb", "depth"):
            original[image_key] = str((base_manifest.parent / original[image_key]).resolve())
        gt_frames.append(original.copy())
        tracked = original.copy()
        tracked["T_world_camera"] = estimated[est_stamps[pose_index]].tolist()
        tracked["pose_timestamp_s"] = est_stamps[pose_index]
        tracked["rgb_pose_delta_s"] = delta
        tracked["pose_source"] = source
        slam_frames.append(tracked)
    if not slam_frames:
        raise ValueError("No estimated poses matched reconstruction frames")
    for label, selected in (("gt", gt_frames), ("slam", slam_frames)):
        manifest = {**base, "frames": selected,
                    "source": {**base.get("source", {}), "pose_source":
                               "TUM ground truth, evaluation/reference only" if label == "gt" else source,
                               "matched_frame_count": len(selected)}}
        (output / f"{label}_scene.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pose_rows = []
    stamp_list = list(estimated)
    tracked_pairs = associate_nearest(timestamps, stamp_list, tolerance_s)
    by_index = {i: (stamp_list[j], delta) for i, j, delta in tracked_pairs}
    for i, stamp in enumerate(timestamps):
        entry = {"timestamp_s": stamp, "tracking_status": "tracked" if i in by_index else "lost"}
        if i in by_index:
            pose_stamp, delta = by_index[i]
            pose = estimated[pose_stamp]
            entry.update({"pose_timestamp_s": pose_stamp, "association_delta_s": delta,
                          "translation_m": pose[:3, 3].tolist(),
                          "rotation_matrix": pose[:3, :3].tolist(),
                          "T_world_camera": pose.tolist()})
        pose_rows.append(entry)
    (output / "estimated_trajectory.json").write_text(json.dumps(pose_rows, indent=2), encoding="utf-8")
    (output / "ground_truth_trajectory.json").write_text(
        json.dumps([{"timestamp_s": s, "T_world_camera": p.tolist()} for s, p in ground_truth.items()],
                   indent=2), encoding="utf-8")
    pose_format = ("COLMAP images.txt qw qx qy qz tx ty tz world-to-camera; inverted, external metric scale"
                   if source == "colmap" else
                   "TUM timestamp tx ty tz qx qy qz qw; optical camera-to-world; metric")
    report = {"framework": source, "pose_file": str(pose_file.resolve()),
              "pose_format": pose_format, "metric_scale_applied": metric_scale,
              "reconstruction_matched_frames": len(slam_frames), **evaluation}
    (output / "trajectory_evaluation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_metadata", type=Path)
    parser.add_argument("pose_file", type=Path)
    parser.add_argument("base_manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", choices=["rtabmap", "orb_slam3", "colmap", "tum_ground_truth"], default="rtabmap")
    parser.add_argument("--metric-scale", type=float,
                        help="Externally calibrated COLMAP SfM units-to-meters factor")
    parser.add_argument("--colmap-timestamps", type=Path,
                        help="JSON mapping from COLMAP image name to TUM timestamp")
    args = parser.parse_args()
    name_stamps = json.loads(args.colmap_timestamps.read_text()) if args.colmap_timestamps else None
    print(json.dumps(run(args.input_metadata, args.pose_file, args.base_manifest,
                         args.output, args.source, metric_scale=args.metric_scale,
                         image_timestamps=name_stamps), indent=2))


if __name__ == "__main__":
    main()
