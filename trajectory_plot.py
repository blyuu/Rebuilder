"""Plot associated GT and rigidly aligned SLAM trajectories for diagnosis."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tum import associate_nearest


def plot(comparison: Path, input_metadata: Path, tolerance_s: float = .02) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = json.loads(input_metadata.read_text())
    estimate_rows = json.loads((comparison / "estimated_trajectory.json").read_text())
    gt_rows = json.loads((comparison / "ground_truth_trajectory.json").read_text())
    evaluation = json.loads((comparison / "trajectory_evaluation.json").read_text())
    aligned = np.array(evaluation["T_gtworld_slamworld_evaluation_only"])
    stamps = [r["timestamp_s"] for r in metadata["frames"]]
    gt_stamps = [r["timestamp_s"] for r in gt_rows]
    gt_by_input = {i: np.array(gt_rows[j]["T_world_camera"])
                   for i, j, _ in associate_nearest(stamps, gt_stamps, tolerance_s, unique=False)}
    est_by_input = {i: np.array(r["T_world_camera"]) for i, r in enumerate(estimate_rows)
                    if r["tracking_status"] == "tracked"}
    common = sorted(gt_by_input.keys() & est_by_input.keys())
    gt_xyz = np.array([gt_by_input[i][:3, 3] for i in common])
    est_xyz = np.array([(aligned @ est_by_input[i])[:3, 3] for i in common])
    _, _, vh = np.linalg.svd(gt_xyz - gt_xyz.mean(0), full_matrices=False)
    g2 = (gt_xyz - gt_xyz.mean(0)) @ vh[:2].T
    e2 = (est_xyz - gt_xyz.mean(0)) @ vh[:2].T
    errors = np.linalg.norm(gt_xyz - est_xyz, axis=1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(g2[:, 0], g2[:, 1], label="GT", lw=2)
    axes[0].plot(e2[:, 0], e2[:, 1], label="RTAB aligned", lw=1)
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("PCA axis 1 (m)")
    axes[0].set_ylabel("PCA axis 2 (m)")
    axes[0].legend()
    axes[0].set_title("Matched camera centers")
    axes[1].plot(np.array(common), errors * 1000, lw=1)
    axes[1].set_xlabel("Original RGB-D frame index")
    axes[1].set_ylabel("Aligned translation error (mm)")
    axes[1].set_title(f"ATE on {len(common)}/{len(stamps)} input frames")
    output = comparison / "trajectory_comparison.png"
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    parser.add_argument("input_metadata", type=Path)
    args = parser.parse_args()
    print(plot(args.comparison, args.input_metadata))
