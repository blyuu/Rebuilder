"""Diagnostic aligned mesh comparison; reference TSDF is not shape ground truth."""

import argparse
import json
from pathlib import Path

import numpy as np


def compare(root: Path) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import open3d as o3d
    from scipy.spatial import cKDTree

    evaluation = json.loads((root / "trajectory_evaluation.json").read_text())
    transform = np.array(evaluation["T_gtworld_slamworld_evaluation_only"])
    gt = o3d.io.read_triangle_mesh(str(root / "gt_tsdf/mesh.ply"))
    slam = o3d.io.read_triangle_mesh(str(root / "slam_tsdf/mesh.ply"))
    if not gt.has_triangles() or not slam.has_triangles():
        raise ValueError("Both meshes must have triangles")
    slam.transform(transform)
    gt_points = np.asarray(gt.vertices)
    slam_points = np.asarray(slam.vertices)
    nearest = cKDTree(gt_points).query(slam_points, workers=-1)[0]
    colors = plt.get_cmap("turbo")(np.clip(nearest / .05, 0, 1))[:, :3]
    slam.vertex_colors = o3d.utility.Vector3dVector(colors)
    if not o3d.io.write_triangle_mesh(str(root / "slam_mesh_aligned_difference.ply"), slam):
        raise OSError("Could not export diagnostic mesh")
    # PCA axes provide an informative 2D view regardless of TUM world orientation.
    combined = np.vstack((gt_points, slam_points))
    center = combined.mean(axis=0)
    _, _, vh = np.linalg.svd(combined[::max(1, len(combined)//20000)] - center, full_matrices=False)
    g2 = (gt_points - center) @ vh[:2].T
    s2 = (slam_points - center) @ vh[:2].T
    step_g = max(1, len(g2) // 15000)
    step_s = max(1, len(s2) // 15000)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].scatter(*g2[::step_g].T, s=.2, c="#3a78a7")
    axes[0].set_title("GT pose TSDF (reference)")
    axes[1].scatter(*s2[::step_s].T, s=.2, c="#b9554b")
    axes[1].set_title("SLAM pose TSDF (rigidly aligned)")
    plot = axes[2].scatter(*s2[::step_s].T, s=.3, c=nearest[::step_s],
                           cmap="turbo", vmin=0, vmax=.05)
    axes[2].set_title("Nearest reference vertex distance")
    fig.colorbar(plot, ax=axes[2], label="m, clipped at 0.05")
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("PCA axis 1 (m)")
        ax.set_ylabel("PCA axis 2 (m)")
    fig.savefig(root / "mesh_comparison.png", dpi=180)
    plt.close(fig)
    report = {"comparison": "nearest vertex distance after evaluation-only SE3 alignment; diagnostic, not accuracy",
              "reference_vertices": len(gt_points), "slam_vertices": len(slam_points),
              "slam_to_reference_nearest_vertex_median_m": float(np.median(nearest)),
              "slam_to_reference_nearest_vertex_p95_m": float(np.percentile(nearest, 95))}
    (root / "mesh_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.root), indent=2))
