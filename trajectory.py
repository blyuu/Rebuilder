"""SE(3) trajectory evaluation; rigid alignment never changes metric scale."""

import math
import numpy as np

from .tum import associate_nearest


def rigid_alignment(estimated: np.ndarray, reference: np.ndarray) -> np.ndarray:
    if estimated.shape != reference.shape or estimated.ndim != 2 or estimated.shape[1] != 3 or len(estimated) < 3:
        raise ValueError("At least three corresponding 3D positions are required")
    x, y = estimated - estimated.mean(0), reference - reference.mean(0)
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.eye(3)
    correction[-1, -1] = np.linalg.det(u @ vt)
    rotation = u @ correction @ vt
    transform = np.eye(4)
    transform[:3, :3] = rotation.T
    transform[:3, 3] = reference.mean(0) - rotation.T @ estimated.mean(0)
    return transform


def rotation_degrees(rotation: np.ndarray) -> float:
    return math.degrees(math.acos(float(np.clip((np.trace(rotation) - 1) / 2, -1, 1))))


def _stats(translations: list[float], rotations: list[float]) -> dict:
    return {"translation_rmse_m": float(np.sqrt(np.mean(np.square(translations)))),
            "translation_mean_m": float(np.mean(translations)),
            "rotation_mean_deg": float(np.mean(rotations)),
            "rotation_rmse_deg": float(np.sqrt(np.mean(np.square(rotations))))}


def evaluate_trajectory(input_timestamps: list[float], estimated: dict[float, np.ndarray],
                        ground_truth: dict[float, np.ndarray], tolerance_s: float = 0.02) -> dict:
    est_stamps, gt_stamps = list(estimated), list(ground_truth)
    est_pairs = associate_nearest(input_timestamps, est_stamps, tolerance_s)
    gt_pairs = associate_nearest(input_timestamps, gt_stamps, tolerance_s, unique=False)
    est_by_input = {i: estimated[est_stamps[j]] for i, j, _ in est_pairs}
    gt_by_input = {i: ground_truth[gt_stamps[j]] for i, j, _ in gt_pairs}
    common = sorted(est_by_input.keys() & gt_by_input.keys())
    if len(common) < 3:
        raise ValueError("Insufficient associated GT and estimated poses")
    est = [est_by_input[i] for i in common]
    gt = [gt_by_input[i] for i in common]
    alignment = rigid_alignment(np.array([p[:3, 3] for p in est]),
                                np.array([p[:3, 3] for p in gt]))

    def absolute(poses):
        return _stats([float(np.linalg.norm(a[:3, 3] - b[:3, 3])) for a, b in zip(poses, gt)],
                      [rotation_degrees(a[:3, :3] @ b[:3, :3].T) for a, b in zip(poses, gt)])

    aligned = [alignment @ pose for pose in est]
    rel_translation, rel_rotation = [], []
    for first, second in zip(common, common[1:]):
        if second != first + 1:
            continue  # A tracking gap is reported, never bridged in RPE.
        est_rel = np.linalg.inv(est_by_input[first]) @ est_by_input[second]
        gt_rel = np.linalg.inv(gt_by_input[first]) @ gt_by_input[second]
        error = np.linalg.inv(gt_rel) @ est_rel
        rel_translation.append(float(np.linalg.norm(error[:3, 3])))
        rel_rotation.append(rotation_degrees(error[:3, :3]))
    return {"input_frames": len(input_timestamps), "tracked_frames": len(est_by_input),
            "tracking_failures": len(input_timestamps) - len(est_by_input),
            "tracking_success_rate": len(est_by_input) / len(input_timestamps),
            "evaluated_frames": len(common), "rpe_pairs": len(rel_translation),
            "ate_before_alignment": absolute(est), "ate_after_rigid_alignment": absolute(aligned),
            "rpe_consecutive": _stats(rel_translation, rel_rotation) if rel_translation else None,
            "T_gtworld_slamworld_evaluation_only": alignment.tolist(),
            "estimated_pose_missing_input_indices": sorted(set(range(len(input_timestamps))) - est_by_input.keys())}
