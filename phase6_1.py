"""Phase 6.1 projective diagnostics and baseline-aware robust refinement.

Only metric predictions and camera poses enter this module. Evaluation sensor
depth is deliberately absent from all function signatures.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates, sobel

from .geometry import Intrinsics
from .phase6 import backproject, project, relative_camera
from .trajectory import rotation_degrees


REASON_NAMES = ('invalid_reference', 'baseline_below_minimum', 'behind_camera',
                'out_of_bounds', 'invalid_source', 'occluded', 'inconsistent', 'accepted')


@dataclass
class Correspondence:
    candidate_m: np.ndarray
    residual_m: np.ndarray
    parallax_px: np.ndarray
    basic_valid: np.ndarray
    source_valid: np.ndarray
    baseline_m: float
    rotation_deg: float
    reason: np.ndarray
    occlusion_delta_m: np.ndarray


def correspondences(ref: np.ndarray, source: np.ndarray, k: Intrinsics,
                    pose_ref: np.ndarray, pose_source: np.ndarray,
                    *, min_baseline_m: float = .005, occlusion_m: float = .08,
                    robust_threshold_m: float = .15) -> Correspondence:
    """Project once and retain rejection reasons for threshold sweeps."""
    if ref.shape != source.shape or ref.shape != (k.height, k.width):
        raise ValueError('Metric depth and intrinsics dimensions must agree')
    t = relative_camera(pose_ref, pose_source)
    baseline = float(np.linalg.norm(t[:3, 3]))
    rotation = rotation_degrees(t[:3, :3])
    h, w = ref.shape
    reason = np.zeros((h, w), dtype=np.uint8)
    if baseline < min_baseline_m:
        reason[np.isfinite(ref) & (ref > .1)] = 1
        nan = np.full(ref.shape, np.nan, dtype=np.float32)
        return Correspondence(nan, nan, nan, np.zeros_like(ref, bool), np.zeros_like(ref, bool),
                              baseline, rotation, reason, nan)
    vv, uu = np.indices(ref.shape)
    p = backproject(uu, vv, ref, k)
    q = p @ t[:3, :3].T + t[:3, 3]
    su, sv, sz = project(q, k)
    safe_u = np.nan_to_num(su, nan=-1., posinf=-1., neginf=-1.)
    safe_v = np.nan_to_num(sv, nan=-1., posinf=-1., neginf=-1.)
    sample = map_coordinates(source, [safe_v, safe_u], order=1, mode='constant', cval=np.nan)
    valid_ref = np.isfinite(ref) & (ref > .1)
    positive = valid_ref & np.isfinite(sz) & (sz > .1)
    inside = positive & (su >= 1) & (sv >= 1) & (su < w-2) & (sv < h-2)
    source_valid = inside & np.isfinite(sample) & (sample > .1)
    residual = np.abs(sample-sz)
    occ_delta = sample-sz
    parallax = np.hypot(su-uu, sv-vv)
    point_source = backproject(su, sv, sample, k)
    point_ref = (point_source-t[:3, 3]) @ t[:3, :3]
    candidate = point_ref[..., 2]
    reason[valid_ref & ~positive] = 2
    reason[positive & ~inside] = 3
    reason[inside & ~source_valid] = 4
    occ = source_valid & (occ_delta < -occlusion_m)
    reason[occ] = 5
    inconsistent = source_valid & ~occ & (residual > robust_threshold_m)
    reason[inconsistent] = 6
    accepted = source_valid & ~occ & ~inconsistent
    reason[accepted] = 7
    for array in (candidate, residual, parallax, occ_delta):
        array[~np.isfinite(array)] = np.nan
    return Correspondence(candidate.astype(np.float32), residual.astype(np.float32),
                          parallax.astype(np.float32), inside, source_valid,
                          baseline, rotation, reason, occ_delta.astype(np.float32))


def summarize_correspondence(c: Correspondence) -> dict:
    counts = {name: int(np.count_nonzero(c.reason == idx)) for idx, name in enumerate(REASON_NAMES)}
    considered = max(int(np.count_nonzero(c.reason != 0)), 1)
    residual_all = c.residual_m[c.source_valid & np.isfinite(c.residual_m)]
    accepted = c.reason == 7
    return {'baseline_m': c.baseline_m, 'relative_rotation_deg': c.rotation_deg,
            'counts': counts, 'valid_reprojections': int(c.source_valid.sum()),
            'occlusion_rejection_rate': counts['occluded']/max(int(c.source_valid.sum()), 1),
            'accepted_coverage': counts['accepted']/considered,
            'accepted_parallax_median_px': float(np.nanmedian(c.parallax_px[accepted])) if accepted.any() else None,
            'residual_median_m': float(np.median(residual_all)) if len(residual_all) else None,
            'residual_p90_m': float(np.percentile(residual_all, 90)) if len(residual_all) else None,
            'residual_histogram': np.histogram(residual_all, bins=[0,.01,.02,.04,.08,.15,.3,1,np.inf])[0].tolist()}


def edge_magnitude(depth: np.ndarray) -> np.ndarray:
    clean = np.nan_to_num(depth, nan=0., posinf=0., neginf=0.)
    return np.hypot(sobel(clean, axis=0)/8, sobel(clean, axis=1)/8)


def improved_refine(ref: np.ndarray, pair_maps: list[Correspondence], *,
                    weight: float = .5, robust_threshold_m: float = .1,
                    min_baseline_m: float = .02, occlusion_m: float = .05,
                    confidence_threshold: float = .25, neighbor_limit: int = 2,
                    max_update_m: float = .12, edge_threshold_m: float = .08) -> tuple[np.ndarray, np.ndarray, dict]:
    """Median consensus from informative neighbors; preserves unsupported pixels.

    Pair maps should be ordered by temporal distance and include their source
    frame offset as an attribute `offset` or have one map per neighbor order.
    Selection is performed per frame, without ground truth, using baseline and
    robust geometric support. At most `neighbor_limit` maps contribute.
    """
    if not 0 <= weight <= 2 or neighbor_limit < 1 or robust_threshold_m <= 0:
        raise ValueError('Invalid refinement configuration')
    selected = []
    for c in pair_maps:
        if c.baseline_m < min_baseline_m:
            continue
        valid = (c.source_valid & (c.occlusion_delta_m >= -occlusion_m) &
                 (c.residual_m <= robust_threshold_m) & np.isfinite(c.candidate_m))
        support = int(valid.sum())
        if support:
            parallax = float(np.nanmedian(c.parallax_px[valid]))
            score = support * min(parallax/3., 1.)
            selected.append((score, c, valid, parallax))
    selected.sort(key=lambda item: item[0], reverse=True)
    selected = selected[:neighbor_limit]
    if not selected:
        return ref.copy(), np.zeros_like(ref, dtype=np.float32), {'selected_neighbors': 0,
            'corrected_pixels': 0, 'mean_absolute_update_m': 0., 'median_selected_parallax_px': None}
    candidates = np.stack([np.where(valid, c.candidate_m, np.nan) for _, c, valid, _ in selected])
    vote_count = np.sum(np.isfinite(candidates), axis=0)
    # Median rejects an inconsistent candidate when at least three survive.
    with np.errstate(all='ignore'):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            consensus = np.nanmedian(candidates, axis=0)
    residual = np.stack([np.where(valid, c.residual_m, np.nan) for _, c, valid, _ in selected])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        median_residual = np.nanmedian(residual, axis=0)
    parallax_factor = np.stack([np.where(valid, np.clip(c.parallax_px/3.,0,1),0.)
                                for _, c, valid, _ in selected]).max(axis=0)
    edge_factor = np.exp(-edge_magnitude(ref)/edge_threshold_m)
    confidence = (np.clip(vote_count/2.,0,1) * parallax_factor *
                  np.exp(-np.nan_to_num(median_residual, nan=np.inf)/robust_threshold_m) * edge_factor)
    confidence = np.nan_to_num(confidence, nan=0., posinf=0.).astype(np.float32)
    apply = (vote_count >= 2) & (confidence >= confidence_threshold) & np.isfinite(consensus)
    delta = np.clip(consensus-ref, -max_update_m, max_update_m)
    result = ref.copy()
    result[apply] = ref[apply] + weight*confidence[apply]*delta[apply]
    return result, confidence, {'selected_neighbors': len(selected),
        'corrected_pixels': int(apply.sum()),
        'mean_absolute_update_m': float(np.mean(np.abs(result[apply]-ref[apply]))) if apply.any() else 0.,
        'median_selected_parallax_px': float(np.median([x[3] for x in selected]))}
