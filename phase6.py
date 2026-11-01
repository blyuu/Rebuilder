"""Geometry-guided metric depth refinement; no evaluation depth is accepted here.

RTAB-Map Feature.depth_{x,y,z} uses x-forward, y-left, z-up. The TUM
optical camera uses x-right, y-down, z-forward, hence optical=(-y,-z,x).
Those features came from RGB-D SLAM and this module is Protocol A only.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from .geometry import Intrinsics, validate_pose


def assert_refinement_inputs(paths: list[str | Path]) -> None:
    """Fail closed when the refinement input list contains evaluation depth."""
    for path in paths:
        name = str(path).lower().replace('\\', '/')
        if any(token in name for token in ('/depth/', 'groundtruth', 'gt_depth', 'oracle', 'tsdf_depth')):
            raise ValueError(f"Evaluation/GT depth in refinement inputs: {path}")


def signal_from_raw(model: str, raw: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw, dtype=np.float64)
    if raw.ndim != 2 or not np.isfinite(raw).all():
        raise ValueError("Raw depth must be a finite image")
    if model == 'depth_anything_v2_small':
        # Relative inverse-depth-like output: larger activation means nearer.
        return np.maximum(np.max(raw) - raw, 0.0) + 1e-3
    if model == 'marigold_v1_1':
        # Affine-invariant depth: near=0, far=1.
        return raw.copy()
    raise ValueError(f"Unknown depth representation: {model}")


def project(points_camera: np.ndarray, k: Intrinsics) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    p = np.asarray(points_camera, dtype=np.float64)
    z = p[..., 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        u = k.fx * p[..., 0] / z + k.cx
        v = k.fy * p[..., 1] / z + k.cy
    return u, v, z


def backproject(u: np.ndarray, v: np.ndarray, z: np.ndarray, k: Intrinsics) -> np.ndarray:
    return np.stack(((u-k.cx)*z/k.fx, (v-k.cy)*z/k.fy, z), axis=-1)


def relative_camera(t_world_ref: np.ndarray, t_world_source: np.ndarray) -> np.ndarray:
    validate_pose(t_world_ref)
    validate_pose(t_world_source)
    return np.linalg.inv(t_world_source) @ t_world_ref


def load_rtab_features(database: Path, timestamps: list[float], intrinsics: Intrinsics,
                       tolerance_s: float = .002, reprojection_px: float = 1.) -> list[np.ndarray]:
    """Return pixel u,v and metric optical z from RTAB RGB-D feature records."""
    with sqlite3.connect(database) as db:
        nodes = db.execute('SELECT id,stamp FROM Node ORDER BY stamp').fetchall()
        result = []
        for stamp in timestamps:
            node_id, matched = min(nodes, key=lambda node: abs(node[1]-stamp))
            if abs(matched-stamp) > tolerance_s:
                result.append(np.empty((0, 3), dtype=np.float64))
                continue
            rows = db.execute('SELECT pos_x,pos_y,depth_x,depth_y,depth_z FROM Feature WHERE node_id=?',
                              (node_id,)).fetchall()
            values = np.asarray(rows, dtype=np.float64).reshape(-1, 5)
            optical = np.column_stack((-values[:, 3], -values[:, 4], values[:, 2]))
            projected_u, projected_v, _ = project(optical, intrinsics)
            error = np.hypot(projected_u-values[:, 0], projected_v-values[:, 1])
            good = (np.isfinite(values).all(axis=1) & (values[:, 2] > .1) &
                    (values[:, 2] < 5.) & (error <= reprojection_px))
            result.append(values[good][:, [0, 1, 2]])
    return result


@dataclass
class Alignment:
    scale: float
    shift: float
    correspondences: int
    inlier_ratio: float
    residual_m: float
    uncertainty_m: float
    success: bool
    failure_reason: str | None = None


def sparse_align(signal: np.ndarray, observations: np.ndarray, *, min_points: int = 20,
                 robust: bool = True, mode: str = 'scale_shift') -> Alignment:
    """Fit z=a*signal+b at RTAB feature pixels, with iterative Huber weights."""
    h, w = signal.shape
    if len(observations) == 0:
        return Alignment(0, 0, 0, 0, np.nan, np.nan, False, 'no_sparse_features')
    u, v, y = observations.T
    valid = np.isfinite(observations).all(axis=1) & (u >= 1) & (v >= 1) & (u < w-2) & (v < h-2) & (y > .1) & (y < 5.)
    u, v, y = u[valid], v[valid], y[valid]
    x = map_coordinates(signal, [v, u], order=1, mode='nearest')
    good = np.isfinite(x) & (x >= 0)
    x, y = x[good], y[good]
    if len(x) < min_points or np.std(x) < 1e-4:
        return Alignment(0, 0, len(x), 0, np.nan, np.nan, False, 'insufficient_or_degenerate')
    if mode not in ('scale_shift', 'scale_only'):
        raise ValueError('mode must be scale_shift or scale_only')
    design = np.column_stack((x, np.ones_like(x))) if mode == 'scale_shift' else x[:, None]
    weights = np.ones_like(x)
    coefficients = np.linalg.lstsq(design, y, rcond=None)[0]
    a, b = (coefficients[0], coefficients[1]) if mode == 'scale_shift' else (coefficients[0], 0.)
    if robust:
        for _ in range(12):
            r = y - (a*x+b)
            sigma = max(1.4826*np.median(np.abs(r-np.median(r))), .01)
            weights = np.minimum(1., 1.5*sigma/np.maximum(np.abs(r), 1e-9))
            coefficients = np.linalg.lstsq(design*np.sqrt(weights[:, None]), y*np.sqrt(weights), rcond=None)[0]
            a, b = (coefficients[0], coefficients[1]) if mode == 'scale_shift' else (coefficients[0], 0.)
    r = y - (a*x+b)
    sigma = max(1.4826*np.median(np.abs(r-np.median(r))), .01)
    inliers = np.abs(r) <= 2.5*sigma
    success = bool(np.isfinite([a, b]).all() and a > 0 and inliers.sum() >= min_points and
                   np.mean(inliers) >= .4)
    return Alignment(float(a), float(b), len(x), float(np.mean(inliers)),
                     float(np.median(np.abs(r))), float(sigma), success,
                     None if success else 'nonphysical_or_outlier_dominated')


def view_consistency(ref_depth: np.ndarray, src_depth: np.ndarray, k: Intrinsics,
                     t_world_ref: np.ndarray, t_world_src: np.ndarray,
                     *, min_baseline_m: float = .005, occlusion_m: float = .08,
                     residual_m: float = .15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Source depth projected back to each reference ray plus valid residual mask."""
    if ref_depth.shape != src_depth.shape or ref_depth.shape != (k.height, k.width):
        raise ValueError('Depth/intrinsics shape mismatch')
    t = relative_camera(t_world_ref, t_world_src)
    if np.linalg.norm(t[:3, 3]) < min_baseline_m:
        shape = ref_depth.shape
        return np.full(shape, np.nan), np.zeros(shape, dtype=bool), np.full(shape, np.nan)
    vv, uu = np.indices(ref_depth.shape)
    p = backproject(uu, vv, ref_depth, k)
    q = p @ t[:3, :3].T + t[:3, 3]
    su, sv, sz = project(q, k)
    safe_u = np.nan_to_num(su, nan=-1., posinf=-1., neginf=-1.)
    safe_v = np.nan_to_num(sv, nan=-1., posinf=-1., neginf=-1.)
    sampled = map_coordinates(src_depth, [safe_v, safe_u], order=1, mode='constant', cval=np.nan)
    residual = np.abs(sampled-sz)
    valid = (np.isfinite(ref_depth) & (ref_depth > .1) & np.isfinite(sampled) &
             (sampled > .1) & (sz > .1) & (su >= 1) & (sv >= 1) &
             (su < k.width-2) & (sv < k.height-2) &
             (sampled >= sz-occlusion_m) & (residual <= residual_m))
    source_points = backproject(su, sv, sampled, k)
    back = (source_points-t[:3, 3]) @ t[:3, :3]
    candidate = back[..., 2]
    candidate[~valid] = np.nan
    residual[~valid] = np.nan
    return candidate, valid, residual


def refine(ref: np.ndarray, neighbors: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
           k: Intrinsics, pose: np.ndarray, *, tau_m: float = .08,
           max_update_m: float = .12) -> tuple[np.ndarray, np.ndarray, dict]:
    """One bounded robust fusion step; rejected constraints leave input unchanged."""
    candidates, weights, residuals = [], [], []
    for depth, source_pose, _ in neighbors:
        candidate, valid, residual = view_consistency(ref, depth, k, pose, source_pose)
        weight = np.where(valid, np.exp(-np.nan_to_num(residual, nan=np.inf)/tau_m), 0.)
        candidates.append(np.nan_to_num(candidate))
        weights.append(weight)
        residuals.append(np.nan_to_num(residual, nan=0.))
    if not weights:
        return ref.copy(), np.zeros_like(ref), {'support_pixels': 0, 'mean_residual_m': None}
    support = np.sum(np.stack(weights), axis=0)
    weighted = np.sum(np.stack(candidates)*np.stack(weights), axis=0)
    proposal = np.divide(weighted, support, out=ref.copy(), where=support > 0)
    delta = np.clip(proposal-ref, -max_update_m, max_update_m)
    confidence = np.clip(support / max(len(neighbors), 1), 0, 1)
    result = ref + confidence*delta
    result[~np.isfinite(ref) | (ref <= 0)] = ref[~np.isfinite(ref) | (ref <= 0)]
    count = int(np.count_nonzero(support))
    mean_residual = float(np.sum(np.stack(residuals)*np.stack(weights))/np.sum(support)) if np.sum(support) else None
    return result, confidence, {'support_pixels': count, 'mean_residual_m': mean_residual,
                                'max_change_m': float(np.nanmax(np.abs(result-ref))),
                                'iterations': 1, 'converged': True}
