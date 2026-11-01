"""Metric semantic voxel fusion. Semantic IDs come from a model configuration."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .geometry import Intrinsics, validate_pose
from .io import write_ply


def validate_taxonomy(id_to_label: dict[int, str]) -> dict[int, str]:
    labels = {int(k): str(v) for k, v in id_to_label.items()}
    if not labels or sorted(labels) != list(range(len(labels))) or any(not v.strip() for v in labels.values()):
        raise ValueError('Taxonomy must contain contiguous IDs 0..C-1 and nonempty names')
    return labels


def palette(n: int) -> np.ndarray:
    # Stable, distinct enough for reproducible class-colored exports.
    ids = np.arange(n, dtype=np.uint32)
    return np.column_stack((((ids * 73 + 31) % 223 + 16),
                            ((ids * 151 + 47) % 223 + 16),
                            ((ids * 199 + 97) % 223 + 16))).astype(np.uint8)


def backproject_semantics(depth: np.ndarray, labels: np.ndarray, confidence: np.ndarray,
                          intrinsics: Intrinsics, t_world_camera: np.ndarray, *,
                          min_depth_m: float = .1, max_depth_m: float = 4.,
                          num_classes: int, unknown_id: int = -1,
                          stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    intrinsics.validate()
    validate_pose(t_world_camera)
    shape = (intrinsics.height, intrinsics.width)
    if any(x.shape != shape for x in (depth, labels, confidence)) or stride < 1:
        raise ValueError('Aligned depth, labels, confidence, and intrinsics are required')
    if not 0 < min_depth_m < max_depth_m:
        raise ValueError('Invalid metric depth bounds')
    v, u = np.mgrid[0:shape[0]:stride, 0:shape[1]:stride]
    z = depth[::stride, ::stride]
    ids = labels[::stride, ::stride]
    scores = confidence[::stride, ::stride]
    geometry_valid = np.isfinite(z) & (z >= min_depth_m) & (z <= max_depth_m)
    semantic_valid = (ids >= 0) & (ids < num_classes) & (ids != unknown_id)
    valid = geometry_valid & semantic_valid & np.isfinite(scores) & (scores > 0) & (scores <= 1)
    z = z[valid].astype(np.float64)
    camera = np.column_stack(((u[valid]-intrinsics.cx)*z/intrinsics.fx,
                              (v[valid]-intrinsics.cy)*z/intrinsics.fy, z))
    world = camera @ t_world_camera[:3, :3].T + t_world_camera[:3, 3]
    return world, ids[valid].astype(np.int32), scores[valid].astype(np.float32), int(geometry_valid.sum())


class SemanticVoxelMap:
    """Per-voxel weighted class votes, point sums, and observation counts."""

    def __init__(self, voxel_size_m: float, id_to_label: dict[int, str], coordinate_unit: str = 'm'):
        if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
            raise ValueError('Voxel size must be positive')
        self.voxel_size_m = float(voxel_size_m)
        if coordinate_unit not in ('m', 'sfm'):
            raise ValueError('Coordinate unit must be m or sfm')
        self.coordinate_unit = coordinate_unit
        self.id_to_label = validate_taxonomy(id_to_label)
        self._voxels: dict[tuple[int, int, int], dict] = {}

    def integrate(self, points: np.ndarray, labels: np.ndarray, confidence: np.ndarray) -> dict:
        if points.ndim != 2 or points.shape[1] != 3 or labels.shape != (len(points),) or confidence.shape != (len(points),):
            raise ValueError('Points, labels and confidence dimensions disagree')
        if not np.isfinite(points).all() or not np.isfinite(confidence).all():
            raise ValueError('Nonfinite semantic observations')
        if np.any((labels < 0) | (labels >= len(self.id_to_label))) or np.any((confidence <= 0) | (confidence > 1)):
            raise ValueError('Invalid class IDs or confidence')
        keys = np.floor(points / self.voxel_size_m).astype(np.int64)
        grouped = {}
        for key, xyz, label, weight in zip(keys, points, labels, confidence):
            k = tuple(int(x) for x in key)
            frame_vote = grouped.get(k)
            if frame_vote is None:
                frame_vote = {'point_sum': np.zeros(3), 'count': 0, 'votes': {}}
                grouped[k] = frame_vote
            frame_vote['point_sum'] += xyz
            frame_vote['count'] += 1
            cid = int(label)
            # Multiple pixels in one frame do not multiply independent evidence.
            frame_vote['votes'][cid] = max(frame_vote['votes'].get(cid, 0.), float(weight))
        matched = agreed = 0
        for k, frame_vote in grouped.items():
            voxel = self._voxels.get(k)
            current_winner = max(frame_vote['votes'], key=frame_vote['votes'].get)
            if voxel is None:
                voxel = {'point_sum': np.zeros(3), 'count': 0, 'views': 0, 'votes': {}}
                self._voxels[k] = voxel
            else:
                matched += 1
                agreed += int(max(voxel['votes'], key=voxel['votes'].get) == current_winner)
            voxel['point_sum'] += frame_vote['point_sum']
            voxel['count'] += frame_vote['count']
            voxel['views'] += 1
            total = sum(frame_vote['votes'].values())
            cap = max(frame_vote['votes'].values())
            for cid, weight in frame_vote['votes'].items():
                voxel['votes'][cid] = voxel['votes'].get(cid, 0.) + cap*weight/total
        return {'observations': len(points), 'frame_voxels': len(grouped),
                'previously_occupied_voxels': matched,
                'agreement_with_previous_winner': agreed/matched if matched else None}

    def query(self, point: np.ndarray) -> dict | None:
        p = np.asarray(point, dtype=np.float64)
        if p.shape != (3,) or not np.isfinite(p).all():
            raise ValueError('Query must be a finite 3D point')
        v = self._voxels.get(tuple(np.floor(p/self.voxel_size_m).astype(np.int64)))
        if v is None:
            return None
        total = sum(v['votes'].values())
        return {'label_id': max(v['votes'], key=v['votes'].get), 'distribution':
                {i: x/total for i, x in v['votes'].items()}, 'observations': v['count'],
                'frames_observed': v['views']}

    def extract(self, label_id: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if label_id is not None and label_id not in self.id_to_label:
            raise ValueError('Unknown class ID')
        rows = [(v['point_sum']/v['count'], max(v['votes'], key=v['votes'].get), v['count'])
                for v in self._voxels.values()]
        if label_id is not None:
            rows = [r for r in rows if r[1] == label_id]
        if not rows:
            return np.empty((0,3)), np.empty(0,dtype=np.int32), np.empty(0,dtype=np.int32)
        return np.stack([r[0] for r in rows]), np.array([r[1] for r in rows],np.int32), np.array([r[2] for r in rows],np.int32)

    def statistics(self) -> dict:
        counts = np.zeros(len(self.id_to_label), dtype=np.int64)
        purities = []
        entropies = []
        shared = 0
        for v in self._voxels.values():
            weights = np.array(list(v['votes'].values()))
            probs = weights/weights.sum()
            counts[max(v['votes'], key=v['votes'].get)] += 1
            if v['views'] >= 2:
                shared += 1
                purities.append(float(probs.max()))
                entropies.append(float(-(probs*np.log(probs)).sum()))
        return {'occupied_voxels': len(self._voxels), 'multi_observed_voxels': shared,
                'mean_shared_voxel_purity': float(np.mean(purities)) if purities else None,
                'mean_shared_voxel_entropy_nats': float(np.mean(entropies)) if entropies else None,
                'voxels_per_class': {str(i): int(n) for i,n in enumerate(counts) if n}}

    def export_ply(self, path: Path) -> None:
        points, ids, _ = self.extract()
        write_ply(path, points, palette(len(self.id_to_label))[ids])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        records = [{'key': list(k), 'point_sum': v['point_sum'].tolist(), 'count': v['count'],
                    'views': v['views'], 'votes': {str(c): w for c,w in v['votes'].items()}} for k,v in self._voxels.items()]
        path.write_text(json.dumps({'schema_version':1,'voxel_size_m':self.voxel_size_m,
            'coordinate_unit':self.coordinate_unit,
            'id_to_label':self.id_to_label,'voxels':records}), encoding='utf-8')

    @classmethod
    def load(cls, path: Path) -> 'SemanticVoxelMap':
        data = json.loads(path.read_text(encoding='utf-8'))
        if data.get('schema_version') != 1:
            raise ValueError('Unsupported semantic map schema')
        result = cls(data['voxel_size_m'], data['id_to_label'], data.get('coordinate_unit','m'))
        for row in data['voxels']:
            result._voxels[tuple(row['key'])] = {'point_sum': np.asarray(row['point_sum'],dtype=np.float64),
                'count': int(row['count']), 'views': int(row['views']),
                'votes': {int(k):float(v) for k,v in row['votes'].items()}}
        return result
