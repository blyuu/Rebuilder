"""Portable artifact discovery and validation for the Windows desktop app."""
from __future__ import annotations

import json
from functools import lru_cache
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import validate_pose
from .io import Scene, load_rgbd, load_scene
from .semantic_map import SemanticVoxelMap, palette


@dataclass(frozen=True)
class ProjectPaths:
    root: Path
    manifest: Path
    slam_manifest: Path | None
    mesh: Path | None
    cloud: Path | None
    semantic_map: Path | None
    gaussian_checkpoint: Path | None
    gaussian_dataset: Path | None


def relative_or_absolute(path: Path | None, root: Path) -> str | None:
    if path is None:return None
    path=path.resolve();root=root.resolve()
    try:return str(path.relative_to(root))
    except ValueError:return str(path)


def discover(root: Path, manifest: Path | None = None) -> ProjectPaths:
    root=root.resolve()
    manifest=(manifest or root/'outputs/tum_fr3_structure_notexture_far/scene.json').resolve()
    scene=load_scene(manifest)
    if not scene.frames:raise ValueError('Dataset has no frames')
    metadata=json.loads(manifest.read_text(encoding='utf-8')).get('source',{})
    source_sequence=str(metadata.get('sequence',manifest.parent.name))
    if 'structure_notexture_far' in source_sequence:
        sequence='fr3_structure_notexture_far'
    elif 'structure_texture_far' in source_sequence:
        sequence='fr3_structure_texture_far'
    else:sequence=source_sequence
    known=sequence in ('fr3_structure_texture_far','fr3_structure_notexture_far')
    canonical=root/('outputs/tum_fr3_structure_texture_far/scene.json' if sequence=='fr3_structure_texture_far' else 'outputs/tum_fr3_structure_notexture_far/scene.json')
    if known and canonical.is_file():
        reference=load_scene(canonical)
        known=len(reference.frames)==len(scene.frames) and all(
            abs(a.timestamp_s-b.timestamp_s)<.002 for a,b in zip(reference.frames,scene.frames))
    else:known=False
    def existing(path):return path.resolve() if path.is_file() else None
    slam=existing(root/'outputs/slam'/sequence/'comparison/slam_scene.json') if known else None
    if slam:
        estimated=load_scene(slam)
        if len(estimated.frames)!=len(scene.frames) or any(abs(a.timestamp_s-b.timestamp_s)>.002 for a,b in zip(scene.frames,estimated.frames)):
            slam=None
    is_texture=sequence=='fr3_structure_texture_far'
    phase='phase6_texture' if is_texture else 'phase6'
    mesh=existing(root/'outputs/tum_fr3_structure_notexture_far/tsdf/mesh.ply') if known and not is_texture else None
    if is_texture:
        mesh=existing(root/'outputs/slam/fr3_structure_texture_far/comparison/gt_tsdf/mesh.ply')
        if mesh is None:mesh=existing(root/'outputs'/phase/'depth_anything_v2_small/meshes/gt_mesh.ply')
    cloud=existing(root/'outputs/tum_fr3_structure_notexture_far/tsdf/cloud.ply') if known and not is_texture else None
    semantic=existing(root/'outputs/phase7/maps'/('texture_far' if is_texture else 'notexture_far')/'sensor_gt_pose/semantic_map.json') if known else None
    checkpoint=existing(root/'outputs/phase5/gt_pose_scene/scene.pt') if known and not is_texture else None
    dataset=existing(root/'outputs/phase5/dataset/dataset.json') if known and not is_texture else None
    return ProjectPaths(root,manifest,slam,mesh,cloud,semantic,checkpoint,dataset)


def scene_summary(scene: Scene) -> dict:
    k=scene.intrinsics
    return {'frames':len(scene.frames),'resolution':[k.width,k.height],
            'fx':k.fx,'fy':k.fy,'cx':k.cx,'cy':k.cy,
            'depth_scale':scene.depth_scale,'first_timestamp_s':scene.frames[0].timestamp_s,
            'last_timestamp_s':scene.frames[-1].timestamp_s,'pose_contract':'T_world_camera',
            'metric_units':'meters'}


def preview_frame(scene: Scene, index: int) -> tuple[np.ndarray,np.ndarray]:
    if not 0<=index<len(scene.frames):raise IndexError(index)
    return load_rgbd(scene.frames[index],scene)


def trajectory(manifest: Path) -> np.ndarray:
    scene=load_scene(manifest)
    for frame in scene.frames:validate_pose(frame.t_world_camera)
    return np.stack([frame.t_world_camera[:3,3] for frame in scene.frames])


def camera_frustums(scene: Scene, depth_m: float=.16, stride: int=2) -> tuple[np.ndarray,np.ndarray]:
    """Optical camera frustum segments in world coordinates; +z points forward."""
    if depth_m<=0 or stride<1:raise ValueError('Positive depth and stride required')
    k=scene.intrinsics
    rays=np.array([[0,0,0],
        [(-k.cx)/k.fx*depth_m,(-k.cy)/k.fy*depth_m,depth_m],
        [(k.width-1-k.cx)/k.fx*depth_m,(-k.cy)/k.fy*depth_m,depth_m],
        [(k.width-1-k.cx)/k.fx*depth_m,(k.height-1-k.cy)/k.fy*depth_m,depth_m],
        [(-k.cx)/k.fx*depth_m,(k.height-1-k.cy)/k.fy*depth_m,depth_m]])
    vertices=[];lines=[]
    for frame in scene.frames[::stride]:
        offset=len(vertices)
        t=frame.t_world_camera
        vertices.extend((rays@t[:3,:3].T+t[:3,3]).tolist())
        for a,b in ((0,1),(0,2),(0,3),(0,4),(1,2),(2,3),(3,4),(4,1)):
            lines.extend((2,offset+a,offset+b))
    return np.asarray(vertices,dtype=np.float64),np.asarray(lines,dtype=np.int64)


@lru_cache(maxsize=2)
def _cached_semantic(path: str, mtime_ns: int, size: int):
    semantic=SemanticVoxelMap.load(Path(path))
    return semantic,semantic.statistics()


def cached_semantic(path: Path):
    """Read-only GUI map cache; invalidate on overwrite, bounded to two maps."""
    path=Path(path).resolve();stat=path.stat()
    return _cached_semantic(str(path),stat.st_mtime_ns,stat.st_size)[0]


def semantic_geometry(path: Path, class_id: int | None=None) -> tuple[np.ndarray,np.ndarray,dict]:
    semantic=cached_semantic(path)
    points,labels,_=semantic.extract(class_id)
    colors=palette(len(semantic.id_to_label))[labels]
    stat=path.stat()
    return points,colors,_cached_semantic(str(path.resolve()),stat.st_mtime_ns,stat.st_size)[1]


def semantic_query(path: Path, xyz: np.ndarray) -> dict | None:
    return cached_semantic(path).query(np.asarray(xyz,dtype=np.float64))


def metric_files(root: Path) -> list[tuple[str,Path]]:
    candidates={
        'TSDF':root/'outputs/tum_fr3_structure_notexture_far/tsdf/run.json',
        'SLAM':root/'outputs/slam/fr3_structure_notexture_far/comparison/trajectory_evaluation.json',
        'Depth':root/'outputs/depth_phase4/evaluation/depth_metrics.json',
        'Gaussian':root/'outputs/phase5/gt_pose_scene/rendering/evaluation.json',
        'Semantic':root/'outputs/phase7/results.json',
        'Phase 6.1':root/'outputs/phase6_1/analysis.json',
        'GUI TSDF':root/'outputs/phase9/smoke_tsdf/run.json',
        'Semantic GT':root/'outputs/phase7/maps/notexture_far/sensor_gt_pose/metrics.json',
        'Semantic RTAB':root/'outputs/phase7/maps/notexture_far/sensor_rtab_pose/metrics.json',
        'Pipeline history':root/'outputs/phase9/history.json',
        'Inverse GT':root/'outputs/phase10/tum_gt/results.json',
        'Inverse RTAB':root/'outputs/phase10/tum_rtab/results.json',
    }
    return list(candidates.items())


def read_metrics(path: Path):
    if not path.is_file():raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding='utf-8'))


def metric_summary(name: str, data) -> str:
    """Compact dashboard values taken directly from result JSON."""
    if not isinstance(data,dict):return f'{len(data)} records' if isinstance(data,list) else str(data)
    def value(number):
        return 'missing' if number is None else f'{number:.4g}' if isinstance(number,(float,int)) else str(number)
    if name in ('TSDF','GUI TSDF'):
        return f"{value(data.get('frames'))} f · {value(data.get('mesh_triangles'))} tri · {value(data.get('elapsed_s'))} s"
    if name=='SLAM':
        ate=data.get('ate_after_rigid_alignment',{})
        return f"ATE {value(ate.get('translation_rmse_m'))} m · tracked {value(data.get('tracking_success_rate'))}"
    if name=='Depth':
        robust=data.get('aggregate',{}).get('depth_anything_v2_small',{}).get('robust',{})
        return f"DA oracle AbsRel {value(robust.get('absrel'))}"
    if name=='Gaussian':
        test=data.get('aggregate',{}).get('test',{})
        return f"test PSNR {value(test.get('psnr_db'))} · SSIM {value(test.get('ssim'))}"
    if name.startswith('Semantic'):
        return f"{value(data.get('voxel_statistics',{}).get('occupied_voxels'))} vox · consistency {value(data.get('cross_view_previous_winner_agreement'))}"
    if name.startswith('Inverse'):
        rows=data.get('metrics',{}).get('test',[])
        if rows:
            psnr=np.mean([r['methods']['lambertian_regularized']['psnr_db'] for r in rows])
            return f"test PSNR {value(psnr)} dB · {value(len(rows))} views"
    return f'{len(data)} fields'


def slice_manifest(source: Path, destination: Path, start: int, stop: int) -> Path:
    """Write a portable, provenance-labelled [start, stop) frame subset."""
    data=read_metrics(source)
    frames=data.get('frames',[])
    if not 0<=start<stop<=len(frames):
        raise ValueError(f'Expected 0 <= start < stop <= {len(frames)}')
    selected=[]
    for frame in frames[start:stop]:
        record=dict(frame)
        for key in ('rgb','depth'):
            record[key]=str((source.parent/record[key]).resolve())
        selected.append(record)
    data['frames']=selected
    data['desktop_subset']={'source_manifest':str(source.resolve()),'start':start,'stop_exclusive':stop}
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(data,indent=2),encoding='utf-8')
    load_scene(destination)
    return destination


def save_session(path: Path, data: dict, root: Path) -> None:
    serialized=dict(data)
    for key in ('manifest','mesh','cloud','slam_manifest','semantic_map','gaussian_checkpoint','gaussian_dataset','output'):
        if serialized.get(key):serialized[key]=relative_or_absolute(Path(serialized[key]),root)
    serialized['schema_version']=1
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(serialized,indent=2),encoding='utf-8')


def load_session(path: Path, root: Path) -> dict:
    data=read_metrics(path)
    if data.get('schema_version')!=1:raise ValueError('Unsupported desktop session schema')
    for key in ('manifest','mesh','cloud','slam_manifest','semantic_map','gaussian_checkpoint','gaussian_dataset','output'):
        if data.get(key):
            p=Path(data[key]);data[key]=str((p if p.is_absolute() else root/p).resolve())
    return data
