"""Versioned project persistence. Opening a project never executes a pipeline."""
from __future__ import annotations
import json
import os
from pathlib import Path
from .app_runtime import VERSION

PATH_KEYS = {'manifest','mesh','cloud','slam_manifest','semantic_map','gaussian_checkpoint',
             'gaussian_dataset','textured_glb','video','video_output','output','inverse_mesh','inverse_output','intrinsics'}


def atomic_json(path: Path, value) -> None:
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def save_project(path: Path, data: dict) -> dict:
    path=Path(path).resolve();record=dict(data)
    record.update(schema_version=2,application_version=VERSION)
    record.setdefault('input_type','rgbd');record.setdefault('workflow','TSDF')
    record.setdefault('scale_convention','meters' if record['input_type']=='rgbd' else 'arbitrary_sfm')
    record.setdefault('pose_contract','T_world_camera; optical x right, y down, z forward')
    for key in PATH_KEYS:
        if record.get(key):
            source=Path(record[key]).resolve()
            try: record[key]=source.relative_to(path.parent).as_posix()
            except ValueError: record[key]=str(source)
    atomic_json(path,record)
    return record


def load_project(path: Path, legacy_root: Path | None=None) -> dict:
    path=Path(path).resolve()
    record=json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(record,dict) or record.get('schema_version') not in (1,2):
        raise ValueError('Unsupported or corrupted RECON-X project; expected schema 1 or 2')
    base=legacy_root if record['schema_version']==1 else path.parent
    if base is None: raise ValueError('Legacy session requires its original root directory')
    missing=[]
    for key in PATH_KEYS:
        if record.get(key):
            if not isinstance(record[key],str):raise ValueError(f'Invalid project path: {key}')
            p=Path(record[key]);p=(p if p.is_absolute() else base/p).resolve()
            record[key]=str(p)
            if key!='output' and not p.exists():missing.append({'field':key,'path':str(p)})
    if record.get('input_type','rgbd') not in ('rgbd','rgb_video','artifacts'):
        raise ValueError('Unsupported project input type')
    record['missing_sources']=missing
    return record


def explain_error(detail: str, stage: str | None=None) -> dict:
    lower=detail.lower()
    if 'insufficient sfm-aligned depth' in lower or 'no valid sfm-aligned depth' in lower:
        return {'message':'Depth alignment failed',
                'recovery':'Inspect the per-frame depth report and SfM camera registration. Capture a slower video with textured surfaces and camera translation; check calibration if available.',
                'details':detail,'stage':stage}
    if stage=='SfM':
        return {'message':'Camera reconstruction failed',
                'recovery':'Inspect SfM registration and camera intrinsics. Capture overlapping textured views with camera translation.',
                'details':detail,'stage':stage}
    if stage=='extract' and any(x in lower for x in ('decoder','codec','video','frame','mp4','mov')):
        return {'message':'Video could not be read',
                'recovery':'Check MP4/MOV codec, orientation and first-frame decode; use a compatible conversion if needed.',
                'details':detail,'stage':stage}
    pairs=[(('out of memory','cuda error: memory','insufficient resources'), 'Insufficient memory', 'Close other GPU applications; reduce resolution or frame count and retry.'),
           (('model','checkpoint','weights'), 'Model asset unavailable or invalid', 'Select the complete local model directory in Home / Runtime assets.'),
           (('registration','landmark','sparse'), 'Camera reconstruction failed', 'Use a sharper, textured video with overlap between views; inspect stage logs.'),
           (('unsupported codec','video decoder could not','could not decode'), 'Video could not be read', 'Use a valid MP4/MOV with a supported codec; check input metadata.'),
           (('calibration','intrinsics'), 'Invalid camera calibration', 'Check positive focal lengths and the calibration image resolution.'),
           (('missing','no such','not found'), 'Required file unavailable', 'Restore or select the file again; check project references.'),
           (('cuda','gpu'), 'GPU runtime unavailable', 'Check the NVIDIA driver and the selected runtime pack.'),
           (('schema','json','manifest'), 'Invalid project or manifest', 'Open a supported RECON-X JSON file or restore its backup.')]
    for tokens,title,action in pairs:
        if any(token in lower for token in tokens):return {'message':title,'recovery':action,'details':detail}
    return {'message':'Workflow could not complete','recovery':'Review the diagnostic log, correct the input or runtime configuration, and retry.','details':detail}
