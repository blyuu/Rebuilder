"""Read-only camera/depth gate retaining every extracted frame and failure."""
import argparse
import json
from pathlib import Path
import numpy as np
from .geometry import validate_pose
from .video_camera import project, validate_camera_model


def statistics(values):
    return {'count':len(values),'median':float(np.median(values)) if len(values) else None,
            'p90':float(np.quantile(values,.9)) if len(values) else None}


def audit(run:Path, output:Path):
    run=Path(run);output=Path(output)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    extraction=read(run/'extracted/extraction.json');sfm=read(run/'sfm/sfm.json')
    if sfm['depth_or_gt_pose_used']:raise ValueError('Expected RGB-only cameras')
    frames={f['image']:f for f in sfm['frames']}
    if len(frames)!=len(sfm['frames']):raise ValueError('Duplicate registered image')
    depth=read(run/'depth/depth_report.json') if (run/'depth/depth_report.json').is_file() else None
    fits={f['image']:f for f in depth['frames']} if depth else {}
    for camera in sfm['cameras'].values():validate_camera_model(camera)
    rows=[];errors=[];pinhole_errors=[];trajectory=[];last=None
    for source in extraction['frames']:
        name=Path(source['image']).name;frame=frames.get(name)
        row={'image':name,'timestamp_s':source['timestamp_s'],'registered':frame is not None,
             'depth_status':'not_run' if depth is None else 'unregistered'}
        if frame:
            if abs(frame['timestamp_s']-source['timestamp_s'])>1e-6 or frame['source_frame_index']!=source['source_frame_index']:
                raise ValueError('Image timestamp/source index association mismatch')
            pose=np.asarray(frame['T_world_camera']);validate_pose(pose)
            cam=sfm['cameras'][str(frame['camera_id'])];local=[];approx=[]
            for obs in frame['observations']:
                point=sfm['landmarks'].get(str(obs['point3D_id']))
                if point is None:continue
                xyz=(np.asarray(point['xyz'])-pose[:3,3])@pose[:3,:3]
                if not np.isfinite(xyz).all() or xyz[2]<=0:continue
                uv=project(cam,xyz);plain=project({'K':cam['K'],'model':'PINHOLE'},xyz)
                local.append(float(np.linalg.norm(uv-np.asarray(obs['xy']))))
                approx.append(float(np.linalg.norm(plain-np.asarray(obs['xy']))))
            errors.extend(local);pinhole_errors.extend(approx)
            row['reprojection_error_px']=statistics(local)
            fit=fits.get(name)
            row['depth_status']='not_run' if depth is None else 'missing_record'
            if fit:
                row.update(depth_status='aligned' if fit['fit']['success'] else 'alignment_failed',
                    alignment_reason=fit['fit']['reason'],valid_depth_fraction=fit['finite_aligned_fraction'],
                    sparse_correspondences=fit['fit']['reprojection_inliers'],
                    scale=fit['fit'].get('scale'),shift=fit['fit'].get('shift'))
            if last:
                previous=np.asarray(last['T_world_camera']);rotation=previous[:3,:3].T@pose[:3,:3]
                ids={o['point3D_id'] for o in frame['observations']}
                prev_ids={o['point3D_id'] for o in last['observations']}
                trajectory.append({'source':last['image'],'target':name,
                    'gap_s':frame['timestamp_s']-last['timestamp_s'],
                    'baseline_sfm':float(np.linalg.norm(pose[:3,3]-previous[:3,3])),
                    'rotation_deg':float(np.degrees(np.arccos(np.clip((np.trace(rotation)-1)/2,-1,1)))),
                    'shared_landmarks':len(ids&prev_ids)})
            last=frame
        rows.append(row)
    known={Path(f['image']).name for f in extraction['frames']}
    if set(frames)-known:raise ValueError('SfM references images outside extraction')
    result={'schema_version':1,'run':str(run.resolve()),'input_frames':len(rows),'registered_frames':len(frames),
        'registered_fraction':len(frames)/len(rows),'no_gt_inputs':True,
        'source_duration_s':extraction['video_metadata']['duration_s_nominal'],
        'extracted_time_range_s':[rows[0]['timestamp_s'],rows[-1]['timestamp_s']],
        'registered_time_range_s':([sfm['frames'][0]['timestamp_s'],sfm['frames'][-1]['timestamp_s']] if frames else None),
        'largest_registered_gap_s':max((p['gap_s'] for p in trajectory),default=None),
        'reprojection_camera_model_px':statistics(errors),'reprojection_ignoring_distortion_px':statistics(pinhole_errors),
        'depth_status_counts':{s:sum(r['depth_status']==s for r in rows) for s in sorted({r['depth_status'] for r in rows})},
        'trajectory':trajectory,'per_frame':rows,'camera_models':sfm['cameras'],
        'interpretation':'Registration, reprojection and temporal coverage are internal diagnostics; no room completeness or absolute accuracy claim.',
        'photoreal_assets':{'rgb_only_gaussian':'generated' if (run/'photoreal/trained/scene.pt').is_file() else 'not_generated',
            'textured_mesh':'generated' if (run/'photoreal/texture/textured.glb').is_file() else 'not_generated',
            'dollhouse':'GUI runtime validation required'}}
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.run,a.output)
    print(json.dumps({k:v for k,v in r.items() if k not in ('per_frame','trajectory','camera_models')},indent=2))
