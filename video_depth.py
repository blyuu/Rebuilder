"""RGB-only relative depth with sparse SfM alignment; depth stays in SfM units."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates

from .depth_models import adapter
from .phase6 import signal_from_raw
from .video_camera import project


def fit_sparse_depth(signal: np.ndarray, observations: list[dict], landmarks: dict,
                     T_world_camera: np.ndarray, min_points: int = 25,
                     K: np.ndarray | None = None, reprojection_limit_px: float = 2.,
                     camera_model: dict | None = None) -> dict:
    """Robust positive affine fit at triangulated feature pixels; no reference depth."""
    h, w = signal.shape
    camera_world = np.linalg.inv(T_world_camera)
    samples = [];reprojection_errors=[];projectable=0
    for obs in observations:
        point = landmarks.get(str(obs['point3D_id']))
        if point is None: continue
        u, v = obs['xy']
        if not (1 <= u < w-2 and 1 <= v < h-2): continue
        xyz = np.asarray(point['xyz'], dtype=np.float64)
        camera=camera_world[:3,:3]@xyz+camera_world[:3,3]
        z=float(camera[2])
        if not (np.isfinite(z) and z>0):continue
        projectable+=1
        if camera_model is not None or K is not None:
            if camera_model is not None:
                u_proj,v_proj=project(camera_model,camera)
            else:
                u_proj=float(K[0,0]*camera[0]/z+K[0,2])
                v_proj=float(K[1,1]*camera[1]/z+K[1,2])
            error=float(np.hypot(u_proj-u,v_proj-v))
            if not np.isfinite(error) or error>reprojection_limit_px:continue
            reprojection_errors.append(error)
        samples.append((u, v, z))
    diagnostics={'observations':len(observations),'projectable_count':projectable,
                 'reprojection_inliers':len(samples)}
    if len(samples) < min_points: return {'success': False, 'reason': 'insufficient_sparse_points', 'count': len(samples),**diagnostics}
    samples = np.asarray(samples)
    span_u=float(np.ptp(samples[:,0])/w);span_v=float(np.ptp(samples[:,1])/h)
    if span_u<.1 or span_v<.1:
        return {'success':False,'reason':'poor_spatial_distribution','count':len(samples),
                'span_u_fraction':span_u,'span_v_fraction':span_v,**diagnostics}
    x = map_coordinates(signal, [samples[:, 1], samples[:, 0]], order=1, mode='nearest')
    y = samples[:, 2]
    valid = np.isfinite(x+y) & (x >= 0) & (y > 0)
    x, y = x[valid], y[valid]
    if len(x) < min_points or np.std(x) < 1e-5: return {'success': False, 'reason': 'degenerate_signal', 'count': len(x),**diagnostics}
    A = np.column_stack((x, np.ones_like(x)))
    weights = np.ones_like(x)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    for _ in range(12):
        residual = y-(a*x+b)
        sigma = max(1.4826*np.median(np.abs(residual-np.median(residual))), .005*np.median(y))
        weights = np.minimum(1., 1.5*sigma/np.maximum(np.abs(residual), 1e-9))
        a, b = np.linalg.lstsq(A*np.sqrt(weights[:,None]), y*np.sqrt(weights), rcond=None)[0]
    residual = y-(a*x+b)
    success = bool(np.isfinite([a,b]).all() and a > 0 and np.median(a*x+b)>0)
    return {'success': success, 'reason': None if success else 'nonphysical_fit',
        'count': len(x), 'robust_inliers':int(np.count_nonzero(weights>=.5)),
        'scale': float(a), 'shift': float(b),**diagnostics,
        'span_u_fraction':span_u,'span_v_fraction':span_v,
        'median_sfm_reprojection_error_px':float(np.median(reprojection_errors)) if reprojection_errors else None,
        'median_absolute_sparse_residual_sfm': float(np.median(np.abs(residual))),
        'p90_absolute_sparse_residual_sfm': float(np.quantile(np.abs(residual), .9))}


def run(extraction: Path, sfm_path: Path, output: Path, model_name='depth_anything_v2_small') -> dict:
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; real GPU inference required')
    extraction, output = Path(extraction), Path(output)
    sfm = json.loads(Path(sfm_path).read_text(encoding='utf-8'))
    if sfm['depth_or_gt_pose_used']: raise ValueError('Expected independent RGB-only SfM')
    output.mkdir(parents=True,exist_ok=True)
    model_options={}
    if model_name=='marigold_v1_1':
        model_options={'processing_resolution':384,'steps':4}
        local_checkpoint=Path(__file__).resolve().parents[2]/'tools/marigold-depth-v1-1'
        if local_checkpoint.is_dir():model_options['checkpoint_path']=str(local_checkpoint)
    model = adapter(model_name,**model_options)
    rows=[]
    for index, frame in enumerate(sfm['frames']):
        image = Image.open(extraction/'images'/frame['image']).convert('RGB')
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize()
        started=time.perf_counter()
        raw=model.predict(image,index)
        torch.cuda.synchronize()
        elapsed=time.perf_counter()-started
        signal=signal_from_raw(model_name,raw)
        camera_model=sfm['cameras'][str(frame['camera_id'])]
        K=np.asarray(camera_model['K'])
        fit=fit_sparse_depth(signal,frame['observations'],sfm['landmarks'],
                             np.asarray(frame['T_world_camera']),K=K,camera_model=camera_model)
        base=output/Path(frame['image']).stem
        np.save(str(base)+'_raw.npy',raw.astype(np.float32))
        valid_pixels=0;valid_min=valid_max=None
        if fit['success']:
            aligned=np.asarray(fit['scale']*signal+fit['shift'],dtype=np.float32)
            aligned[~np.isfinite(aligned) | (aligned<=0)] = 0
            valid_pixels=int(np.count_nonzero(aligned));valid_min=float(np.min(aligned[aligned>0])) if valid_pixels else None
            valid_max=float(np.max(aligned)) if valid_pixels else None
            np.save(str(base)+'_aligned_sfm.npy',aligned)
            p2,p98=np.quantile(aligned[aligned>0],[.02,.98])
            preview=np.uint8(np.clip((aligned-p2)/max(p98-p2,1e-6),0,1)*255)
            Image.fromarray(preview).save(str(base)+'_depth_preview.png')
        rows.append({'image':frame['image'],'timestamp_s':frame['timestamp_s'],
            'source_frame_index':frame['source_frame_index'],'fit':fit,
            'valid_aligned_pixels':valid_pixels,'finite_aligned_fraction':valid_pixels/signal.size,
            'aligned_min_sfm':valid_min,'aligned_max_sfm':valid_max,
            'inference_s':elapsed,'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated())})
        print(f"depth {index+1}/{len(sfm['frames'])} fit={fit['success']} {elapsed:.3f}s",flush=True)
    report={'schema_version':1,'model':model_name,'checkpoint_revision':model.checkpoint_revision,
        'model_options':model_options,
        'raw_unit':'relative model output','aligned_unit':'arbitrary SfM scale, not meters',
        'alignment_source':'RGB-only triangulated COLMAP points; no reference depth or poses',
        'camera_projection':'COLMAP camera model including SIMPLE_RADIAL distortion',
        'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),
        'frames':rows,'successful_alignments':sum(row['fit']['success'] for row in rows),
        'registered_frames':sfm['registered_frames'],
        'total_sparse_observations':sum(r['fit']['observations'] for r in rows),
        'total_reprojection_inliers':sum(r['fit']['reprojection_inliers'] for r in rows),
        'total_robust_alignment_inliers':sum(r['fit'].get('robust_inliers',0) for r in rows),
        'total_inference_s':sum(row['inference_s'] for row in rows),
        'peak_gpu_allocated_bytes':max(row['peak_gpu_allocated_bytes'] for row in rows)}
    (output/'depth_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['successful_alignments']<3:
        reasons={reason:sum(r['fit']['reason']==reason for r in rows) for reason in
                 ('insufficient_sparse_points','poor_spatial_distribution','degenerate_signal','nonphysical_fit')}
        raise ValueError(f"Insufficient SfM-aligned depth: {report['successful_alignments']}/"
            f"{sfm['registered_frames']} registered frames usable (minimum 3); "
            f"{report['total_reprojection_inliers']} projected correspondences; reasons={reasons}. "
            f"Inspect depth/depth_report.json and SfM camera model.")
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('extraction',type=Path);p.add_argument('sfm',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='depth_anything_v2_small')
    a=p.parse_args();run(a.extraction,a.sfm,a.output,a.model)


if __name__=='__main__':main()
