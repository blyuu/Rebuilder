"""Post hoc TUM diagnostics. Reference poses/depth never enter RGB-only reconstruction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .tum import _rows, associate_nearest, quaternion_to_pose


def similarity_alignment(estimated: np.ndarray, reference: np.ndarray) -> tuple[float,np.ndarray,np.ndarray]:
    """Least-squares Sim(3) fit, needed because monocular SfM has arbitrary scale."""
    if estimated.shape != reference.shape or estimated.ndim != 2 or estimated.shape[1] != 3 or len(estimated)<3:
        raise ValueError('At least three corresponding 3D centers required')
    a=estimated-estimated.mean(0);b=reference-reference.mean(0)
    u,s,vt=np.linalg.svd(a.T@b)
    d=np.diag([1.,1.,np.linalg.det(u@vt)])
    R=vt.T@d@u.T
    scale=float(np.trace(np.diag(s)@d)/np.sum(a*a))
    t=reference.mean(0)-scale*R@estimated.mean(0)
    if scale<=0 or not np.isfinite(scale):raise ValueError('Nonpositive similarity scale')
    return scale,R,t


def evaluate(sfm_path:Path,source_path:Path,tum_sequence:Path,depth_dir:Path,output:Path,
             *,tolerance_s:float=.03)->dict:
    sfm=json.loads(Path(sfm_path).read_text(encoding='utf-8'))
    source=json.loads(Path(source_path).read_text(encoding='utf-8'))
    if source['depth_or_gt_pose_used_to_encode'] or sfm['depth_or_gt_pose_used']:
        raise ValueError('Source or SfM violated RGB-only provenance')
    stamps=source['original_rgb_timestamps_s']
    gt_rows=_rows(Path(tum_sequence)/'groundtruth.txt',8)
    timestamps=[v[0] for v in gt_rows]
    associated=associate_nearest([stamps[f['source_frame_index']] for f in sfm['frames']],timestamps,tolerance_s,unique=False)
    est=[];gt=[];pair_rows=[]
    for frame_index,gt_index,dt in associated:
        frame=sfm['frames'][frame_index]
        pose=quaternion_to_pose([float(v) for v in gt_rows[gt_index][1][:3]],
                                [float(v) for v in gt_rows[gt_index][1][3:]])
        est.append(np.asarray(frame['T_world_camera']))
        gt.append(pose)
        pair_rows.append({'image':frame['image'],'source_frame_index':frame['source_frame_index'],
                          'rgb_timestamp_s':stamps[frame['source_frame_index']],
                          'gt_timestamp_s':gt_rows[gt_index][0],'association_delta_s':dt})
    if len(est)<3:raise ValueError('Insufficient GT associations for post hoc evaluation')
    est=np.asarray(est);gt=np.asarray(gt)
    scale,R,t=similarity_alignment(est[:,:3,3],gt[:,:3,3])
    centers=(scale*(R@est[:,:3,3].T)).T+t
    errors=np.linalg.norm(centers-gt[:,:3,3],axis=1)
    rotations=[]
    for a,b in zip(est,gt):
        delta=R@a[:3,:3]@b[:3,:3].T
        rotations.append(float(np.degrees(np.arccos(np.clip((np.trace(delta)-1)/2,-1,1)))))
    depth_rows=[]
    depth_gt=_rows(Path(tum_sequence)/'depth.txt',2)
    depth_matches=associate_nearest([r['rgb_timestamp_s'] for r in pair_rows],
                                    [r[0] for r in depth_gt],tolerance_s,unique=False)
    for row_i,depth_i,_ in depth_matches:
        row=pair_rows[row_i]
        path=Path(depth_dir)/(Path(row['image']).stem+'_aligned_sfm.npy')
        if not path.is_file():continue
        predicted=np.load(path)*scale # Oracle pose scale: evaluation only.
        target_path=Path(tum_sequence)/depth_gt[depth_i][1][0]
        sensor=np.asarray(Image.open(target_path),dtype=np.float32)/5000.
        sensor=np.asarray(Image.fromarray(sensor).resize((predicted.shape[1],predicted.shape[0]),Image.Resampling.NEAREST))
        valid=np.isfinite(predicted)&np.isfinite(sensor)&(predicted>0)&(sensor>.1)&(sensor<5.)
        if not np.any(valid):continue
        p=predicted[valid];g=sensor[valid]
        depth_rows.append({'image':row['image'],'pixels':len(p),
            'absrel':float(np.mean(np.abs(p-g)/g)),'rmse_m':float(np.sqrt(np.mean((p-g)**2))),
            'delta1':float(np.mean(np.maximum(p/g,g/p)<1.25))})
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    result={'evaluation_only':True,'gt_used_for_reconstruction':False,
        'scale_warning':'Sim3 scale from GT trajectory is oracle evaluation scale; RGB-only pipeline is nonmetric',
        'input_frames':sfm['input_frames'],'registered_frames':sfm['registered_frames'],
        'evaluated_poses':len(est),'pose_association_tolerance_s':tolerance_s,
        'sim3_scale_m_per_sfm_unit':scale,'sim3_R':R.tolist(),'sim3_t_m':t.tolist(),
        'ate_sim3_rmse_m':float(np.sqrt(np.mean(errors**2))),
        'ate_sim3_median_m':float(np.median(errors)),'rotation_mean_deg':float(np.mean(rotations)),
        'depth_metric_note':'Post hoc GT pose scale applied; not a deployable metric-depth result',
        'depth_evaluated_frames':len(depth_rows),
        'depth_mean_frame_absrel':float(np.mean([r['absrel'] for r in depth_rows])) if depth_rows else None,
        'depth_mean_frame_rmse_m':float(np.mean([r['rmse_m'] for r in depth_rows])) if depth_rows else None,
        'depth_mean_frame_delta1':float(np.mean([r['delta1'] for r in depth_rows])) if depth_rows else None,
        'per_frame_pose':[{**row,'translation_error_m':float(e),'rotation_error_deg':float(r)}
            for row,e,r in zip(pair_rows,errors,rotations)],'per_frame_depth':depth_rows}
    (output/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('sfm',type=Path);p.add_argument('source',type=Path)
    p.add_argument('tum_sequence',type=Path);p.add_argument('depth',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();result=evaluate(a.sfm,a.source,a.tum_sequence,a.depth,a.output)
    print(json.dumps({k:v for k,v in result.items() if not k.startswith('per_frame')},indent=2))


if __name__=='__main__':main()
