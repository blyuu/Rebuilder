"""Adjacent-view depth reprojection diagnostics; does not alter aligned predictions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from .video_camera import backproject,project


def pair_diagnostic(ref:np.ndarray,other:np.ndarray,K_ref:np.ndarray,K_other:np.ndarray,
                    T_world_ref:np.ndarray,T_world_other:np.ndarray,
                    *,stride:int=4,occlusion_relative:float=.05,
                    camera_ref:dict|None=None,camera_other:dict|None=None)->dict:
    h,w=ref.shape
    if other.ndim!=2 or stride<1 or occlusion_relative<0:raise ValueError('Invalid diagnostic input')
    v,u=np.mgrid[0:h:stride,0:w:stride]
    z=ref[::stride,::stride]
    valid=np.isfinite(z)&(z>0)
    sample_count=int(valid.sum())
    if not sample_count:return {'sampled_valid_pixels':0,'reprojected_in_bounds':0,'depth_pairs':0,
        'coverage':0.,'occlusion_rejection_rate':None,'median_relative_residual':None}
    u,v,z=u[valid],v[valid],z[valid]
    xyz=(backproject(camera_ref,u,v,z) if camera_ref is not None else
         np.column_stack(((u-K_ref[0,2])*z/K_ref[0,0],(v-K_ref[1,2])*z/K_ref[1,1],z)))
    world=xyz@T_world_ref[:3,:3].T+T_world_ref[:3,3]
    other_camera=world@T_world_other[:3,:3]+(-T_world_other[:3,:3].T@T_world_other[:3,3])
    zz=other_camera[:,2]
    with np.errstate(divide='ignore',invalid='ignore'):
        if camera_other is not None:
            projected=project(camera_other,other_camera)
            uu,vv=projected[:,0],projected[:,1]
        else:
            uu=K_other[0,0]*other_camera[:,0]/zz+K_other[0,2]
            vv=K_other[1,1]*other_camera[:,1]/zz+K_other[1,2]
    in_bounds=np.isfinite(uu+vv+zz)&(zz>0)&(uu>=0)&(vv>=0)&(uu<other.shape[1]-1)&(vv<other.shape[0]-1)
    if not np.any(in_bounds):return {'sampled_valid_pixels':sample_count,'reprojected_in_bounds':0,
        'depth_pairs':0,'coverage':0.,'occlusion_rejection_rate':None,'median_relative_residual':None}
    measured=map_coordinates(other,[vv[in_bounds],uu[in_bounds]],order=1,mode='nearest')
    projected=zz[in_bounds]
    good=np.isfinite(measured)&(measured>0)
    residual=np.abs(measured[good]-projected[good])/np.maximum(measured[good],1e-6)
    occluded=projected[good]>measured[good]*(1+occlusion_relative)
    accepted=residual[~occluded]
    baseline=float(np.linalg.norm(T_world_ref[:3,3]-T_world_other[:3,3]))
    relative=T_world_other[:3,:3].T@T_world_ref[:3,:3]
    rotation_deg=float(np.degrees(np.arccos(np.clip((np.trace(relative)-1)/2,-1,1))))
    return {'sampled_valid_pixels':sample_count,'reprojected_in_bounds':int(in_bounds.sum()),
        'depth_pairs':int(good.sum()),'coverage':float(good.sum()/sample_count),
        'occlusion_rejection_rate':float(np.mean(occluded)) if len(occluded) else None,
        'median_relative_residual':float(np.median(accepted)) if len(accepted) else None,
        'p90_relative_residual':float(np.quantile(accepted,.9)) if len(accepted) else None,
        'baseline_sfm':baseline,'relative_rotation_deg':rotation_deg}


def run(sfm_path:Path,depth_dir:Path,output:Path,*,stride:int=4)->dict:
    sfm=json.loads(Path(sfm_path).read_text(encoding='utf-8'))
    rows=[]
    frames=sfm['frames']
    for i,(first,second) in enumerate(zip(frames,frames[1:])):
        path_a=Path(depth_dir)/(Path(first['image']).stem+'_aligned_sfm.npy')
        path_b=Path(depth_dir)/(Path(second['image']).stem+'_aligned_sfm.npy')
        if not path_a.is_file() or not path_b.is_file():continue
        a=np.load(path_a);b=np.load(path_b)
        ka=np.asarray(sfm['cameras'][str(first['camera_id'])]['K'])
        kb=np.asarray(sfm['cameras'][str(second['camera_id'])]['K'])
        result=pair_diagnostic(a,b,ka,kb,np.asarray(first['T_world_camera']),
                               np.asarray(second['T_world_camera']),stride=stride,
                               camera_ref=sfm['cameras'][str(first['camera_id'])],
                               camera_other=sfm['cameras'][str(second['camera_id'])])
        rows.append({'source_image':first['image'],'target_image':second['image'],**result})
    valid=[r for r in rows if r['median_relative_residual'] is not None]
    report={'pose_source':'RGB-only SfM','depth_source':'sparse-aligned monocular',
        'unit':'arbitrary SfM scale','pair_count':len(rows),'valid_pairs':len(valid),
        'mean_coverage':float(np.mean([r['coverage'] for r in rows])) if rows else None,
        'median_pair_relative_residual':float(np.median([r['median_relative_residual'] for r in valid])) if valid else None,
        'mean_occlusion_rejection_rate':float(np.mean([r['occlusion_rejection_rate'] for r in valid])) if valid else None,
        'note':'Consistency diagnostic only; no depth refinement or independent accuracy claim',
        'pairs':rows}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'multiview.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('sfm',type=Path);p.add_argument('depth',type=Path)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    print(json.dumps({k:v for k,v in run(a.sfm,a.depth,a.output).items() if k!='pairs'},indent=2))


if __name__=='__main__':main()
