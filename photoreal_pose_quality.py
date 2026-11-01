"""Conservative SfM quality gate. Reprojection alone cannot prove scene geometry."""
import argparse,json
from pathlib import Path
import numpy as np
from .video_camera import project


def filter_sfm(path,output,minimum_observations=25,max_median_px=2.,min_shared=10):
    sfm=json.loads(Path(path).read_text());frames=sfm['frames'];per=[];owners={}
    for i,f in enumerate(frames):
        cam=sfm['cameras'][str(f['camera_id'])];inv=np.linalg.inv(np.asarray(f['T_world_camera']))
        obs=f['observations'];world=np.asarray([sfm['landmarks'][str(o['point3D_id'])]['xyz'] for o in obs],float) if obs else np.empty((0,3))
        camera=world@inv[:3,:3].T+inv[:3,3];positive=camera[:,2]>0
        residual=np.full(len(obs),np.inf)
        if np.any(positive):
            uv=project(cam,camera[positive]);xy=np.asarray([o['xy'] for o in obs])[positive]
            residual[positive]=np.linalg.norm(uv-xy,axis=1)
        med=float(np.median(residual[np.isfinite(residual)])) if np.isfinite(residual).any() else None
        okay=len(obs)>=minimum_observations and med is not None and med<=max_median_px and np.mean(positive)>.95
        for o in obs:owners.setdefault(o['point3D_id'],[]).append(i)
        per.append({'image':f['image'],'observations':len(obs),'median_reprojection_px':med,'positive_depth_fraction':float(np.mean(positive)) if len(obs) else 0.,'local_quality':bool(okay)})
    edges={}
    for ids in owners.values():
        for a in range(len(ids)):
            for b in range(a+1,len(ids)):
                if per[ids[a]]['local_quality'] and per[ids[b]]['local_quality']:
                    key=(ids[a],ids[b]);edges[key]=edges.get(key,0)+1
    parent=list(range(len(frames)))
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for (a,b),count in edges.items():
        if count>=min_shared:parent[find(a)]=find(b)
    groups={}
    for i,r in enumerate(per):
        if r['local_quality']:groups.setdefault(find(i),[]).append(i)
    selected=max(groups.values(),key=len) if groups else []
    if len(selected)<3:raise ValueError('Too few quality gated SfM cameras; inspect pose_quality.json')
    kept={frames[i]['image'] for i in selected}
    for i,r in enumerate(per):r['accepted']=i in selected;r['rejection_reason']=None if i in selected else ('insufficient observations or high residual' if not r['local_quality'] else 'disconnected camera graph')
    report={'input_cameras':len(frames),'accepted_cameras':len(selected),'rejected_cameras':len(frames)-len(selected),
      'minimum_observations':minimum_observations,'max_median_reprojection_px':max_median_px,'min_shared_landmarks_for_connection':min_shared,
      'components':[len(x) for x in groups.values()],'camera_reports':per,
      'warning':'Small reprojection residual cannot rule out wrong global geometry or scale drift.'}
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    (output/'pose_quality.json').write_text(json.dumps(report,indent=2))
    subset=dict(sfm,frames=[f for f in frames if f['image'] in kept],registered_frames=len(selected),
      unregistered_images=sfm['unregistered_images']+[f['image'] for f in frames if f['image'] not in kept],pose_quality_gate=str((output/'pose_quality.json').resolve()))
    (output/'filtered_sfm.json').write_text(json.dumps(subset,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='camera_reports'}),flush=True)
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('sfm',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();filter_sfm(a.sfm,a.output)
