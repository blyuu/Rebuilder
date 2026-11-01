"""Summarize camera connectivity and every preserved RGB-only artifact."""
import argparse,json
from pathlib import Path
import numpy as np
from .photoreal_audit import audit


def finalize(root):
    root=Path(root)
    def read(path):
        p=root/path
        return json.loads(p.read_text()) if p.is_file() else None
    extraction=read('extracted/extraction.json');sfm=read('sfm/sfm.json')
    if not extraction or not sfm:raise ValueError('Extraction and SfM required')
    frames=sfm['frames'];owners={}
    for i,f in enumerate(frames):
        for o in f['observations']:owners.setdefault(o['point3D_id'],[]).append(i)
    pairs={}
    for ids in owners.values():
        for a in range(len(ids)):
            for b in range(a+1,len(ids)):
                key=(ids[a],ids[b]);pairs[key]=pairs.get(key,0)+1
    strong={pair:value for pair,value in pairs.items() if value>=10}
    parent=list(range(len(frames)))
    def find(i):
        while parent[i]!=i:i=parent[i]
        return i
    for a,b in strong:parent[find(a)]=find(b)
    components={}
    for i in range(len(frames)):components.setdefault(find(i),[]).append(i)
    depth=read('depth/depth_report.json');dense=read('dense/dense_report.json');gaussian=read('photoreal/trained/training.json')
    quality=read('sfm/quality/pose_quality.json')
    eval=read('photoreal/evaluation/evaluation.json');texture=read('photoreal/texture/texture_report.json');semantic=read('semantic/semantic_report.json')
    times=[f['timestamp_s'] for f in frames];alltimes=[f['timestamp_s'] for f in extraction['frames']]
    perframe=[{'image':f['image'],'timestamp_s':f['timestamp_s'],'observations':len(f['observations']),
      'linked_cameras_10_shared':sum(a==i or b==i for a,b in strong)} for i,f in enumerate(frames)]
    report={'source':extraction['source_video'],'decoded_frames':extraction.get('frames_decoded'),
      'selected_frames':len(extraction['frames']),'registered_frames':len(frames),
      'registration_fraction':len(frames)/len(extraction['frames']),
      'selected_time_range_s':[min(alltimes),max(alltimes)],
      'registered_time_range_s':[min(times),max(times)] if times else None,
      'largest_registered_gap_s':max((b-a for a,b in zip(times,times[1:])),default=None),
      'camera_graph_10_shared':{'edges':len(strong),'components':[{'size':len(ids),'first_s':times[ids[0]],'last_s':times[ids[-1]]} for ids in components.values()]},
      'per_registered_camera':perframe,
      'sfm_reprojection':'COLMAP bundle adjusted model; see diagnostics audit for per-observation residuals',
      'depth_aligned_frames':depth.get('successful_alignments') if depth else None,
      'pose_quality':{k:v for k,v in quality.items() if k!='camera_reports'} if quality else None,
      'tsdf':{'frames':dense['frames_integrated'],'triangles':dense['triangles'],'support_proxy':dense['input_observation_support_proxy']} if dense else None,
      'gaussian':gaussian,'heldout':eval['aggregate'] if eval else None,'texture':texture,
      'semantic_voxels':semantic.get('occupied_voxels') if semantic else None,
      'GT_depth_or_pose_used':False,'unit':'arbitrary SfM scale','complete_room_verified':False,
      'quality_note':'Registration, partial dense mesh, and appearance metrics do not establish complete or accurate room geometry.'}
    (root/'photoreal_summary.json').write_text(json.dumps(report,indent=2))
    if (root/'depth').is_dir():
        try:audit(root,root/'photoreal/audit.json')
        except Exception as exc:(root/'photoreal/audit_error.txt').write_text(str(exc))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();print(json.dumps(finalize(a.run),indent=2))
