"""Actual RGB SegFormer inference and SfM-coordinate semantic voxel fusion."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .semantic_map import SemanticVoxelMap, palette, validate_taxonomy
from .video_camera import backproject

MODEL='nvidia/segformer-b0-finetuned-ade-512-512'
REVISION='489d5cd81a0b59fab9b7ea758d3548ebe99677da'


def fuse_semantic_pixels(depth: np.ndarray, labels: np.ndarray, scores: np.ndarray,
                         K: np.ndarray, pose: np.ndarray, *, stride: int = 6,
                         max_depth: float = 35., threshold: float = .30,
                         camera_model:dict|None=None):
    if depth.shape != labels.shape or depth.shape != scores.shape:
        raise ValueError('Depth, labels and score resolution must match')
    v,u=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride]
    z=depth[::stride,::stride];l=labels[::stride,::stride];c=scores[::stride,::stride]
    valid=np.isfinite(z)&(z>0)&(z<max_depth)&(l>=0)&np.isfinite(c)&(c>=threshold)
    if not np.any(valid):return np.empty((0,3)),np.empty(0,np.int32),np.empty(0,np.float32),0
    xyz=(backproject(camera_model,u[valid],v[valid],z[valid]) if camera_model is not None else
         np.column_stack(((u[valid]-K[0,2])*z[valid]/K[0,0],
                          (v[valid]-K[1,2])*z[valid]/K[1,1],z[valid])))
    points=xyz@pose[:3,:3].T+pose[:3,3]
    return points,l[valid].astype(np.int32),c[valid].astype(np.float32),int(valid.sum())


def run(extraction: Path,sfm_path: Path,depth_dir: Path,output: Path,
        *,voxel_size_sfm: float=.4,stride:int=6) -> dict:
    import torch
    import torch.nn.functional as F
    import transformers
    from transformers import AutoImageProcessor, SegformerForSemanticSegmentation
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    extraction,depth_dir,output=Path(extraction),Path(depth_dir),Path(output)
    sfm=json.loads(Path(sfm_path).read_text(encoding='utf-8'))
    output.mkdir(parents=True,exist_ok=True)
    from .app_runtime import model_path
    checkpoint=model_path('segformer',MODEL)
    processor=AutoImageProcessor.from_pretrained(checkpoint,revision=REVISION)
    model=SegformerForSemanticSegmentation.from_pretrained(checkpoint,revision=REVISION).to('cuda').eval()
    taxonomy=validate_taxonomy(model.config.id2label)
    semantic=SemanticVoxelMap(voxel_size_sfm,taxonomy,coordinate_unit='sfm')
    rows=[]
    for i,frame in enumerate(sfm['frames']):
        depth_path=depth_dir/(Path(frame['image']).stem+'_aligned_sfm.npy')
        if not depth_path.is_file():continue
        depth=np.load(depth_path)
        rgb=Image.open(extraction/'images'/frame['image']).convert('RGB')
        inputs={k:v.to('cuda') for k,v in processor(images=rgb,return_tensors='pt').items()}
        torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter()
        with torch.inference_mode():
            logits=model(**inputs).logits
            probs=F.interpolate(logits,size=depth.shape,mode='bilinear',align_corners=False).softmax(dim=1)[0]
            confidence,labels=probs.max(dim=0)
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        labels=labels.cpu().numpy().astype(np.int16)
        confidence=confidence.cpu().numpy().astype(np.float32)
        labels[confidence<.30]=-1
        np.savez_compressed(output/(Path(frame['image']).stem+'_semantic.npz'),labels=labels,confidence=confidence)
        color=palette(len(taxonomy))[np.maximum(labels,0)]
        color[labels<0]=[35,35,35]
        Image.fromarray(np.uint8(.5*np.asarray(rgb)+.5*color)).save(output/(Path(frame['image']).stem+'_overlay.png'))
        camera_model=sfm['cameras'][str(frame['camera_id'])]
        K=np.asarray(camera_model['K'])
        xyz,ids,scores,count=fuse_semantic_pixels(depth,labels,confidence,K,np.asarray(frame['T_world_camera']),
                                                  stride=stride,camera_model=camera_model)
        stat=semantic.integrate(xyz,ids,scores)
        rows.append({'image':frame['image'],'timestamp_s':frame['timestamp_s'],
            'inference_s':elapsed,'peak_gpu_allocated_bytes':int(torch.cuda.max_memory_allocated()),
            'valid_semantic_depth_samples':count,**stat})
        print(f'semantic {i+1}/{len(sfm["frames"])} samples={count}',flush=True)
    semantic.save(output/'semantic_map.json')
    semantic.export_ply(output/'semantic_points.ply')
    loaded=SemanticVoxelMap.load(output/'semantic_map.json')
    points,ids,counts=loaded.extract()
    report={'model':MODEL,'revision':REVISION,'torch':torch.__version__,
        'transformers':transformers.__version__,'gpu':torch.cuda.get_device_name(0),
        'geometry_source':'RGB-only SfM + RGB-only relative depth sparse aligned to SfM',
        'coordinate_unit':'arbitrary SfM units; not meters','voxel_size_sfm':voxel_size_sfm,
        'pixel_stride':stride,'confidence_threshold':.30,'frames':len(rows),
        'occupied_voxels':len(points),'statistics':loaded.statistics(),
        'taxonomy':taxonomy,'per_frame':rows,
        'peak_gpu_allocated_bytes':max(row['peak_gpu_allocated_bytes'] for row in rows),
        'total_inference_s':sum(row['inference_s'] for row in rows)}
    (output/'semantic_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('extraction',type=Path);p.add_argument('sfm',type=Path)
    p.add_argument('depth',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();run(a.extraction,a.sfm,a.depth,a.output)


if __name__=='__main__':main()
