"""RGB-only Gaussian adapter. SfM reconstruction is transductive; no GT inputs."""
import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from .video_camera import rectify
from .geometry import validate_pose
from .gaussian_scene import render, load_dataset


def prepare(run, output, width=240):
    run, output = Path(run), Path(output)
    selected=run/'sfm/quality/filtered_sfm.json'
    sfm = json.loads((selected if selected.is_file() else run/'sfm/sfm.json').read_text())
    if sfm['depth_or_gt_pose_used']:
        raise ValueError('RGB-only adapter rejects depth/GT poses')
    frames = sfm['frames']
    if len(frames) < 8:
        raise ValueError('At least eight registered cameras required for disjoint splits')
    output.mkdir(parents=True, exist_ok=True)
    groups = {'train': [], 'validation': [], 'test': []}
    records, observations = [], {}
    for i, f in enumerate(frames):
        # Uniform held-out views, fixed before training, never selected by quality.
        group = 'test' if i % 8 == 7 else 'validation' if i % 8 == 3 else 'train'
        groups[group].append(i)
        pose = np.asarray(f['T_world_camera']); validate_pose(pose)
        cam = sfm['cameras'][str(f['camera_id'])]
        rgb = np.asarray(Image.open(run/'extracted/images'/f['image']).convert('RGB'))
        mask, image = rectify(cam, np.ones(rgb.shape[:2],np.float32), rgb)
        height = round(cam['height'] * width / cam['width'])
        valid = np.asarray(Image.fromarray((mask>0).astype(np.uint8)*255).resize((width,height),Image.Resampling.NEAREST))>0
        image = np.asarray(Image.fromarray(image).resize((width,height),Image.Resampling.LANCZOS)).copy()
        image[~valid]=128
        name=f'view_{i:04d}.png'; Image.fromarray(image).save(output/name)
        np.save(output/f'mask_{i:04d}.npy',valid)
        k=np.asarray(cam['K'],dtype=np.float64).copy(); k[0,:]*=width/cam['width'];k[1,:]*=height/cam['height']
        intr={'fx':float(k[0,0]),'fy':float(k[1,1]),'cx':float(k[0,2]),'cy':float(k[1,2])}
        records.append({'image':name,'mask':f'mask_{i:04d}.npy','split':group,'timestamp_s':f['timestamp_s'],
          'intrinsics':intr,'T_world_camera':pose.tolist(),'T_camera_world':np.linalg.inv(pose).tolist(),'source_image':f['image']})
        if group=='train':
            for obs in f['observations']:
                pid=str(obs['point3D_id']);p=sfm['landmarks'][pid]
                if p['error_px']>2 or p['track_length']<3:continue
                u,v=np.rint(obs['xy']).astype(int)
                if 0<=u<rgb.shape[1] and 0<=v<rgb.shape[0]:observations.setdefault(pid,[]).append(rgb[v,u]/255.)
    ids=[pid for pid,colors in observations.items() if len(colors)>=2]
    if len(ids)<25:raise ValueError('Insufficient training-observed sparse geometry')
    xyz=np.asarray([sfm['landmarks'][pid]['xyz'] for pid in ids],np.float32)
    colors=np.asarray([np.median(observations[pid],axis=0) for pid in ids],np.float32)
    distances=cKDTree(xyz).query(xyz,k=2)[0][:,1]
    unit=float(np.median(distances[distances>0]))
    np.savez(output/'initial.npz',xyz=xyz,rgb=colors,scales=np.clip(distances,unit*.2,unit*3))
    data={'frames':records,'splits':groups,'width':width,'height':height,'intrinsics':records[0]['intrinsics'],
      'scale_limits':[unit*.02,unit*8],'pose_source':sfm['pose_source'],'unit':'arbitrary SfM units',
      'depth_initialization':'none; RGB-only sparse triangulation with training-view colors',
      'evaluation_protocol':'Held-out RGB never in optimization or color initialization. SfM uses all images (transductive pose/geometry estimation).',
      'rectification':'SIMPLE_RADIAL to per-frame pinhole, invalid pixels masked', 'source_run':str(run.resolve())}
    (output/'dataset.json').write_text(json.dumps(data,indent=2))
    return data


def train(dataset_path, output, iterations=1200):
    import torch,gsplat
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    dataset_path,output=Path(dataset_path),Path(output);output.mkdir(parents=True,exist_ok=True)
    data=load_dataset(dataset_path);initial=np.load(dataset_path.parent/'initial.npz')
    xyz,rgb=initial['xyz'],initial['rgb'];n=len(xyz)
    torch.manual_seed(2026);rng=np.random.default_rng(2026)
    q=torch.zeros((n,4),device='cuda');q[:,0]=1
    params={'means':torch.nn.Parameter(torch.tensor(xyz,device='cuda')),
      'quats':torch.nn.Parameter(q),
      'log_scales':torch.nn.Parameter(torch.tensor(np.log(initial['scales'])[:,None].repeat(3,axis=1),device='cuda',dtype=torch.float32)),
      'opacity_logits':torch.nn.Parameter(torch.zeros(n,device='cuda')),
      'color_logits':torch.nn.Parameter(torch.logit(torch.tensor(rgb,device='cuda').clamp(.01,.99)))}
    unit=float(np.median(initial['scales']))
    rates={'means':unit*.02,'quats':.001,'log_scales':.005,'opacity_logits':.01,'color_logits':.015}
    optimizer=torch.optim.Adam([{'params':[v],'lr':rates[k]} for k,v in params.items()])
    targets={i:torch.tensor(np.asarray(Image.open(dataset_path.parent/data['frames'][i]['image']),np.float32)/255.,device='cuda') for i in data['splits']['train']}
    masks={i:torch.tensor(np.load(dataset_path.parent/data['frames'][i]['mask']),device='cuda') for i in targets}
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.perf_counter();history=[]
    for step in range(iterations):
        i=int(rng.choice(data['splits']['train']));optimizer.zero_grad(set_to_none=True)
        pred,_=render(params,data,data['frames'][i]);loss=(pred-targets[i]).abs()[masks[i]].mean()
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite Gaussian loss')
        loss.backward();optimizer.step();history.append({'iteration':step+1,'frame':i,'loss':loss.item()})
        if step%50==0:print(f'Gaussian {step+1}/{iterations} loss={loss.item():.6f}',flush=True)
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    torch.save({'parameters':{k:v.detach().cpu() for k,v in params.items()},'gaussians':n,'dataset':str(dataset_path.resolve()),
      'depth_initialization':data['depth_initialization'],'pose_source':data['pose_source'],'iterations':iterations},output/'scene.pt')
    with (output/'training_loss.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=history[0]);writer.writeheader();writer.writerows(history)
    report={'iterations':iterations,'gaussians':n,'elapsed_s':elapsed,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated(),
      'peak_gpu_reserved_bytes':torch.cuda.max_memory_reserved(),'initial_loss':history[0]['loss'],'final_loss':history[-1]['loss'],
      'torch':torch.__version__,'gsplat':gsplat.__version__,'cuda':torch.version.cuda,'gpu':torch.cuda.get_device_name(),
      'learning_rates':rates,'quaternions':'optimized','densification':False,'evaluation_protocol':data['evaluation_protocol'],
      'depth_initialization':data['depth_initialization'],'loss':'L1 on valid rectified training pixels','resolution':[data['width'],data['height']]}
    (output/'training.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('--iterations',type=int,default=1200);p.add_argument('--width',type=int,default=240)
    a=p.parse_args();out=a.run/'photoreal';prepare(a.run,out/'dataset',a.width);train(out/'dataset/dataset.json',out/'trained',a.iterations)
    from .neural_eval import evaluate
    evaluate(out/'dataset/dataset.json',out/'trained/scene.pt',out/'evaluation')
if __name__=='__main__':main()
