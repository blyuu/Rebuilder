"""Persistent CUDA renderer controlled by JSON-lines; camera requests contain no GT."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
from PIL import Image
from .gaussian_scene import load_checkpoint,load_dataset,render
from .geometry import validate_pose


def serve(checkpoint,dataset,output):
    import torch
    data=load_dataset(Path(dataset));params=load_checkpoint(Path(checkpoint));output=Path(output);output.mkdir(parents=True,exist_ok=True)
    print(json.dumps({'ready':True}),flush=True)
    for line in sys.stdin:
        try:
            request=json.loads(line);pose=np.asarray(request['T_world_camera']);validate_pose(pose)
            width,height=int(request['width']),int(request['height'])
            if not (32<=width<=1920 and 32<=height<=1920):raise ValueError('Render resolution outside limits')
            local=dict(data,width=width,height=height);frame={'T_camera_world':np.linalg.inv(pose).tolist(),'intrinsics':request['intrinsics']}
            torch.cuda.synchronize();start=time.perf_counter()
            with torch.no_grad():rgb,alpha=render(params,local,frame)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            path=output/'free_view.png';temp=output/'free_view.tmp.png'
            Image.fromarray((rgb.clamp(0,1).cpu().numpy()*255).round().astype(np.uint8)).save(temp);temp.replace(path)
            result={'image':str(path.resolve()),'render_s':elapsed,'request':request,'mean_alpha':float(alpha.mean())}
            (output/'free_view.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
        except Exception as exc:print(json.dumps({'error':str(exc)}),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('checkpoint');p.add_argument('dataset');p.add_argument('output');a=p.parse_args();serve(a.checkpoint,a.dataset,a.output)
