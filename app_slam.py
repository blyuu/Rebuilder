"""Prepare manifest RGB/depth input without passing GT poses to RTAB-Map."""
import argparse
import shutil
from pathlib import Path
from .io import load_scene

def main():
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('output',type=Path);p.add_argument('--max-frames',type=int,default=180);a=p.parse_args()
    scene=load_scene(a.manifest)
    for sub in ('rgb_sync','depth_sync'):(a.output/sub).mkdir(parents=True,exist_ok=True)
    for frame in scene.frames[:a.max_frames]:
        for sub,source in [('rgb_sync',frame.rgb_path),('depth_sync',frame.depth_path)]:
            shutil.copy2(source,a.output/sub/f'{frame.timestamp_s:.6f}.png')
    if (a.output/'groundtruth.txt').exists():raise ValueError('GT must never enter SLAM input')
    print('Prepared RGB-D only; no GT pose input',flush=True)

if __name__=='__main__':main()
