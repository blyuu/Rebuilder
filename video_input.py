"""Video metadata and selective RGB extraction; never touches reference depth or pose."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np


def inspect_video(path: Path) -> dict:
    path=Path(path)
    if not path.is_file() or path.suffix.lower() not in ('.mp4','.mov'):
        raise ValueError('Expected an existing MP4 or MOV file')
    stream=cv2.VideoCapture(str(path))
    try:
        if not stream.isOpened():raise ValueError('Video decoder could not open file')
        fps=float(stream.get(cv2.CAP_PROP_FPS));count=int(stream.get(cv2.CAP_PROP_FRAME_COUNT))
        width=int(stream.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not 0<fps<1000 or not 0<count or width<16 or height<16:
            raise ValueError('Video has invalid FPS, count or resolution')
        decoded,frame=stream.read()
        if not decoded or frame is None:raise ValueError('Video decoder could not decode the first frame')
        return {'decoder_verified':True,'source':str(path.resolve()),'fps':fps,'reported_frames':count,
            'width':width,'height':height,'duration_s_nominal':count/fps,
            'orientation_note':'Decoder output orientation is used; rotation metadata must be checked for each device',
            'stabilization_note':'Digital stabilization and changing intrinsics are not corrected'}
    finally:stream.release()


def scaled_intrinsics(width:int,height:int,target_width:int,target_height:int,
                      known:dict|None=None) -> dict:
    if min(width,height,target_width,target_height)<1:raise ValueError('Invalid image dimensions')
    if known:
        fx,fy,cx,cy=(float(known[k]) for k in ('fx','fy','cx','cy'))
        source='explicit_or_known_calibration'
        if fx<=0 or fy<=0 or not np.isfinite([fx,fy,cx,cy]).all():raise ValueError('Invalid known intrinsics')
    else:
        focal=1.2*max(width,height)
        fx=fy=focal;cx=(width-1)/2;cy=(height-1)/2
        source='heuristic_1.2_times_max_dimension; uncertain; COLMAP may refine'
    sx=target_width/width;sy=target_height/height
    return {'width':target_width,'height':target_height,'fx':fx*sx,'fy':fy*sy,
            'cx':(cx+.5)*sx-.5,'cy':(cy+.5)*sy-.5,'source':source,
            'distortion':'uncalibrated unless supplied; no cropping applied'}


def extract_video(path:Path,output:Path,*,stride:int=3,max_frames:int=60,
                  target_width:int=480,blur_threshold:float=15,
                  duplicate_mae:float=1.5,known_intrinsics:dict|None=None,
                  sampling:str='sequential')->dict:
    if stride<1 or max_frames<2 or target_width<16 or blur_threshold<0 or duplicate_mae<0:
        raise ValueError('Invalid extraction settings')
    start=time.perf_counter();meta=inspect_video(path)
    if sampling not in ('sequential','uniform'):raise ValueError('Unsupported frame sampling mode')
    effective_stride=(max(stride,int(np.ceil(meta['reported_frames']/max_frames)))
                      if sampling=='uniform' else stride)
    width=target_width;height=round(meta['height']*width/meta['width'])
    if height%2:height+=1
    intrinsic=scaled_intrinsics(meta['width'],meta['height'],width,height,known_intrinsics)
    output=Path(output);images=output/'images';images.mkdir(parents=True,exist_ok=True)
    stream=cv2.VideoCapture(str(path));accepted=[];previous=None;rejected={'blur':0,'duplicate':0,'decode':0}
    index=0
    try:
        while len(accepted)<max_frames:
            okay,bgr=stream.read()
            if not okay:break
            stamp=stream.get(cv2.CAP_PROP_POS_MSEC)/1000
            if index%effective_stride:
                index+=1;continue
            if stamp<=0 and index:stamp=index/meta['fps']
            resized=cv2.resize(bgr,(width,height),interpolation=cv2.INTER_AREA)
            gray=cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY)
            blur=float(cv2.Laplacian(gray,cv2.CV_64F).var())
            if blur<blur_threshold:rejected['blur']+=1
            elif previous is not None and float(np.mean(np.abs(gray.astype(np.int16)-previous.astype(np.int16))))<duplicate_mae:
                rejected['duplicate']+=1
            else:
                name=f'frame_{len(accepted):04d}.png';target=images/name
                if not cv2.imwrite(str(target),resized):raise OSError(target)
                accepted.append({'index':len(accepted),'source_frame_index':index,
                    'timestamp_s':float(stamp),'image':f'images/{name}','blur_score':blur})
                previous=gray
            index+=1
    finally:stream.release()
    if len(accepted)<2:raise ValueError('Fewer than two accepted video frames')
    report={'schema_version':1,'source_video':str(Path(path).resolve()),'video_metadata':meta,
            'frames_decoded':index,'frames_extracted':len(accepted),'rejected':rejected,
            'settings':{'stride':stride,'effective_stride':effective_stride,'sampling':sampling,
                'max_frames':max_frames,'target_width':target_width,
                'blur_threshold':blur_threshold,'duplicate_mae':duplicate_mae},
            'intrinsics_initial':intrinsic,'frames':accepted,'elapsed_s':time.perf_counter()-start}
    (output/'extraction.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=3);p.add_argument('--max-frames',type=int,default=60)
    p.add_argument('--width',type=int,default=480);p.add_argument('--blur-threshold',type=float,default=15)
    p.add_argument('--duplicate-mae',type=float,default=1.5);p.add_argument('--intrinsics-json',type=Path)
    p.add_argument('--sampling',choices=('sequential','uniform'),default='sequential')
    a=p.parse_args();known=json.loads(a.intrinsics_json.read_text()) if a.intrinsics_json else None
    result=extract_video(a.video,a.output,stride=a.stride,max_frames=a.max_frames,
        target_width=a.width,blur_threshold=a.blur_threshold,duplicate_mae=a.duplicate_mae,
        known_intrinsics=known,sampling=a.sampling)
    print(json.dumps({k:v for k,v in result.items() if k!='frames'},indent=2),flush=True)


if __name__=='__main__':main()
