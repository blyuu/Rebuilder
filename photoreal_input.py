"""Opt-in full-clip RGB selector; preserves the existing extraction contract."""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from .video_input import inspect_video, scaled_intrinsics


def temporal_bin(index, count, bins):
    if count<1 or bins<1 or not 0<=index<count:raise ValueError('Invalid temporal bin input')
    return min(bins-1,int(index*bins/count))


def continuity(previous, current):
    """Descriptor-match fraction, not triangulation, parallax or pose accuracy."""
    if previous is None:return None
    if current is None or len(previous)<2 or len(current)<2:return 0.
    pairs=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(current,previous,k=2)
    return sum(len(pair)==2 and pair[0].distance<.75*pair[1].distance for pair in pairs)/len(current)


def extract(video:Path, output:Path, *, max_frames=120, candidate_stride=3,
            width=480, blur_threshold=15., candidates_per_bin=3):
    if max_frames<8 or candidate_stride<1 or width<32 or candidates_per_bin<1 or blur_threshold<0:
        raise ValueError('Invalid photoreal frame selection settings')
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise FileExistsError('Use a fresh extraction directory')
    start=time.perf_counter();meta=inspect_video(video)
    count=meta['reported_frames'];bins=min(max_frames,count)
    height=round(meta['height']*width/meta['width']);height+=height%2
    images=output/'images';images.mkdir(parents=True,exist_ok=True)
    cv2.setRNGSeed(2026);orb=cv2.ORB_create(nfeatures=600)
    selected=[];empty=[];previous=None;buffer=[];current_bin=0;decoded=0;blur_rejected=0
    def flush(bin_id):
        nonlocal previous,buffer
        if not buffer:empty.append(bin_id);return
        candidates=[]
        peak=max(row[0] for row in buffer)
        for sharpness,index,stamp,bgr in buffer:
            gray=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY)
            thumb=cv2.resize(gray,(160,max(32,round(height*160/width))))
            _,descriptor=orb.detectAndCompute(thumb,None)
            overlap=continuity(previous,descriptor)
            score=.5*sharpness/max(peak,1e-6)+(overlap or 0.)
            candidates.append((score,index,stamp,bgr,sharpness,descriptor,overlap))
        choice=max(candidates,key=lambda row:(row[0],-row[1]))
        score,index,stamp,bgr,sharpness,descriptor,overlap=choice
        name=f'frame_{len(selected):04d}.png'
        if not cv2.imwrite(str(images/name),bgr):raise OSError(images/name)
        selected.append({'index':len(selected),'source_frame_index':index,'timestamp_s':stamp,
            'image':'images/'+name,'blur_score':sharpness,'temporal_bin':bin_id,
            'continuity_match_fraction':overlap,'selection_score':score,
            'candidate_count':len(buffer)})
        previous=descriptor;buffer=[]
    capture=cv2.VideoCapture(str(video))
    try:
        while True:
            ok,bgr=capture.read()
            if not ok:break
            index=decoded;decoded+=1
            if index>=count:raise ValueError('Decoded count exceeds metadata; inspect video timestamps')
            bin_id=temporal_bin(index,count,bins)
            while current_bin<bin_id:flush(current_bin);current_bin+=1
            if index%candidate_stride and index!=count-1:continue
            stamp=capture.get(cv2.CAP_PROP_POS_MSEC)/1000
            if stamp<=0 and index:stamp=index/meta['fps']
            resized=cv2.resize(bgr,(width,height),interpolation=cv2.INTER_AREA)
            sharp=float(cv2.Laplacian(cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY),cv2.CV_64F).var())
            if sharp<blur_threshold:blur_rejected+=1;continue
            buffer.append((sharp,index,float(stamp),resized))
            buffer=sorted(buffer,key=lambda row:(-row[0],row[1]))[:candidates_per_bin]
        flush(current_bin)
        empty.extend(range(current_bin+1,bins))
    finally:capture.release()
    if len(selected)<8:raise ValueError('Fewer than eight sharp temporal bins; photoreal selection failed')
    result={'schema_version':1,'source_video':str(Path(video).resolve()),'video_metadata':meta,
        'frames_decoded':decoded,'frames_extracted':len(selected),'frames':selected,
        'intrinsics_initial':scaled_intrinsics(meta['width'],meta['height'],width,height),
        'settings':{'sampling':'temporal_bins_sharpness_feature_continuity','max_frames':max_frames,
            'candidate_stride':candidate_stride,'target_width':width,'blur_threshold':blur_threshold,
            'candidates_per_bin':candidates_per_bin,'seed':2026},
        'selection_diagnostics':{'empty_bins':empty,'blur_rejected':blur_rejected,
            'early_decode_termination':decoded<count,'decoded_fraction':decoded/count,
            'selected_time_range_s':[selected[0]['timestamp_s'],selected[-1]['timestamp_s']],
            'max_selected_gap_s':max(b['timestamp_s']-a['timestamp_s'] for a,b in zip(selected,selected[1:])),
            'note':'Feature continuity is a 2D descriptor heuristic, not geometric overlap or reconstruction coverage.'},
        'elapsed_s':time.perf_counter()-start}
    (output/'extraction.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if decoded<count:
        raise ValueError(f'Decode ended at {decoded}/{count} reported frames; inspect saved extraction diagnostics')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('video',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--max-frames',type=int,default=120);p.add_argument('--candidate-stride',type=int,default=3)
    p.add_argument('--width',type=int,default=480);a=p.parse_args()
    r=extract(a.video,a.output,max_frames=a.max_frames,candidate_stride=a.candidate_stride,width=a.width)
    print(json.dumps({k:v for k,v in r.items() if k!='frames'},indent=2))
