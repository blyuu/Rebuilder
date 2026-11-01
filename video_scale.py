"""Optional physical scale from a user-measured separation of two SfM landmarks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def reference_scale(landmarks:dict,point_id_a:int,point_id_b:int,distance_m:float)->dict:
    if point_id_a==point_id_b or not np.isfinite(distance_m) or distance_m<=0:
        raise ValueError('Distinct landmarks and positive physical distance required')
    try:
        a=np.asarray(landmarks[str(point_id_a)]['xyz'],dtype=np.float64)
        b=np.asarray(landmarks[str(point_id_b)]['xyz'],dtype=np.float64)
    except KeyError as exc:raise ValueError('Unknown SfM landmark ID') from exc
    if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(a+b).all():
        raise ValueError('Invalid landmark coordinates')
    distance_sfm=float(np.linalg.norm(a-b))
    if distance_sfm<1e-6:raise ValueError('Degenerate landmark separation')
    return {'source':'user-measured physical distance between two RGB-only SfM landmarks',
        'landmark_ids':[point_id_a,point_id_b], 'distance_sfm':distance_sfm,
        'distance_m':float(distance_m),'scale_m_per_sfm_unit':float(distance_m/distance_sfm),
        'note':'External measurement; original SfM pose/depth/mesh artifacts remain in SfM units'}


def main():
    p=argparse.ArgumentParser();p.add_argument('sfm',type=Path)
    p.add_argument('--point-ids',type=int,nargs=2,required=True)
    p.add_argument('--distance-m',type=float,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    sfm=json.loads(a.sfm.read_text(encoding='utf-8'))
    if sfm['depth_or_gt_pose_used']:raise ValueError('Expected RGB-only SfM landmarks')
    result=reference_scale(sfm['landmarks'],*a.point_ids,a.distance_m)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
