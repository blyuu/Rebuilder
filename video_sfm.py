"""RGB-only PyCOLMAP sequential SfM. All translations are in arbitrary SfM units."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .geometry import validate_pose


def colmap_camera_to_world(cam_from_world_3x4: np.ndarray) -> np.ndarray:
    """Invert COLMAP world-to-optical-camera SE(3) into RECON-X T_world_camera."""
    extrinsic=np.asarray(cam_from_world_3x4,dtype=np.float64)
    if extrinsic.shape!=(3,4) or not np.isfinite(extrinsic).all():
        raise ValueError('COLMAP extrinsic must be finite 3x4')
    T_camera_world=np.eye(4);T_camera_world[:3,:]=extrinsic
    validate_pose(T_camera_world)
    result=np.linalg.inv(T_camera_world)
    validate_pose(result)
    return result


def normalize_arbitrary_sfm_scale(report: dict, target_median_z: float = 20.) -> dict:
    """Choose a reproducible SfM unit using triangulated optical depths only."""
    depths=[]
    for frame in report['frames']:
        world_to_camera=np.linalg.inv(np.asarray(frame['T_world_camera'],dtype=np.float64))
        for obs in frame['observations']:
            landmark=report['landmarks'].get(str(obs['point3D_id']))
            if landmark is None:continue
            z=float((world_to_camera@np.r_[landmark['xyz'],1.])[2])
            if np.isfinite(z) and z>0:depths.append(z)
    if len(depths)<25:raise ValueError('Insufficient triangulated observations for SfM scale normalization')
    median=float(np.median(depths));factor=target_median_z/median
    if not np.isfinite(factor) or factor<=0:raise ValueError('Invalid SfM scale normalization')
    for frame in report['frames']:
        pose=np.asarray(frame['T_world_camera'],dtype=np.float64)
        pose[:3,3]*=factor;frame['T_world_camera']=pose.tolist()
    for landmark in report['landmarks'].values():
        landmark['xyz']=(np.asarray(landmark['xyz'],dtype=np.float64)*factor).tolist()
    report['scale_normalization']={'source':'median positive triangulated optical depth',
        'original_median_depth_sfm':median,'factor':factor,'target_median_depth_sfm':target_median_z,
        'unit':'arbitrary SfM units; not meters','observation_count':len(depths)}
    return report


def run_sfm(extraction: Path, output: Path, *, known_intrinsics: dict | None = None,
            overlap: int = 10, max_features: int = 4096,
            auto_fallback: bool = True, exhaustive: bool = False) -> dict:
    import pycolmap

    extraction = Path(extraction)
    if overlap < 1 or max_features < 256: raise ValueError('Invalid SfM matching settings')
    data = json.loads((extraction / 'extraction.json').read_text(encoding='utf-8'))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    image_path = extraction / 'images'
    database = output / 'database.db'
    if database.exists():
        raise FileExistsError('SfM database already exists; use a fresh output directory')
    names = [Path(row['image']).name for row in data['frames']]
    reader = pycolmap.ImageReaderOptions()
    if known_intrinsics:
        vals = [float(known_intrinsics[k]) for k in ('fx', 'fy', 'cx', 'cy')]
        sx = data['intrinsics_initial']['width'] / float(known_intrinsics['width'])
        sy = data['intrinsics_initial']['height'] / float(known_intrinsics['height'])
        vals = [vals[0]*sx, vals[1]*sy, (vals[2]+.5)*sx-.5, (vals[3]+.5)*sy-.5]
        reader.camera_model = 'PINHOLE'
        reader.camera_params = ','.join(f'{v:.9f}' for v in vals)
    else:
        reader.camera_model = 'SIMPLE_RADIAL'
    features = pycolmap.FeatureExtractionOptions()
    features.max_image_size = max(data['intrinsics_initial']['width'], data['intrinsics_initial']['height'])
    features.sift.max_num_features = max_features
    features.num_threads=1
    features.use_gpu = False
    matching = pycolmap.FeatureMatchingOptions()
    matching.num_threads=1
    matching.use_gpu = False
    pairing = pycolmap.SequentialPairingOptions()
    pairing.overlap = overlap
    pairing.quadratic_overlap = True
    times = {}
    start = time.perf_counter()
    pycolmap.extract_features(database, image_path, image_names=names,
        camera_mode=pycolmap.CameraMode.SINGLE, reader_options=reader,
        extraction_options=features, device=pycolmap.Device.cpu)
    times['features_s'] = time.perf_counter()-start
    start = time.perf_counter()
    if exhaustive:
        pycolmap.match_exhaustive(database, matching_options=matching, device=pycolmap.Device.cpu)
    else:
        pycolmap.match_sequential(database, matching_options=matching,
            pairing_options=pairing, device=pycolmap.Device.cpu)
    times['matching_s'] = time.perf_counter()-start
    start = time.perf_counter()
    mapping_options=pycolmap.IncrementalPipelineOptions()
    mapping_options.random_seed=1
    mapping_options.mapper.random_seed=1
    mapping_options.triangulation.random_seed=1
    mapping_options.num_threads=1
    mapping_options.mapper.num_threads=1
    models = pycolmap.incremental_mapping(database, image_path, output / 'models',options=mapping_options)
    times['mapping_s'] = time.perf_counter()-start
    if not models:
        raise RuntimeError('COLMAP registered no camera model')
    model = max(models.values(), key=lambda item: item.num_reg_images())
    model_dir = output / 'selected_model'
    model_dir.mkdir(exist_ok=True)
    model.write(model_dir)
    frame_by_name = {Path(row['image']).name: row for row in data['frames']}
    registered = []
    for img in model.images.values():
        if not img.has_pose:
            continue
        T_world_camera = colmap_camera_to_world(np.asarray(img.cam_from_world().matrix()))
        row = frame_by_name[img.name]
        registered.append({'image': img.name, 'source_frame_index': row['source_frame_index'],
            'timestamp_s': row['timestamp_s'], 'T_world_camera': T_world_camera.tolist(),
            'camera_id': img.camera_id, 'observations': [
                {'xy': np.asarray(p.xy).tolist(), 'point3D_id': int(p.point3D_id)}
                for p in img.points2D if p.has_point3D()]})
    registered.sort(key=lambda row: row['source_frame_index'])
    landmarks = {str(pid): {'xyz': np.asarray(pt.xyz).tolist(), 'error_px': float(pt.error),
        'track_length': len(pt.track.elements)} for pid, pt in model.points3D.items()}
    cameras = {str(cid): {'model': cam.model_name, 'width': cam.width, 'height': cam.height,
        'params': np.asarray(cam.params).tolist(), 'K': np.asarray(cam.calibration_matrix()).tolist()}
        for cid, cam in model.cameras.items()}
    seen = {row['image'] for row in registered}
    report = {'schema_version': 1, 'pose_source': 'RGB-only PyCOLMAP sequential SfM',
        'pose_convention': 'T_world_camera, OpenCV optical camera x right y down z forward',
        'translation_unit': 'arbitrary SfM scale; not meters', 'depth_or_gt_pose_used': False,
        'pycolmap_version': pycolmap.__version__, 'input_frames': len(names),
        'matching_method':'exhaustive' if exhaustive else 'sequential', 'matching_overlap':overlap,'max_sift_features':max_features,'mapping_random_seed':1,
        'mapping_num_threads':1,'feature_num_threads':1,'matching_num_threads':1,
        'registered_frames': len(registered), 'unregistered_images': sorted(set(names)-seen),
        'landmark_count': len(landmarks), 'timing': times, 'cameras': cameras,
        'frames': registered, 'landmarks': landmarks}
    report['camera_initialization']='provided_PINHOLE' if known_intrinsics else 'uncalibrated_SIMPLE_RADIAL'
    if auto_fallback and known_intrinsics is None and len(registered)<max(5,int(.25*len(names))):
        initial=data['intrinsics_initial'];focal=.585*max(initial['width'],initial['height'])
        heuristic={'width':initial['width'],'height':initial['height'],'fx':focal,'fy':focal,
                   'cx':(initial['width']-1)/2,'cy':(initial['height']-1)/2}
        alternative=run_sfm(extraction,output/'pinhole_fallback',known_intrinsics=heuristic,
            overlap=overlap,max_features=max_features,auto_fallback=False,exhaustive=exhaustive)
        observation_count=sum(len(frame['observations']) for frame in alternative['frames'])
        report['fallback_trial']={'camera_model':'PINHOLE','initial_focal_px':focal,
            'registered_frames':alternative['registered_frames'],'landmark_count':alternative['landmark_count'],
            'observation_count':observation_count,'selected':False}
        if (alternative['registered_frames']>report['registered_frames']
                and alternative['registered_frames']>=5 and alternative['landmark_count']>=25
                and observation_count>=25):
            alternative['camera_initialization']='uncalibrated_PINHOLE_heuristic_fallback'
            alternative['fallback_trial']={'original_model':'SIMPLE_RADIAL',
                'original_registered_frames':report['registered_frames'],
                'selected_model':'PINHOLE','selected_registered_frames':alternative['registered_frames'],
                'initial_focal_px':focal,'selection_rule':'more registered frames, at least five cameras, 25 landmarks and 25 observations'}
            report=normalize_arbitrary_sfm_scale(alternative)
    (output/'sfm.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('extraction', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--intrinsics-json', type=Path)
    p.add_argument('--overlap',type=int,default=10)
    p.add_argument('--max-features',type=int,default=4096)
    p.add_argument('--exhaustive',action='store_true')
    a=p.parse_args()
    known=json.loads(a.intrinsics_json.read_text()) if a.intrinsics_json else None
    result=run_sfm(a.extraction,a.output,known_intrinsics=known,
                   overlap=a.overlap,max_features=a.max_features,exhaustive=a.exhaustive)
    print(json.dumps({k:v for k,v in result.items() if k not in ('frames','landmarks')}, indent=2))


if __name__=='__main__':main()
