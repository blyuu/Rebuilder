"""TSDF fusion of SfM-aligned RGB-only monocular depth in arbitrary SfM units."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from .video_camera import rectify


def fuse(extraction: Path, sfm_path: Path, depth_dir: Path, output: Path,
         *, voxel_length: float = .4, depth_trunc: float = 35.) -> dict:
    import open3d as o3d
    if voxel_length <= 0 or depth_trunc <= 0: raise ValueError('Positive SfM-unit settings required')
    extraction, depth_dir, output = Path(extraction), Path(depth_dir), Path(output)
    sfm=json.loads(Path(sfm_path).read_text(encoding='utf-8'))
    depth_report=json.loads((depth_dir/'depth_report.json').read_text(encoding='utf-8'))
    if sfm['depth_or_gt_pose_used'] or 'SfM' not in depth_report['aligned_unit']:
        raise ValueError('Expected SfM-only independent pose and depth alignment')
    output.mkdir(parents=True,exist_ok=True)
    # This host's Open3D 0.20 ScalableTSDFVolume returns empty surfaces even for
    # synthetic planes; match the verified Phase 2 bounded uniform implementation.
    low=np.full(3,np.inf);high=np.full(3,-np.inf);usable=[];diagnostics=[]
    for frame in sfm['frames']:
        depth_path=depth_dir/(Path(frame['image']).stem+'_aligned_sfm.npy')
        if not depth_path.is_file():
            diagnostics.append({'image':frame['image'],'reason':'alignment_missing','valid_pixels':0});continue
        depth=np.load(depth_path,mmap_mode='r')
        cam=sfm['cameras'][str(frame['camera_id'])]
        if depth.shape!=(cam['height'],cam['width']):raise ValueError(f'Depth/intrinsics resolution mismatch: {frame["image"]}')
        k=np.asarray(cam['K'])
        if not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0:raise ValueError('Invalid SfM intrinsics')
        rectified,_=rectify(cam,depth)
        v,u=np.mgrid[0:rectified.shape[0]:16,0:rectified.shape[1]:16]
        z=rectified[::16,::16];valid=np.isfinite(z)&(z>0)&(z<depth_trunc)
        full_valid=int(np.count_nonzero(np.isfinite(rectified)&(rectified>0)&(rectified<depth_trunc)))
        diagnostics.append({'image':frame['image'],'reason':None if np.any(valid) else
            'no_depth_below_truncation' if np.any(np.isfinite(rectified)&(rectified>0)) else 'alignment_empty',
            'valid_pixels':full_valid,'sampled_valid_pixels':int(np.count_nonzero(valid))})
        if not np.any(valid):continue
        usable.append(frame)
        xyz=np.column_stack(((u[valid]-k[0,2])*z[valid]/k[0,0],
            (v[valid]-k[1,2])*z[valid]/k[1,1],z[valid]))
        pose=np.asarray(frame['T_world_camera'])
        world=xyz@pose[:3,:3].T+pose[:3,3]
        low=np.minimum(low,world.min(axis=0));high=np.maximum(high,world.max(axis=0))
    input_report={'registered_frames':sfm['registered_frames'],'usable_aligned_depth_frames':len(usable),
        'depth_trunc_sfm':depth_trunc,'per_frame':diagnostics,'scale_status':'arbitrary_sfm_scale'}
    (output/'dense_input_diagnostics.json').write_text(json.dumps(input_report,indent=2),encoding='utf-8')
    if len(usable)<3:
        raise ValueError(f'Insufficient SfM-aligned depth: {len(usable)}/{sfm["registered_frames"]} '
            f'registered frames usable below depth truncation {depth_trunc} SfM units (minimum 3). '
            'Inspect dense_input_diagnostics.json and depth/depth_report.json.')
    margin=max(2*voxel_length,8*voxel_length)
    length=float(np.max(high-low)+2*margin)
    resolution=int(np.ceil(length/voxel_length))
    if resolution>256:raise ValueError(f'Uniform TSDF needs resolution {resolution}>256; increase voxel length')
    origin=(low+high)/2-length/2
    volume=o3d.pipelines.integration.UniformTSDFVolume(length=length,resolution=resolution,
        sdf_trunc=4*voxel_length,color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        origin=origin)
    frames=[];start=time.perf_counter()
    cameras=sfm['cameras']
    for frame in usable:
        depth_path=depth_dir/(Path(frame['image']).stem+'_aligned_sfm.npy')
        if not depth_path.is_file(): continue
        depth=np.load(depth_path).astype(np.float32)
        rgb=np.asarray(Image.open(extraction/'images'/frame['image']).convert('RGB'))
        cam=cameras[str(frame['camera_id'])]
        depth,rgb=rectify(cam,depth,rgb)
        depth[~np.isfinite(depth) | (depth<=0) | (depth>depth_trunc)]=0
        K=np.asarray(cam['K'])
        intrinsic=o3d.camera.PinholeCameraIntrinsic(int(cam['width']),int(cam['height']),
            float(K[0,0]),float(K[1,1]),float(K[0,2]),float(K[1,2]))
        # Legacy Open3D integration on this host needs uint16 depth input.
        raw=np.rint(depth*1000).astype(np.uint16)
        rgbd=o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(rgb)),o3d.geometry.Image(raw),
            depth_scale=1000.,depth_trunc=depth_trunc,convert_rgb_to_intensity=False)
        volume.integrate(rgbd,intrinsic,np.linalg.inv(np.asarray(frame['T_world_camera'])))
        frames.append({'image':frame['image'],'valid_depth_pixels':int(np.count_nonzero(depth))})
    fusion_s=time.perf_counter()-start
    mesh=volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    cloud=volume.extract_point_cloud()
    # Input-observation support proxy, never independent scene coverage.
    from scipy.spatial import cKDTree
    tree=cKDTree(np.asarray(cloud.points)) if len(cloud.points) else None
    support_total=support_hits=0
    if tree is not None:
        for frame in usable:
            depth_path=depth_dir/(Path(frame['image']).stem+'_aligned_sfm.npy')
            if not depth_path.is_file():continue
            cam=sfm['cameras'][str(frame['camera_id'])]
            depth,_=rectify(cam,np.load(depth_path,mmap_mode='r'))
            K=np.asarray(cam['K'])
            v,u=np.mgrid[0:depth.shape[0]:16,0:depth.shape[1]:16]
            z=depth[::16,::16];valid=np.isfinite(z)&(z>0)&(z<depth_trunc)
            if not np.any(valid):continue
            camera=np.column_stack(((u[valid]-K[0,2])*z[valid]/K[0,0],
                (v[valid]-K[1,2])*z[valid]/K[1,1],z[valid]))
            pose=np.asarray(frame['T_world_camera'])
            world=camera@pose[:3,:3].T+pose[:3,3]
            distance,_=tree.query(world,k=1)
            support_total+=len(distance);support_hits+=int(np.count_nonzero(distance<=2*voxel_length))
    mesh_path=output/'rgb_only_sfm_tsdf_mesh.ply'
    cloud_path=output/'rgb_only_sfm_tsdf_points.ply'
    if not len(mesh.triangles) or not len(cloud.points):raise RuntimeError('Uniform TSDF extracted an empty surface')
    if not o3d.io.write_triangle_mesh(str(mesh_path),mesh):raise OSError(mesh_path)
    if not o3d.io.write_point_cloud(str(cloud_path),cloud):raise OSError(cloud_path)
    result={'input':'RGB-only SfM poses + relative monocular depth aligned to RGB-only sparse geometry',
        'translation_unit':'arbitrary SfM units; mesh is not metric','frames_integrated':len(frames),
        'distortion_rectified':any(cam['model']=='SIMPLE_RADIAL' for cam in cameras.values()),
        'voxel_length_sfm':voxel_length,'sdf_trunc_sfm':4*voxel_length,
        'depth_trunc_sfm':depth_trunc,'fusion_s':fusion_s,
        'volume_length_sfm':length,'volume_resolution':resolution,'volume_origin_sfm':origin.tolist(),
        'vertices':len(mesh.vertices),'triangles':len(mesh.triangles),'points':len(cloud.points),
        'input_observation_support_proxy':support_hits/support_total if support_total else None,
        'support_sample_count':support_total,'support_threshold_sfm':2*voxel_length,
        'support_note':'Fraction of sampled input depth world points near extracted cloud; not independent completeness or accuracy',
        'mesh':str(mesh_path.resolve()),'point_cloud':str(cloud_path.resolve()),'per_frame':frames}
    (output/'dense_report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('extraction',type=Path);p.add_argument('sfm',type=Path)
    p.add_argument('depth',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--voxel-length',type=float,default=.4);p.add_argument('--depth-trunc',type=float,default=35.)
    a=p.parse_args();print(json.dumps(fuse(a.extraction,a.sfm,a.depth,a.output,
        voxel_length=a.voxel_length,depth_trunc=a.depth_trunc),indent=2))


if __name__=='__main__':main()
