"""COLMAP PINHOLE/SIMPLE_RADIAL pixel geometry for RGB-only video."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import map_coordinates


def validate_camera_model(camera):
    model=camera.get('model','PINHOLE')
    if model not in ('PINHOLE','SIMPLE_RADIAL'):
        raise ValueError(f'Unsupported camera model: {model}; explicit conversion required')
    k=np.asarray(camera['K'])
    if k.shape!=(3,3) or not np.isfinite(k).all() or min(k[0,0],k[1,1])<=0:
        raise ValueError('Invalid camera intrinsics')
    if model=='SIMPLE_RADIAL' and (len(camera.get('params',[]))!=4 or not np.isfinite(camera['params']).all()):
        raise ValueError('Invalid SIMPLE_RADIAL parameters')


def normalized_rays(camera:dict,u,v):
    validate_camera_model(camera)
    K=np.asarray(camera['K'],dtype=np.float64)
    xd=(np.asarray(u,dtype=np.float64)-K[0,2])/K[0,0]
    yd=(np.asarray(v,dtype=np.float64)-K[1,2])/K[1,1]
    if camera.get('model','PINHOLE')!='SIMPLE_RADIAL':return xd,yd
    k=float(camera['params'][3]);x=xd.copy();y=yd.copy()
    for _ in range(15):
        f=1+k*(x*x+y*y)
        if np.any(f<=0):raise ValueError('Non-invertible SIMPLE_RADIAL distortion within image')
        x=xd/f;y=yd/f
    return x,y


def backproject(camera:dict,u,v,z):
    x,y=normalized_rays(camera,u,v)
    z=np.asarray(z,dtype=np.float64)
    return np.column_stack((x*z,y*z,z))


def project(camera:dict,xyz):
    validate_camera_model(camera)
    xyz=np.asarray(xyz,dtype=np.float64)
    K=np.asarray(camera['K'],dtype=np.float64)
    x=xyz[...,0]/xyz[...,2];y=xyz[...,1]/xyz[...,2]
    if camera.get('model','PINHOLE')=='SIMPLE_RADIAL':
        f=1+float(camera['params'][3])*(x*x+y*y)
        x=x*f;y=y*f
    return np.stack((K[0,0]*x+K[0,2],K[1,1]*y+K[1,2]),axis=-1)


def rectify(camera:dict,depth:np.ndarray,rgb:np.ndarray|None=None):
    """Resample distorted input onto the same-K ideal pinhole grid for Open3D."""
    validate_camera_model(camera)
    if camera.get('model','PINHOLE')!='SIMPLE_RADIAL':return depth,rgb
    K=np.asarray(camera['K'],dtype=np.float64)
    v,u=np.indices(depth.shape,dtype=np.float64)
    x=(u-K[0,2])/K[0,0];y=(v-K[1,2])/K[1,1]
    f=1+float(camera['params'][3])*(x*x+y*y)
    source_u=K[0,0]*x*f+K[0,2];source_v=K[1,1]*y*f+K[1,2]
    coords=[source_v,source_u]
    clean=np.asarray(depth,dtype=np.float32).copy()
    clean[~np.isfinite(clean) | (clean<=0)]=0
    rect_depth=map_coordinates(clean,coords,order=0,mode='constant',cval=0).astype(np.float32)
    rect_rgb=None
    if rgb is not None:
        rect_rgb=np.stack([map_coordinates(rgb[...,i],coords,order=1,mode='constant',cval=0)
                           for i in range(3)],axis=-1).astype(np.uint8)
    return rect_depth,rect_rgb


def optical_pose(position,target,up):
    """VTK view to OpenCV T_world_camera (x right, y down, z forward)."""
    position,target,up=map(lambda x:np.asarray(x,float),(position,target,up))
    z=target-position;z/=np.linalg.norm(z)
    x=np.cross(z,up);x/=np.linalg.norm(x);y=np.cross(z,x)
    pose=np.eye(4);pose[:3,:3]=np.column_stack([x,y,z]);pose[:3,3]=position
    return pose
