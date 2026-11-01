"""ROS/Gazebo boundary: metric frames and transport-neutral map serialization."""
from __future__ import annotations

import struct
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np

from .geometry import validate_pose


# A vector in ROS camera_link (forward, left, up) becomes optical
# (right, down, forward). This is the REP-103 optical-frame convention.
R_OPTICAL_LINK = np.array([[0., -1., 0.], [0., 0., -1.], [1., 0., 0.]])
T_OPTICAL_LINK = np.eye(4)
T_OPTICAL_LINK[:3, :3] = R_OPTICAL_LINK


def floor_aligned_transform(floor_points_recon: np.ndarray) -> np.ndarray:
    """Map RECON-X world to z-up Gazebo/ROS map using semantic floor points.

    The original +Z chooses the sign of the floor normal. This assumption is
    checked against the measured floor normal and recorded by the exporter.
    """
    p = np.asarray(floor_points_recon, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3 or len(p) < 20 or not np.isfinite(p).all():
        raise ValueError('At least 20 finite floor points are required')
    center = np.median(p, axis=0)
    distance = np.linalg.norm(p-center, axis=1)
    subset = p[distance <= np.quantile(distance, .8)]
    _, _, vt = np.linalg.svd(subset-subset.mean(axis=0), full_matrices=False)
    up = vt[-1]
    if up[2] < 0: up = -up
    if up[2] < .9:
        raise ValueError('Semantic floor is not near the RECON-X world +Z plane')
    x = np.array([1.,0.,0.])
    x -= np.dot(x,up)*up
    x /= np.linalg.norm(x)
    y = np.cross(up,x)
    rotation = np.stack((x,y,up))
    transform = np.eye(4)
    transform[:3,:3] = rotation
    transform[:3,3] = -(rotation @ center)
    validate_pose(transform)
    return transform


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    validate_pose(transform)
    p=np.asarray(points,dtype=np.float64)
    if p.ndim!=2 or p.shape[1]!=3 or not np.isfinite(p).all():
        raise ValueError('Expected finite Nx3 points')
    return p @ transform[:3,:3].T + transform[:3,3]


def optical_pose_to_map_link(t_recon_optical: np.ndarray, t_map_recon: np.ndarray) -> np.ndarray:
    validate_pose(t_recon_optical);validate_pose(t_map_recon)
    result=t_map_recon @ t_recon_optical @ T_OPTICAL_LINK
    validate_pose(result)
    return result


def pack_semantic_points(points: np.ndarray, class_ids: np.ndarray, confidence: np.ndarray) -> bytes:
    """PointCloud2 records: little-endian float32 xyz, int32 class, float32 confidence."""
    p=np.asarray(points,dtype=np.float32)
    ids=np.asarray(class_ids,dtype=np.int32)
    scores=np.asarray(confidence,dtype=np.float32)
    if p.ndim!=2 or p.shape[1]!=3 or ids.shape!=(len(p),) or scores.shape!=(len(p),):
        raise ValueError('Semantic point arrays disagree')
    if not np.isfinite(p).all() or not np.isfinite(scores).all() or np.any((scores<0)|(scores>1)):
        raise ValueError('Invalid semantic point values')
    dtype=np.dtype([('x','<f4'),('y','<f4'),('z','<f4'),('class_id','<i4'),('confidence','<f4')])
    records=np.empty(len(p),dtype=dtype)
    for i,name in enumerate(('x','y','z')): records[name]=p[:,i]
    records['class_id']=ids;records['confidence']=scores
    return records.tobytes()


def semantic_point_fields() -> list[tuple[str,int,int]]:
    # ROS PointField FLOAT32=7, INT32=5.
    return [('x',0,7),('y',4,7),('z',8,7),('class_id',12,5),('confidence',16,7)]


def gazebo_world(mesh_uri: str, floor_xy: tuple[float,float], *, robot_z: float=.14) -> str:
    """Generate SDF with original mesh for visuals and a separate flat support collision.

    This is a constrained motion sandbox, not a physically faithful room.
    """
    width,height=floor_xy
    if width<=.5 or height<=.5 or not np.isfinite([width,height,robot_z]).all():
        raise ValueError('Invalid support floor dimensions')
    if not (mesh_uri.startswith('file://') or mesh_uri.startswith('model://')):
        raise ValueError('Mesh URI must be file:// or model://')
    sdf=ET.Element('sdf',version='1.9')
    world=ET.SubElement(sdf,'world',name='reconx_tum')
    ET.SubElement(world,'physics',name='default',type='ode')
    ET.SubElement(world,'plugin',filename='gz-sim-physics-system',name='gz::sim::systems::Physics')
    ET.SubElement(world,'plugin',filename='gz-sim-user-commands-system',name='gz::sim::systems::UserCommands')
    ET.SubElement(world,'plugin',filename='gz-sim-scene-broadcaster-system',name='gz::sim::systems::SceneBroadcaster')
    light=ET.SubElement(world,'light',name='sun',type='directional')
    ET.SubElement(light,'direction').text='-0.3 -0.3 -1'
    ET.SubElement(world,'gravity').text='0 0 -9.81'
    scene=ET.SubElement(world,'model',name='reconx_visual_mesh')
    ET.SubElement(scene,'static').text='true'
    link=ET.SubElement(scene,'link',name='mesh')
    visual=ET.SubElement(link,'visual',name='tsdf_visual')
    ET.SubElement(ET.SubElement(ET.SubElement(visual,'geometry'),'mesh'),'uri').text=mesh_uri
    floor=ET.SubElement(world,'model',name='simplified_support_floor')
    ET.SubElement(floor,'static').text='true'
    floor_link=ET.SubElement(floor,'link',name='floor')
    ET.SubElement(floor_link,'pose').text='0 0 -0.05 0 0 0'
    for tag in ('visual','collision'):
        element=ET.SubElement(floor_link,tag,name='support_'+tag)
        ET.SubElement(ET.SubElement(ET.SubElement(element,'geometry'),'box'),'size').text=f'{width:.4f} {height:.4f} 0.1'
    robot=ET.SubElement(world,'model',name='reconx_diffbot')
    ET.SubElement(robot,'pose').text=f'0 0 {robot_z:.4f} 0 0 0'
    base=ET.SubElement(robot,'link',name='base_link')
    ET.SubElement(ET.SubElement(base,'inertial'),'mass').text='2.0'
    inertial=base.find('inertial');ET.SubElement(inertial,'inertia')
    for name,val in [('ixx','.02'),('iyy','.02'),('izz','.02'),('ixy','0'),('ixz','0'),('iyz','0')]:
        ET.SubElement(inertial.find('inertia'),name).text=val
    for tag in ('visual','collision'):
        element=ET.SubElement(base,tag,name='base_'+tag)
        ET.SubElement(ET.SubElement(ET.SubElement(element,'geometry'),'cylinder'),'radius').text='0.12'
        ET.SubElement(element.find('geometry').find('cylinder'),'length').text='0.08'
    for side,y in [('left',.13),('right',-.13)]:
        wheel=ET.SubElement(robot,'link',name=side+'_wheel')
        ET.SubElement(wheel,'pose').text=f'0 {y} -0.07 1.57079632679 0 0'
        inertia=ET.SubElement(ET.SubElement(wheel,'inertial'),'inertia')
        ET.SubElement(wheel.find('inertial'),'mass').text='.25'
        for name,val in [('ixx','.001'),('iyy','.001'),('izz','.001'),('ixy','0'),('ixz','0'),('iyz','0')]:
            ET.SubElement(inertia,name).text=val
        for tag in ('visual','collision'):
            element=ET.SubElement(wheel,tag,name=side+'_'+tag)
            cylinder=ET.SubElement(ET.SubElement(element,'geometry'),'cylinder')
            ET.SubElement(cylinder,'radius').text='.07';ET.SubElement(cylinder,'length').text='.04'
        joint=ET.SubElement(robot,'joint',name=side+'_joint',type='revolute')
        ET.SubElement(joint,'parent').text='base_link';ET.SubElement(joint,'child').text=side+'_wheel'
        axis=ET.SubElement(joint,'axis');ET.SubElement(axis,'xyz').text='0 0 1'
        ET.SubElement(ET.SubElement(axis,'limit'),'effort').text='5'
    drive=ET.SubElement(robot,'plugin',filename='gz-sim-diff-drive-system',name='gz::sim::systems::DiffDrive')
    for name,value in [('left_joint','left_joint'),('right_joint','right_joint'),('wheel_separation','.26'),
                       ('wheel_radius','.07'),('topic','/cmd_vel'),('odom_topic','/odom')]:
        ET.SubElement(drive,name).text=value
    return ET.tostring(sdf,encoding='unicode')
