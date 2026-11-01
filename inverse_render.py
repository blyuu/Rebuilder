"""TSDF raycast observations and a constrained diffuse inverse-rendering experiment."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import least_squares
from scipy.ndimage import uniform_filter

from .io import load_rgbd, load_scene


def srgb_to_linear(rgb):
    x=np.asarray(rgb,dtype=np.float64)
    if np.any(~np.isfinite(x)) or np.any((x<0)|(x>1)):raise ValueError('RGB must be finite in [0,1]')
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)


def linear_to_srgb(rgb):
    x=np.clip(np.asarray(rgb,dtype=np.float64),0,1)
    return np.where(x<=.0031308,12.92*x,1.055*x**(1/2.4)-.055)


def unit_normals(normals):
    n=np.asarray(normals,dtype=np.float64)
    if n.ndim!=2 or n.shape[1]!=3:raise ValueError('Expected (N,3) normals')
    length=np.linalg.norm(n,axis=1)
    valid=np.isfinite(n).all(axis=1)&(length>.5)&(length<1.5)
    out=np.zeros_like(n)
    out[valid]=n[valid]/length[valid,None]
    return out,valid


def shade(normals,light,ambient=.6):
    if ambient<=0 or not np.isfinite(ambient):raise ValueError('Positive ambient required')
    l=np.asarray(light,dtype=np.float64)
    if l.shape!=(3,) or not np.isfinite(l).all():raise ValueError('Light must be a finite 3-vector')
    return ambient+np.maximum(0,np.asarray(normals)@l)


def project_world(points,t_world_camera,k):
    p=np.asarray(points,dtype=np.float64)
    cam=(p-t_world_camera[:3,3])@t_world_camera[:3,:3]
    z=cam[:,2]
    u=k.fx*cam[:,0]/np.maximum(z,1e-8)+k.cx
    v=k.fy*cam[:,1]/np.maximum(z,1e-8)+k.cy
    return np.column_stack((u,v,z))


def mesh_data(path):
    import open3d as o3d
    mesh=o3d.io.read_triangle_mesh(str(path))
    vertices=np.asarray(mesh.vertices);triangles=np.asarray(mesh.triangles)
    if len(vertices)==0 or len(triangles)==0:raise ValueError('Empty TSDF mesh')
    raw=np.cross(vertices[triangles[:,1]]-vertices[triangles[:,0]],
                 vertices[triangles[:,2]]-vertices[triangles[:,0]])
    double_area=np.linalg.norm(raw,axis=1)
    normals,valid=unit_normals(raw/np.maximum(double_area[:,None],1e-12))
    valid &= double_area>max(1e-10,float(np.median(double_area))*1e-4)
    return mesh,normals,valid


def raycast_frame(raycast,face_normals,face_valid,frame,scene,width=160,height=120,
                  depth_tolerance_m=.08):
    import open3d as o3d
    if width<1 or height<1 or depth_tolerance_m<=0:raise ValueError('Invalid raycast settings')
    rgb,depth=load_rgbd(frame,scene)
    k=scene.intrinsics
    u=(np.arange(width)+.5)*k.width/width-.5
    v=(np.arange(height)+.5)*k.height/height-.5
    uu,vv=np.meshgrid(u,v)
    direction_cam=np.stack(((uu-k.cx)/k.fx,(vv-k.cy)/k.fy,np.ones_like(uu)),axis=-1)
    direction_world=direction_cam@frame.t_world_camera[:3,:3].T
    origin=np.broadcast_to(frame.t_world_camera[:3,3],direction_world.shape)
    rays=np.concatenate((origin,direction_world),axis=-1).astype(np.float32)
    hit=raycast.cast_rays(o3d.core.Tensor(rays))
    t=hit['t_hit'].numpy();face=hit['primitive_ids'].numpy().astype(np.int64)
    valid_hit=np.isfinite(t)&(face>=0)&(face<len(face_normals))
    safe_face=np.where(valid_hit,face,0)
    # Nearest-neighbor RGB-D sampling at the exact projected ray centers.
    ix=np.clip(np.rint(uu).astype(int),0,k.width-1)
    iy=np.clip(np.rint(vv).astype(int),0,k.height-1)
    measured=depth[iy,ix]
    front=np.sum(face_normals[safe_face]*(-direction_world),axis=-1)>1e-5
    valid=valid_hit&face_valid[safe_face]&front&np.isfinite(measured)&(measured>.1)&(measured<4)
    valid &= np.abs(t-measured)<=depth_tolerance_m
    observed=srgb_to_linear(rgb[iy,ix].astype(np.float64)/255)
    saturated=np.any(rgb[iy,ix]>=250,axis=-1)
    valid &= ~saturated
    return {'face':safe_face,'normal':face_normals[safe_face],
            'rgb':observed,'valid':valid,'depth_residual':np.where(valid_hit,np.abs(t-measured),np.nan),
            'hit':valid_hit,'front':front,'sensor_valid':np.isfinite(measured)&(measured>.1)&(measured<4),
            'measured_depth':measured,'rendered_depth':t}


def observations(frames):
    index=np.concatenate([f['face'][f['valid']] for f in frames])
    normal=np.concatenate([f['normal'][f['valid']] for f in frames])
    rgb=np.concatenate([f['rgb'][f['valid']] for f in frames])
    return index,normal,rgb


def face_mean(index,rgb,n_faces,default=.5):
    index=np.asarray(index);rgb=np.asarray(rgb)
    if index.ndim!=1 or rgb.shape!=(len(index),3) or n_faces<1 or np.any(index<0) or np.any(index>=n_faces):
        raise ValueError('Invalid face observations')
    counts=np.bincount(index,minlength=n_faces)
    sums=np.stack([np.bincount(index,weights=rgb[:,c],minlength=n_faces) for c in range(3)],axis=1)
    values=np.full((n_faces,3),default,dtype=np.float64)
    seen=counts>0
    values[seen]=sums[seen]/counts[seen,None]
    return values,counts


def optimize(index,normals,rgb,n_faces,adjacency=None,regularization=0,
             iterations=8,ambient=.6,anchor=.5):
    index=np.asarray(index);normals=np.asarray(normals);rgb=np.asarray(rgb)
    if (len(index)==0 or normals.shape!=(len(index),3) or rgb.shape!=(len(index),3)
            or not np.isfinite(normals).all() or not np.isfinite(rgb).all()
            or np.any((rgb<0)|(rgb>1)) or not 0<anchor<1 or regularization<0 or iterations<1):
        raise ValueError('Invalid optimization input')
    appearance,counts=face_mean(index,rgb,n_faces)
    albedo=np.clip(appearance.copy(),.01,.99)
    direction=np.mean(normals,axis=0)
    direction/=max(np.linalg.norm(direction),1e-9)
    light=.6*direction+np.array([.04,-.04,0])
    losses=[]
    for _ in range(iterations):
        shading=shade(normals,light,ambient)
        values=rgb/shading[:,None]
        albedo,_=face_mean(index,values,n_faces)
        if regularization>0 and adjacency is not None:
            a,b=adjacency
            # Adjacent faces with strongly different observed appearance are less coupled.
            similarity=np.exp(-np.sum((appearance[a]-appearance[b])**2,axis=1)/.015)
            weights=regularization*similarity
            sums=np.zeros_like(albedo);degree=np.zeros(n_faces)
            np.add.at(sums,a,weights[:,None]*albedo[b]);np.add.at(sums,b,weights[:,None]*albedo[a])
            np.add.at(degree,a,weights);np.add.at(degree,b,weights)
            albedo=(counts[:,None]*albedo+sums)/(counts[:,None]+degree[:,None]+1e-9)
        albedo=np.clip(albedo,.01,.99)
        # Gauge: mean observed albedo luminance is fixed, removing global scale freedom.
        mean=float(np.mean(albedo[index]));albedo=np.clip(albedo*anchor/max(mean,1e-8),.01,.99)
        # Subsample deterministic observations to bound solve time.
        step=max(1,len(index)//10000)
        sample=np.arange(0,len(index),step)
        def residual(l):
            return ((albedo[index[sample]]*shade(normals[sample],l,ambient)[:,None]
                     -rgb[sample])).ravel()
        fit=least_squares(residual,light,bounds=(-1.5,1.5),max_nfev=35)
        light=fit.x
        loss=float(np.mean((albedo[index]*shade(normals,light,ambient)[:,None]-rgb)**2))
        losses.append(loss)
    return {'albedo':albedo,'light':light,'counts':counts,'losses':losses,
            'ambient':ambient,'anchor':anchor,'regularization':regularization}


def face_adjacency(triangles):
    owners={};pairs=[]
    for face,tri in enumerate(np.asarray(triangles)):
        for x,y in ((tri[0],tri[1]),(tri[1],tri[2]),(tri[2],tri[0])):
            key=(min(x,y),max(x,y))
            if key in owners:pairs.append((owners[key],face))
            else:owners[key]=face
    return np.asarray(pairs,dtype=np.int64).T if pairs else np.empty((2,0),dtype=np.int64)


def metric_values(pred,target,mask):
    if not np.any(mask):return {'pixels':0,'mae_linear':None,'rmse_linear':None,'psnr_db':None,'ssim_srgb':None}
    delta=pred[mask]-target[mask]
    mse=float(np.mean(delta**2))
    # Structural similarity uses full images with invalid pixels zeroed in both inputs.
    p=np.zeros_like(pred);t=np.zeros_like(target)
    p[mask]=linear_to_srgb(pred[mask]);t[mask]=linear_to_srgb(target[mask])
    mu_p=uniform_filter(p,size=(7,7,1));mu_t=uniform_filter(t,size=(7,7,1))
    var_p=uniform_filter(p*p,size=(7,7,1))-mu_p*mu_p
    var_t=uniform_filter(t*t,size=(7,7,1))-mu_t*mu_t
    cov=uniform_filter(p*t,size=(7,7,1))-mu_p*mu_t
    ssim_map=((2*mu_p*mu_t+.01**2)*(2*cov+.03**2))/((mu_p*mu_p+mu_t*mu_t+.01**2)*(var_p+var_t+.03**2))
    ssim=float(np.mean(ssim_map[mask]))
    return {'pixels':int(mask.sum()),'mae_linear':float(np.mean(np.abs(delta))),
            'rmse_linear':float(np.sqrt(mse)),'psnr_db':float(-10*np.log10(max(mse,1e-12))),
            'ssim_srgb_masked':ssim}


def save_rgb(path,array):
    Image.fromarray(np.rint(np.clip(array,0,1)*255).astype(np.uint8)).save(path)


def run(manifest,mesh_path,output,width=160,height=120,depth_tolerance_m=.08,
        regularization=8.0,pose_source='GT'):
    import open3d as o3d
    import psutil
    started=time.perf_counter();output=Path(output);output.mkdir(parents=True,exist_ok=True)
    scene=load_scene(Path(manifest));mesh,normals,face_valid=mesh_data(Path(mesh_path))
    geometry_s=time.perf_counter()-started
    tensor=o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    caster=o3d.t.geometry.RaycastingScene();caster.add_triangles(tensor)
    frames=[raycast_frame(caster,normals,face_valid,f,scene,width,height,depth_tolerance_m)
            for f in scene.frames]
    correspondence_s=time.perf_counter()-started-geometry_s
    n=len(frames);train=list(range(min(12,n-4)));validation=list(range(len(train),n-2));test=list(range(n-2,n))
    if not train or not validation or not test:raise ValueError('Need at least six frames for train/validation/test')
    idx,nrm,rgb=observations([frames[i] for i in train]);n_faces=len(normals)
    if not len(idx):raise ValueError('No visible TSDF/RGB-D correspondences')
    appearance,counts=face_mean(idx,rgb,n_faces)
    constant=np.tile(np.mean(rgb,axis=0),(n_faces,1))
    adjacency=face_adjacency(np.asarray(mesh.triangles))
    fit_start=time.perf_counter()
    plain=optimize(idx,nrm,rgb,n_faces,adjacency,0)
    smooth=optimize(idx,nrm,rgb,n_faces,adjacency,regularization)
    optimize_s=time.perf_counter()-fit_start
    models={'constant_albedo':{'albedo':constant,'light':[0,0,0],'ambient':1},
            'observed_appearance':{'albedo':appearance,'light':[0,0,0],'ambient':1},
            'lambertian_unregularized':plain,'lambertian_regularized':smooth}
    metrics={};render_start=time.perf_counter()
    for split,indices in [('validation',validation),('test',test)]:
        rows=[]
        for i in indices:
            f=frames[i];safe=f['face'];native=f['valid']&(counts[safe]>0)
            # All models use exactly the same valid intersection mask.
            mask=native.copy()
            item={'frame':i,'timestamp_s':scene.frames[i].timestamp_s,
                  'native_coverage':{name:float(native.mean()) for name in models},
                  'common_coverage':float(mask.mean()),'correspondences':int(mask.sum()),'methods':{}}
            if split=='test':
                folder=output/f'frame_{i:04d}';folder.mkdir(exist_ok=True)
                save_rgb(folder/'original.png',linear_to_srgb(f['rgb']))
                save_rgb(folder/'mask.png',np.repeat(mask[:,:,None],3,axis=2).astype(float))
            for name,model in models.items():
                shading=shade(f['normal'],model['light'],model['ambient'])
                albedo=model['albedo'][safe]
                predicted=np.clip(albedo*shading[:,:,None],0,1)
                item['methods'][name]=metric_values(predicted,f['rgb'],mask)
                if split=='test':
                    save_rgb(folder/f'{name}_reconstructed.png',linear_to_srgb(predicted))
                    save_rgb(folder/f'{name}_albedo.png',linear_to_srgb(albedo))
                    save_rgb(folder/f'{name}_shading.png',np.repeat(np.clip(shading[:,:,None]/2,0,1),3,axis=2))
                    save_rgb(folder/f'{name}_error.png',np.clip(np.abs(predicted-f['rgb'])*4,0,1))
            if split=='test':save_rgb(folder/'normals.png',(f['normal']+1)/2)
            rows.append(item)
        metrics[split]=rows
    rendering_s=time.perf_counter()-render_start
    # Persist triangle albedo for GUI inspection, with mesh topology retained.
    np.savez_compressed(output/'scene_material.npz',albedo=smooth['albedo'],
        light=smooth['light'],counts=counts,normal=normals)
    config={'manifest':str(Path(manifest).resolve()),'mesh':str(Path(mesh_path).resolve()),
            'pose_source':pose_source,'width':width,'height':height,
            'depth_tolerance_m':depth_tolerance_m,'regularization':regularization,
            'ambient_fixed':.6,'albedo_mean_anchor':.5,
            'train_indices':train,'validation_indices':validation,'test_indices':test,
            'color_space_fit':'linear RGB','ssim':'7x7 local SSIM in sRGB; invalid pixels zeroed and averaged on common mask',
            'geometry_from_sensor_depth':True}
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    result={'config':config,'mesh_triangles':n_faces,'normal_valid_fraction':float(face_valid.mean()),
            'train_correspondences':len(idx),'train_observed_faces':int((counts>0).sum()),
            'frame_diagnostics':[{'frame':i,'hit':int(f['hit'].sum()),'front':int((f['hit']&f['front']).sum()),
                'sensor_valid':int(f['sensor_valid'].sum()),
                'depth_consistent':int((f['hit']&f['front']&f['sensor_valid']&
                    (f['depth_residual']<=depth_tolerance_m)).sum()),
                'depth_and_front_valid':int(f['valid'].sum()),'coverage':float(f['valid'].mean()),
                'median_depth_residual_m':float(np.nanmedian(f['depth_residual']))} for i,f in enumerate(frames)],
            'metrics':metrics,'light_unregularized':plain['light'].tolist(),
            'light_regularized':smooth['light'].tolist(),
            'loss_unregularized':plain['losses'],'loss_regularized':smooth['losses'],
            'timings_s':{'geometry':geometry_s,'correspondence':correspondence_s,
                'optimization':optimize_s,'render_and_metrics':rendering_s,
                'total':time.perf_counter()-started},
            'process_rss_bytes':psutil.Process().memory_info().rss,
            'gpu_allocated_bytes':0}
    (output/'results.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'output':str(output),'train_correspondences':len(idx),
        'test':[{k:v['psnr_db'] for k,v in row['methods'].items()} for row in metrics['test']],
        'timings_s':result['timings_s']},indent=2),flush=True)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('manifest',type=Path);p.add_argument('mesh',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--width',type=int,default=160)
    p.add_argument('--height',type=int,default=120);p.add_argument('--regularization',type=float,default=8)
    p.add_argument('--depth-tolerance-m',type=float,default=.08)
    p.add_argument('--pose-source',default='GT')
    a=p.parse_args();run(a.manifest,a.mesh,a.output,a.width,a.height,a.depth_tolerance_m,
                         a.regularization,a.pose_source)


if __name__=='__main__':main()
