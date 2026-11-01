"""Bake real RGB observations into independent triangle UV charts, never fill geometry."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates, distance_transform_edt
from .video_camera import project


def bake(run, max_triangles=20000, tile=12):
    import open3d as o3d
    import trimesh
    run=Path(run);out=run/'photoreal/texture';out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    selected=run/'sfm/quality/filtered_sfm.json'
    sfm=json.loads((selected if selected.is_file() else run/'sfm/sfm.json').read_text())
    if sfm['depth_or_gt_pose_used']:raise ValueError('RGB-only texture expects SfM poses')
    mesh=o3d.io.read_triangle_mesh(str(run/'dense/rgb_only_sfm_tsdf_mesh.ply'))
    original=len(mesh.triangles)
    mesh.remove_duplicated_vertices();mesh.remove_duplicated_triangles();mesh.remove_degenerate_triangles();mesh.remove_unreferenced_vertices()
    if len(mesh.triangles)>max_triangles:mesh=mesh.simplify_quadric_decimation(max_triangles)
    mesh.compute_vertex_normals();mesh.compute_triangle_normals()
    xyz=np.asarray(mesh.vertices);faces=np.asarray(mesh.triangles);tri=xyz[faces];n=len(faces)
    if not n:raise ValueError('No mesh surface to texture')
    centers=tri.mean(axis=1);normals=np.asarray(mesh.triangle_normals);extent=float(np.linalg.norm(np.ptp(xyz,axis=0)))
    # Rank only positive, in-frame camera observations; no depth/GT needed.
    scores=np.zeros((len(sfm['frames']),n))
    for i,f in enumerate(sfm['frames']):
        pose=np.asarray(f['T_world_camera']);c=(centers-pose[:3,3])@pose[:3,:3];cam=sfm['cameras'][str(f['camera_id'])]
        uv=project(cam,c);direction=pose[:3,3]-centers;distance=np.linalg.norm(direction,axis=1)
        cosine=np.abs(np.sum(normals*direction,axis=1))/np.maximum(distance,1e-9)
        valid=(c[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<cam['width']-1)&(uv[:,1]>=0)&(uv[:,1]<cam['height']-1)&(cosine>.15)
        scores[i]=np.where(valid,cosine**2/np.maximum(distance**2,1e-9),0)
    top=np.argsort(scores,axis=0)[-min(3,len(scores)):]
    # Each triangle owns a non-overlapping padded square; barycentric texels remain true photograph samples.
    yy,xx=np.mgrid[0:tile,0:tile];u=(xx-1)/(tile-3);v=(yy-1)/(tile-3)
    inside=(u>=0)&(v>=0)&(u+v<=1);bary=np.stack([1-u-v,u,v],axis=-1)[inside]
    points=np.einsum('pk,nkj->npj',bary,tri).astype(np.float32);pcount=len(bary)
    sums=np.zeros((n,pcount,3),np.float64);weights=np.zeros((n,pcount),np.float64);observations=np.zeros((n,pcount),np.int32)
    scene=o3d.t.geometry.RaycastingScene();scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    for i,f in enumerate(sfm['frames']):
        ids=np.flatnonzero(np.any(top==i,axis=0)&(scores[i]>0))
        if not len(ids):continue
        pose=np.asarray(f['T_world_camera']);cam=sfm['cameras'][str(f['camera_id'])];rgb=np.asarray(Image.open(run/'extracted/images'/f['image']).convert('RGB'),np.float32)
        world=points[ids].reshape(-1,3);camera=(world-pose[:3,3])@pose[:3,:3];uv=project(cam,camera)
        direction=world-pose[:3,3];distance=np.linalg.norm(direction,axis=1);direction/=np.maximum(distance[:,None],1e-9)
        rays=np.column_stack([np.broadcast_to(pose[:3,3],world.shape),direction]).astype(np.float32)
        hit=scene.cast_rays(o3d.core.Tensor(rays))['t_hit'].numpy()
        valid=(camera[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<cam['width']-1)&(uv[:,1]>=0)&(uv[:,1]<cam['height']-1)&(np.abs(hit-distance)<max(extent*1e-4,1e-5))
        colors=np.stack([map_coordinates(rgb[:,:,channel],[uv[:,1],uv[:,0]],order=1,mode='nearest') for channel in range(3)],axis=-1).reshape(len(ids),pcount,3)
        w=valid.reshape(len(ids),pcount)*scores[i,ids,None]
        sums[ids]+=colors*w[:,:,None];weights[ids]+=w;observations[ids]+=valid.reshape(len(ids),pcount)
        print(f'Texture camera {i+1}/{len(sfm["frames"])} visible_texels={valid.sum()}',flush=True)
    colors=np.full_like(sums,128);np.divide(sums,weights[:,:,None],out=colors,where=weights[:,:,None]>0)
    grid=int(np.ceil(np.sqrt(n)));size=grid*tile;atlas=np.full((size,size,3),128,np.uint8);uvs=[]
    nearest=distance_transform_edt(~inside,return_distances=False,return_indices=True)
    for i in range(n):
        patch=np.zeros((tile,tile,3),np.uint8);patch[inside]=np.clip(np.rint(colors[i]),0,255).astype(np.uint8);patch[~inside]=patch[nearest[0][~inside],nearest[1][~inside]]
        row,col=divmod(i,grid);atlas[row*tile:(row+1)*tile,col*tile:(col+1)*tile]=patch
        coords=np.array([[1,1],[tile-2,1],[1,tile-2]],float)+[col*tile+.5,row*tile+.5]
        coords/=size;coords[:,1]=1-coords[:,1];uvs.extend(coords)
    image=Image.fromarray(atlas);image.save(out/'atlas.png')
    flat=tri.reshape(-1,3);ff=np.arange(len(flat)).reshape(-1,3);uvs=np.asarray(uvs)
    material=trimesh.visual.material.PBRMaterial(baseColorTexture=image,metallicFactor=0,roughnessFactor=1,doubleSided=True)
    result=trimesh.Trimesh(vertices=flat,faces=ff,process=False,visual=trimesh.visual.texture.TextureVisuals(uv=uvs,material=material))
    result.export(out/'textured.glb')
    # Portable explicit arrays for native VTK viewer, same geometry and UV as GLB.
    np.savez(out/'surface.npz',vertices=flat,faces=ff,uv=uvs)
    independent=trimesh.load(out/'textured.glb',force='scene')
    from pygltflib import GLTF2
    gltf=GLTF2().load(str(out/'textured.glb'))
    valid_glb=bool(gltf.images and gltf.textures and all(p.attributes.TEXCOORD_0 is not None and p.material is not None for m in gltf.meshes for p in m.primitives))
    if not valid_glb:raise RuntimeError('GLB lost texture/UV during export')
    report={'source_mesh':str((run/'dense/rgb_only_sfm_tsdf_mesh.ply').resolve()),'original_triangles':original,'triangles':n,
      'atlas_resolution':[size,size],'uv_charts':n,'visible_texel_fraction':float(np.mean(weights>0)),
      'multiview_texel_fraction':float(np.mean(observations>1)),'unobserved_texels':'neutral gray, not hallucinated',
      'angle_weight':'absolute normal cosine squared / distance squared, two-sided surface',
      'occlusion':'raycast first-hit distance tolerance 1e-4 scene diagonal','max_views_per_chart':3,
      'glb_texture_valid':valid_glb,'independent_reload_geometries':len(independent.geometry),'elapsed_s':time.perf_counter()-start,
      'geometry_accuracy':'not validated; texturing does not correct distorted geometry'}
    (out/'texture_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
    return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);a=p.parse_args();bake(a.run)
