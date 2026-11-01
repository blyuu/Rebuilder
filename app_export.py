"""Artifact export with independent readback; no reconstruction or scale conversion."""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from .app_project import atomic_json

FORMATS={'mesh':('ply','obj','glb'),'textured_mesh':('glb',),'cloud':('ply',),'semantic':('json',),'camera':('json',),'gaussian':('pt',)}


def textured_mesh(source,destination):
    from pygltflib import GLTF2
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Choose a different destination')
    shutil.copy2(source,destination)
    gltf=GLTF2().load(str(destination))
    if not gltf.images or not gltf.textures or not gltf.meshes or not all(
            p.attributes.TEXCOORD_0 is not None and p.material is not None
            for m in gltf.meshes for p in m.primitives):
        destination.unlink(missing_ok=True)
        raise ValueError('Exported GLB lost UV texture or material')
    return {'path':str(destination.resolve()),'bytes':destination.stat().st_size,
            'validated':True,'textures_exported':True,'uv_coordinates':True,
            'geometry_accuracy':'not established; RGB texture only'}

def scale_status(snapshot):
    return 'arbitrary_sfm_scale' if snapshot.get('input_type')=='rgb_video' else snapshot.get('scale_convention','unknown')

def geometry(source, destination, kind='mesh', scale='unknown'):
    import open3d as o3d
    import trimesh
    destination=Path(destination);source=Path(source)
    if destination.suffix[1:].lower() not in FORMATS[kind]:raise ValueError('Unsupported export format')
    if not source.is_file():raise FileNotFoundError(source)
    if source.resolve()==destination.resolve():raise ValueError('Export must not overwrite its source')
    if kind=='mesh':
        original=o3d.io.read_triangle_mesh(str(source))
        points=np.asarray(original.vertices);faces=np.asarray(original.triangles);colors=np.asarray(original.vertex_colors)
        if not len(faces):raise ValueError('No triangle geometry available')
        model=trimesh.Trimesh(vertices=points,faces=faces,vertex_colors=np.rint(colors*255).astype('uint8') if len(colors) else None,process=False)
    else:
        original=o3d.io.read_point_cloud(str(source));points=np.asarray(original.points);colors=np.asarray(original.colors);faces=np.empty((0,3),dtype=int)
        model=trimesh.points.PointCloud(points,colors=np.rint(colors*255).astype('uint8') if len(colors) else None)
    if not len(points) or not np.isfinite(points).all():raise ValueError('Empty or nonfinite geometry')
    # Parent must exist: typos/invalid destinations must not silently create directory trees.
    if not destination.parent.is_dir():raise FileNotFoundError(destination.parent)
    if destination.suffix.lower()=='.obj':
        from trimesh.exchange.obj import export_obj
        destination.write_text(export_obj(model,include_normals=False,include_color=False,include_texture=False),encoding='utf-8')
    else:model.export(str(destination))
    fmt=destination.suffix.lower()
    if fmt=='.glb':
        from pygltflib import GLTF2
        gltf=GLTF2().load(str(destination));blob=gltf.binary_blob()
        if gltf.asset.version!='2.0' or len(gltf.meshes)!=1:raise ValueError('Unexpected glTF structure')
        for node in gltf.nodes:
            if node.matrix and not np.allclose(np.asarray(node.matrix).reshape(4,4),np.eye(4)):raise ValueError('Unexpected node transform')
            if node.translation or node.rotation or node.scale:raise ValueError('Unexpected TRS transform')
        primitive=gltf.meshes[0].primitives[0]
        if primitive.mode!=4:raise ValueError('Expected triangle primitive')
        def accessor(index):
            a=gltf.accessors[index];v=gltf.bufferViews[a.bufferView]
            dtype={5126:'<f4',5125:'<u4',5123:'<u2',5121:'u1'}[a.componentType]
            components={'SCALAR':1,'VEC3':3,'VEC4':4}[a.type]
            if v.byteStride:raise ValueError('Unexpected interleaved accessor')
            return np.frombuffer(blob,dtype=dtype,count=a.count*components,offset=(v.byteOffset or 0)+(a.byteOffset or 0)).reshape(a.count,components)
        actual=accessor(primitive.attributes.POSITION);triangles=accessor(primitive.indices).reshape(-1,3)
        rgb=accessor(primitive.attributes.COLOR_0)[:,:3]/255 if primitive.attributes.COLOR_0 is not None else np.empty((0,3))
        parser='pygltflib independent accessor readback'
    else:
        loaded=o3d.io.read_triangle_mesh(str(destination)) if kind=='mesh' else o3d.io.read_point_cloud(str(destination))
        actual=np.asarray(loaded.vertices if kind=='mesh' else loaded.points)
        triangles=np.asarray(loaded.triangles) if kind=='mesh' else faces
        rgb=np.asarray(loaded.vertex_colors if kind=='mesh' else loaded.colors)
        parser='Open3D'
    if fmt=='.obj':
        # Assimp/Open3D may reorder OBJ vertices. Remap by nearest unique coordinates,
        # then compare oriented triangles without relying on parser vertex order.
        from scipy.spatial import cKDTree
        distances,mapping=cKDTree(points).query(actual)
        if actual.shape!=points.shape or np.max(distances)>1e-5:raise ValueError('OBJ vertex geometry changed')
        def canonical(f):
            f=np.asarray(f);f=np.take_along_axis(f,(np.argmin(f,axis=1)[:,None]+np.arange(3))%3,axis=1)
            return f[np.lexsort(f.T[::-1])]
        if not np.array_equal(canonical(mapping[triangles]),canonical(faces)):raise ValueError('OBJ topology or winding changed')
    else:
        if actual.shape!=points.shape or not np.allclose(actual,points,atol=1e-5):raise ValueError('Vertex coordinates changed during export')
        if not np.array_equal(triangles,faces):raise ValueError('Topology changed during export')
    if len(colors) and fmt!='.obj' and (rgb.shape!=colors.shape or not np.allclose(rgb,colors,atol=1/255+.00001)):
        raise ValueError('Vertex colors changed during export')
    result={'path':str(destination.resolve()),'bytes':destination.stat().st_size,'vertices':len(points),'triangles':len(faces),
        'bounds':[points.min(0).tolist(),points.max(0).tolist()],'scale_status':scale,'transform':'identity; original world coordinates',
        'vertex_colors_preserved':bool(len(colors) and len(rgb) and np.allclose(rgb,colors,atol=1/255+.00001)),
        'textures_exported':False,'parser':parser,'validated':True}
    atomic_json(destination.with_suffix(destination.suffix+'.validation.json'),result)
    return result

def semantic(source,destination):
    from .semantic_map import SemanticVoxelMap,palette
    source=Path(source);destination=Path(destination)
    model=SemanticVoxelMap.load(source);points,ids,counts=model.extract()
    if not len(points):raise ValueError('No semantic voxels')
    cloud=destination.with_suffix('.ply');model.export_ply(cloud)
    votes=json.loads(source.read_text(encoding='utf-8'))['voxels'];colors=palette(len(model.id_to_label))
    data={'schema_version':1,'geometry':cloud.name,'coordinate_unit':model.coordinate_unit,
          'scale_status':'arbitrary_sfm_scale' if model.coordinate_unit=='sfm' else 'meters',
          'classes':[{'id':k,'name':v,'color':colors[k].tolist()} for k,v in model.id_to_label.items()],
          'positions':points.tolist(),'class_ids':ids.tolist(),'observation_counts':counts.tolist(),
          'vote_purity':[max(v['votes'].values())/sum(v['votes'].values()) for v in votes],
          'confidence_definition':'normalized fused class votes; not calibrated accuracy','source_voxels':votes}
    atomic_json(destination,data)
    import open3d as o3d
    read=o3d.io.read_point_cloud(str(cloud))
    if not np.allclose(np.asarray(read.points),points,atol=1e-6) or not np.allclose(np.asarray(read.colors)*255,colors[ids],atol=1):raise ValueError('Semantic geometry validation failed')
    if json.loads(destination.read_text(encoding='utf-8'))!=data:raise ValueError('Semantic metadata validation failed')
    return {'path':str(destination),'geometry':str(cloud),'vertices':len(points),'bytes':destination.stat().st_size+cloud.stat().st_size,'validated':True}

def camera(snapshot,destination):
    data=dict(snapshot);data['scale_status']=scale_status(snapshot)
    if snapshot.get('video_output'):
        sfm=Path(snapshot['video_output'])/'sfm/sfm.json'
        if sfm.is_file():data['sfm']=json.loads(sfm.read_text(encoding='utf-8'))
    atomic_json(Path(destination),data)
    if json.loads(Path(destination).read_text(encoding='utf-8'))!=data:raise ValueError('Metadata readback mismatch')
    return {'path':str(destination),'bytes':Path(destination).stat().st_size,'validated':True,'scale_status':data['scale_status']}

def gaussian(source,destination):
    import torch
    source=Path(source);destination=Path(destination)
    if source.resolve()==destination.resolve():raise ValueError('Choose a different destination')
    before=torch.load(source,map_location='cpu',weights_only=False)
    required={'means','log_scales','quats','opacity_logits','color_logits'}
    if not required.issubset(before.get('parameters',{})):raise ValueError('Unsupported native Gaussian parameters')
    shutil.copy2(source,destination);after=torch.load(destination,map_location='cpu',weights_only=False)
    for key,value in before['parameters'].items():
        if not torch.equal(value,after['parameters'][key]):raise ValueError('Gaussian serialization mismatch: '+key)
    return {'path':str(destination),'bytes':destination.stat().st_size,'validated':True,'format':'native RECON-X Torch checkpoint; external dataset references retained','parameters':list(before['parameters'])}

def main():
    p=argparse.ArgumentParser();p.add_argument('request',type=Path);a=p.parse_args();r=json.loads(a.request.read_text(encoding='utf-8'))
    kind=r['kind'];dest=Path(r['destination'])
    if dest.suffix[1:] not in FORMATS[kind]:raise ValueError('Unsupported format')
    if not dest.parent.is_dir():raise FileNotFoundError(dest.parent)
    if kind in ('mesh','cloud'):result=geometry(r['source'],dest,kind,scale_status(r['snapshot']))
    elif kind=='camera':result=camera(r['snapshot'],dest)
    elif kind=='textured_mesh':result=textured_mesh(r['source'],dest)
    elif kind=='semantic':result=semantic(r['source'],dest)
    else:result=gaussian(r['source'],dest)
    atomic_json(a.request.with_name('export_result.json'),result);print(json.dumps(result))

if __name__=='__main__':main()
