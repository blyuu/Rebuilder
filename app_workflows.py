"""Declarative stages reuse research CLI contracts without shell activation."""
from pathlib import Path
from .app_runtime import runtime


def video_stages(root:Path,video:Path,output:Path,*,stride=5,max_frames=20,overlap=10,
                 model='depth_anything_v2_small',voxel=.4,depth_trunc=35.,intrinsics=None):
    if not video.is_file() or video.suffix.lower() not in ('.mp4','.mov'):raise ValueError('Select an existing MP4/MOV video')
    if stride<1 or max_frames<2 or voxel<=0:raise ValueError('Invalid video pipeline configuration')
    calibrated=['--intrinsics-json',str(intrinsics)] if intrinsics else []
    v=runtime(root,'video');d=runtime(root,'depth');c=runtime(root,'recon')
    e=output/'extracted';s=output/'sfm/sfm.json';depth=output/'depth'
    return [('extract',v,['-m','reconx.video_input',str(video),'--output',str(e),'--stride',str(stride),'--max-frames',str(max_frames),*calibrated]),
            ('SfM',v,['-m','reconx.video_sfm',str(e),'--output',str(output/'sfm'),'--overlap',str(overlap),*calibrated]),
            ('depth',d,['-m','reconx.video_depth',str(e),str(s),'--output',str(depth),'--model',model]),
            ('multiview',c,['-m','reconx.video_multiview',str(s),str(depth),'--output',str(output/'multiview')]),
            ('TSDF',c,['-m','reconx.video_dense',str(e),str(s),str(depth),'--output',str(output/'dense'),'--voxel-length',str(voxel),'--depth-trunc',str(depth_trunc)]),
            ('semantics',d,['-m','reconx.video_semantic',str(e),str(s),str(depth),'--output',str(output/'semantic')]),
            ('summary',c,['-m','reconx.video_summary',str(output)])]


def photoreal_stages(root:Path,video:Path,output:Path,*,candidate_stride=3,max_frames=120,
                     overlap=10,model='depth_anything_v2_small',voxel=.4,depth_trunc=35.,intrinsics=None):
    """Single RGB-video workflow. Each completed stage writes durable artifacts."""
    if not video.is_file() or video.suffix.lower() not in ('.mp4','.mov'):
        raise ValueError('Select an existing MP4/MOV video')
    if candidate_stride<1 or max_frames<8 or voxel<=0:raise ValueError('Invalid photoreal workflow settings')
    v=runtime(root,'video');d=runtime(root,'depth');c=runtime(root,'recon');g=runtime(root,'gsplat')
    e=output/'extracted';s=output/'sfm/sfm.json';depth=output/'depth'
    filtered=output/'sfm/quality/filtered_sfm.json'
    calibrated=['--intrinsics-json',str(intrinsics)] if intrinsics else []
    return [('extract',v,['-m','reconx.photoreal_input',str(video),'--output',str(e),'--max-frames',str(max_frames),'--candidate-stride',str(candidate_stride)]),
            ('SfM',v,['-m','reconx.video_sfm',str(e),'--output',str(output/'sfm'),'--overlap',str(overlap),'--exhaustive',*calibrated]),
            ('depth',d,['-m','reconx.video_depth',str(e),str(s),'--output',str(depth),'--model',model]),
            ('multiview',c,['-m','reconx.video_multiview',str(s),str(depth),'--output',str(output/'multiview')]),
            ('pose quality',c,['-m','reconx.photoreal_pose_quality',str(s),'--output',str(output/'sfm/quality')]),
            ('TSDF',c,['-m','reconx.video_dense',str(e),str(filtered),str(depth),'--output',str(output/'dense'),'--voxel-length',str(voxel),'--depth-trunc',str(depth_trunc)]),
            ('RGB-only 3DGS',g,['-m','reconx.photoreal_gaussian',str(output)]),
            ('UV texture',c,['-m','reconx.photoreal_texture',str(output)]),
            ('semantics',d,['-m','reconx.video_semantic',str(e),str(filtered),str(depth),'--output',str(output/'semantic')]),
            ('summary',c,['-m','reconx.photoreal_finalize',str(output)])]
