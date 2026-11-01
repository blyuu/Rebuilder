"""Validated interpreter-specific subprocess contracts for desktop runs."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Job:
    name: str
    program: Path
    arguments: tuple[str,...]
    output: Path


def interpreter(root: Path, kind: str) -> Path:
    from .app_runtime import runtime
    return runtime(root,kind)


def build_job(name: str, root: Path, manifest: Path, output: Path, *,
              voxel_size_m: float=.03, truncation_m: float=.09,
              frame_index: int=14, gaussian_dataset: Path | None=None,
              gaussian_checkpoint: Path | None=None, inverse_mesh: Path | None=None,
              inverse_regularization: float=8, pose_source: str='GT') -> Job:
    root=root.resolve();manifest=manifest.resolve();output=output.resolve()
    if not manifest.is_file():raise FileNotFoundError(f'Missing dataset manifest: {manifest}')
    if not output.parent.is_dir():raise FileNotFoundError(f'Missing output parent: {output.parent}')
    if name=='TSDF':
        if not 0<voxel_size_m<truncation_m:raise ValueError('Expected 0 < voxel size < truncation')
        return Job(name,interpreter(root,'recon'),('-m','reconx.tsdf',str(manifest),
            '--output',str(output),'--voxel-size-m',str(voxel_size_m),
            '--sdf-trunc-m',str(truncation_m),'--max-depth-m','4.0'),output)
    if name in ('Depth Anything V2','Marigold v1.1'):
        model='depth_anything_v2_small' if name.startswith('Depth') else 'marigold_v1_1'
        args=['-m','reconx.depth_infer',str(manifest),model,'--output',str(output)]
        if model=='marigold_v1_1':
            from .app_runtime import model_path
            checkpoint=Path(model_path('marigold_v1_1',str(root/'tools/marigold-depth-v1-1')))
            if not checkpoint.is_dir():raise FileNotFoundError(f'Missing Marigold checkpoint: {checkpoint}')
            args.extend(['--checkpoint-path',str(checkpoint),'--processing-resolution','384','--steps','4'])
        return Job(name,interpreter(root,'depth'),tuple(args),output)
    if name=='Semantic map':
        script=root/'scripts/run_phase7.py'
        if not (output/'model_metadata.json').is_file():
            raise FileNotFoundError('SegFormer predictions missing; run Phase 7 inference first')
        return Job(name,interpreter(root,'recon'),(str(script),'map','--output',str(output)),output)
    if name=='Gaussian render':
        if gaussian_dataset is None or not gaussian_dataset.is_file() or gaussian_checkpoint is None or not gaussian_checkpoint.is_file():
            raise FileNotFoundError('Saved Gaussian dataset and checkpoint are required')
        return Job(name,interpreter(root,'gsplat'),(str(root/'scripts/render_phase9_view.py'),
            str(gaussian_dataset),str(gaussian_checkpoint),'--frame',str(frame_index),
            '--output',str(output)),output)
    if name=='Inverse rendering':
        if inverse_mesh is None or not inverse_mesh.is_file():
            raise FileNotFoundError(f'TSDF mesh is required: {inverse_mesh}')
        if inverse_regularization<0:raise ValueError('Regularization must be nonnegative')
        return Job(name,interpreter(root,'recon'),('-m','reconx.inverse_render',str(manifest),
            str(inverse_mesh),'--output',str(output),'--regularization',str(inverse_regularization),
            '--pose-source',pose_source),output)
    raise ValueError(f'Unsupported pipeline: {name}')
