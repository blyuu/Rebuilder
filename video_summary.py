"""Collect independently generated Phase 11 artifacts into an auditable run summary."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(root:Path)->dict:
    root=Path(root)
    def read(path):return json.loads((root/path).read_text(encoding='utf-8'))
    extraction=read('extracted/extraction.json')
    sfm=read('sfm/sfm.json')
    depth=read('depth/depth_report.json')
    dense=read('dense/dense_report.json') if (root/'dense/dense_report.json').is_file() else read('dense_uniform/dense_report.json')
    semantic=read('semantic/semantic_report.json')
    multi=read('multiview/multiview.json') if (root/'multiview/multiview.json').is_file() else None
    if not dense['triangles'] and (root/'dense_uniform/dense_report.json').is_file():
        dense=read('dense_uniform/dense_report.json')
    evaluation=read('evaluation/evaluation.json') if (root/'evaluation/evaluation.json').is_file() else None
    source=read('source.json') if (root/'source.json').is_file() else None
    summary={'input_video':extraction['source_video'],'source_description':source['source'] if source else 'external video; provenance unspecified',
        'source_camera_type':source['camera_type'] if source else 'unknown',
        'rgb_only_reconstruction':bool(not sfm['depth_or_gt_pose_used']),
        'reconstruction_unit':'arbitrary SfM units; not meters',
        'input_video_frames':extraction['video_metadata']['reported_frames'],
        'extracted_frames':extraction['frames_extracted'],
        'registered_frames':sfm['registered_frames'],'registration_rate':sfm['registered_frames']/sfm['input_frames'],
        'sparse_landmarks':sfm['landmark_count'],'depth_aligned_frames':depth['successful_alignments'],
        'tsdf_frames':dense['frames_integrated'],'mesh_triangles':dense['triangles'],
        'dense_point_count':dense['points'],'semantic_voxels':semantic['occupied_voxels'],
        'input_observation_support_proxy':dense.get('input_observation_support_proxy'),
        'multiview_median_relative_residual':multi['median_pair_relative_residual'] if multi else None,
        'sfm_operation_times_s':sfm['timing'],'depth_inference_s':depth['total_inference_s'],
        'depth_peak_gpu_allocated_bytes':depth['peak_gpu_allocated_bytes'],
        'tsdf_fusion_s':dense['fusion_s'],'semantic_inference_s':semantic['total_inference_s'],
        'semantic_peak_gpu_allocated_bytes':semantic['peak_gpu_allocated_bytes'],
        'reference_evaluation_only':evaluation is not None,
        'ate_sim3_rmse_m_evaluation_only':evaluation['ate_sim3_rmse_m'] if evaluation else None,
        'artifacts':{'extraction':'extracted/extraction.json','sfm':'sfm/sfm.json',
            'depth':'depth/depth_report.json','dense':str(Path(dense['mesh']).parent/'dense_report.json'),
            'semantic':'semantic/semantic_report.json','mesh':dense['mesh'],
            'semantic_cloud':'semantic/semantic_points.ply'}}
    (root/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    print(json.dumps(summarize(a.output),indent=2))


if __name__=='__main__':main()
