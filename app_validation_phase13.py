"""Opt-in packaged EXE validation for an actual RGB smartphone video."""
import json
import time
from pathlib import Path
from PySide6.QtCore import QTimer
from .app_project import atomic_json,explain_error


def validate(app,window,config):
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();state={'step':'import','gaps':[],'last':started}
    def finish(result,code):
        atomic_json(out/'validation.json',result);app.exit(code)
    def poll():
        now=time.perf_counter();state['gaps'].append(now-state['last']);state['last']=now
        try:
            if now-started>900:raise TimeoutError('Phase 13 video validation exceeded 15 minutes')
            if window.manager.active or window.manager.queue:
                QTimer.singleShot(100,poll);return
            if state['step']=='import':
                window.import_video_at(Path(config['video']));state['step']='inspect';QTimer.singleShot(100,poll);return
            if state['step']=='inspect':
                metadata=window.project_data.get('video_metadata',{})
                if not metadata.get('decoder_verified'):raise ValueError('EXE could not decode source video')
                window.video_stride.setValue(config.get('stride',3));window.video_max_frames.setValue(config.get('max_frames',60))
                window.run_video_pipeline();state['step']='pipeline';QTimer.singleShot(100,poll);return
            if state['step']=='pipeline':
                state['step']='capture';QTimer.singleShot(1000,poll);return
            metadata=window.project_data.get('video_metadata',{})
            run=window.video_output;sfm=json.loads((run/'sfm/sfm.json').read_text(encoding='utf-8'))
            depth=json.loads((run/'depth/depth_report.json').read_text(encoding='utf-8'))
            dense=json.loads((run/'dense/dense_report.json').read_text(encoding='utf-8'))
            semantic=json.loads((run/'semantic/semantic_report.json').read_text(encoding='utf-8'))
            extraction=json.loads((run/'extracted/extraction.json').read_text(encoding='utf-8'))
            window.save_project_at(out/'validated.rebuilder')
            window.fit_scene()
            window.plotter.render();window.grab().save(str(out/'application.png'))
            window.plotter.screenshot(str(out/'viewer.png'))
            records=window.manager.records
            viewer={'camera_position':[[float(x) for x in row] for row in window.plotter.camera_position],
                    'actors':{name:{'visible':bool(actor.GetVisibility()),
                                    'bounds':list(actor.GetBounds())}
                              for name,actor in window.actors.items()},
                    'layers':{name:check.isChecked() for name,check in window.visible_checks.items()},
                    'fit_scene_called':True}
            checks={'decoder_verified':metadata['decoder_verified'],
                    'extracted_frames':extraction['frames_extracted'],
                    'registered_frames':sfm['registered_frames'],
                    'depth_alignments':depth['successful_alignments'],
                    'triangles':dense['triangles'],
                    'semantic_voxels':semantic['occupied_voxels'],
                    'interactive_mesh':'Mesh' in window.actors,
                    'no_gt_depth_or_pose':not sfm['depth_or_gt_pose_used'],
                    'correct_error_classification':explain_error('File "video_dense.py": No valid SfM-aligned depth','TSDF')['message']=='Depth alignment failed'}
            okay=(all(r['status']=='completed' for r in records)
                  and checks['extracted_frames']>=20 and checks['registered_frames']>=5
                  and checks['depth_alignments']>=3 and checks['triangles']>0
                  and checks['semantic_voxels']>0 and checks['interactive_mesh']
                  and checks['no_gt_depth_or_pose'] and checks['correct_error_classification'])
            result={'success':okay,'source':config['video'],'output':str(run),'checks':checks,
                    'scale_normalization':sfm.get('scale_normalization'),
                    'camera_initialization':sfm.get('camera_initialization'),
                    'sfm_fallback_trial':sfm.get('fallback_trial'),
                    'viewer':viewer,
                    'jobs':records,'gui_max_event_loop_gap_s':max(state['gaps']),
                    'wall_s':now-started,'errors':window.last_errors}
            finish(result,0 if okay else 2)
        except Exception as exc:
            import traceback
            atomic_json(out/'validation_error.json',{'message':str(exc),'traceback':traceback.format_exc(),
                'state':state['step'],'jobs':window.manager.records,'window_errors':window.last_errors})
            app.exit(3)
    QTimer.singleShot(100,poll)
