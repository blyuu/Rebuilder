"""EXE automation for the actual one-click RGB-video pipeline."""
import json,time
from pathlib import Path
from PySide6.QtCore import QTimer
from .app_project import atomic_json


def validate(app,w,config):
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True);start=time.perf_counter();record={}
    def finish():
        record['wall_s']=time.perf_counter()-start;record['executable']=__import__('sys').executable
        atomic_json(out/'workflow_validation.json',record)
        app.exit(0 if record.get('status')=='completed' and record.get('textured_glb') and record.get('scene_checkpoint') else 2)
    def poll():
        jobs=[r for r in w.manager.records if r['name']=='RGB video']
        if jobs and jobs[-1]['status'] in ('completed','failed','cancelled'):
            job=jobs[-1];record.update(status=job['status'],job=job,run=job['output'])
            root=Path(job['output']);record['textured_glb']=(root/'photoreal/texture/textured.glb').is_file()
            record['scene_checkpoint']=(root/'photoreal/trained/scene.pt').is_file()
            record['mesh']=(root/'dense/rgb_only_sfm_tsdf_mesh.ply').is_file()
            record['gui_mesh_loaded']='Mesh' in w.actors
            finish();return
        QTimer.singleShot(500,poll)
    def start_job():
        try:
            w.video_input.setText(str(Path(config['photoreal_video'])))
            w.video_max_frames.setValue(int(config.get('max_frames',120)))
            w.video_stride.setValue(int(config.get('candidate_stride',3)))
            w.run_video_pipeline();poll()
        except Exception as exc:record.update(status='validation_error',error=str(exc));finish()
    QTimer.singleShot(1000,start_job)
