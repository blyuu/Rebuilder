"""Packaged cancellation check; uses actual decoder/SfM child processes."""
import time
from pathlib import Path
from PySide6.QtCore import QTimer
from .app_project import atomic_json

def validate(app,w,config):
    out=Path(config['output']);out.mkdir(parents=True,exist_ok=True);result={};start=time.perf_counter()
    def poll():
        jobs=[j for j in w.manager.records if j['name']=='RGB video']
        if jobs and jobs[-1]['status']=='cancelled':
            r=jobs[-1];result.update(status=r['status'],stage_results=r['stage_results'],run=r['output'],
              process_stopped=w.manager.process is None,elapsed_s=time.perf_counter()-start)
            atomic_json(out/'cancel_validation.json',result);app.exit(0 if result['process_stopped'] else 2);return
        if time.perf_counter()-start>40:
            result['status']='timeout';atomic_json(out/'cancel_validation.json',result);app.exit(2);return
        QTimer.singleShot(250,poll)
    def begin():
        w.video_input.setText(str(config['photoreal_cancel_video']));w.video_max_frames.setValue(60)
        w.run_video_pipeline();QTimer.singleShot(4000,w.cancel_pipeline);poll()
    QTimer.singleShot(1000,begin)
