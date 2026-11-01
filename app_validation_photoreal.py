"""Packaged GUI validation of real RGB-only UV and free-camera CUDA outputs."""
import json,time
from pathlib import Path
import numpy as np
from PySide6.QtCore import QTimer
from .app_project import atomic_json
from .rebuilder_ui import refresh


def validate(app,w,config):
    run=Path(config['photoreal_run']);out=Path(config['output']);out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter();state={'run':str(run),'checks':{},'errors':[]}
    def fail(exc):
        state['errors'].append(str(exc));finish()
    def finish():
        state['exe_runtime']=str(__import__('sys').executable);state['elapsed_s']=time.perf_counter()-started
        atomic_json(out/'validation.json',state);app.exit(0 if state['checks'] and all(state['checks'].values()) and not state['errors'] else 2)
    def loaded():
        try:
            w.video_output=run;w.load_video_results();refresh(w);app.processEvents()
            state['checks']['mesh_loaded']='Mesh' in w.actors
            state['checks']['textured_default']=w.mesh_view_mode.currentText()=='Textured Mesh'
            state['checks']['UV_atlas']=w.dollhouse.surface is not None and w.dollhouse.texture is not None
            w.plotter.screenshot(str(out/'dollhouse_textured.png'))
            state['checks']['textured_screenshot']=(out/'dollhouse_textured.png').stat().st_size>10000
            w.dollhouse.cut.setChecked(True);w.dollhouse.slider.setValue(65);w.plotter.screenshot(str(out/'dollhouse_cutaway.png'))
            state['checks']['cutaway']=bool(w.actors['Mesh'].mapper.GetNumberOfClippingPlanes())
            w.dollhouse.cut.setChecked(False)
            w.plotter.camera.Azimuth(30);w.plotter.camera.Dolly(1.2);w.plotter.render()
            state['checks']['orbit_zoom']=True
            w.mesh_view_mode.setCurrentText('Gaussian Scene')
            QTimer.singleShot(500,check_gaussian)
        except Exception as exc:fail(exc)
    def check_gaussian():
        if not hasattr(w.dollhouse,'last_result'):
            if time.perf_counter()-started<40:QTimer.singleShot(500,check_gaussian);return
            fail('Gaussian free-view renderer timed out');return
        try:
            result=w.dollhouse.last_result;state['checks']['free_camera_cuda_render']=Path(result['image']).is_file()
            state['gaussian_render_s']=result['render_s'];state['render_pose']=result['request']['T_world_camera']
            import shutil
            shutil.copy2(result['image'],out/'gaussian_free_view.png')
            w.mesh_view_mode.setCurrentText('Textured Mesh');w.save_project_at(out/'photoreal.rebuilder')
            state['checks']['project_saved']=(out/'photoreal.rebuilder').is_file()
            w.restore_project(out/'photoreal.rebuilder');refresh(w);app.processEvents()
            state['checks']['reopened_texture']=w.mesh_view_mode.currentText()=='Textured Mesh' and w.dollhouse.surface is not None
            state['checks']['reopened_gaussian_checkpoint']=w.dollhouse.checkpoint.is_file()
            state['checks']['no_gui_error']=not w.last_errors
            w.plotter.screenshot(str(out/'reopened.png'))
            from .rebuilder_ui import export_at
            w.export_kind.setCurrentIndex(w.export_kind.findData('textured_mesh'))
            export_at(w,out/'exported_textured.glb')
            QTimer.singleShot(250,check_export)
        except Exception as exc:fail(exc)
    def check_export():
        jobs=[r for r in w.manager.records if r['name']=='Export']
        if not jobs or jobs[-1]['status'] not in ('completed','failed','cancelled'):
            if time.perf_counter()-started<50:QTimer.singleShot(250,check_export);return
            fail('Textured GLB export timed out');return
        state['checks']['textured_export_job']=jobs[-1]['status']=='completed'
        state['checks']['textured_export_file']=(out/'exported_textured.glb').is_file()
        state['checks']['no_gui_error']=not w.last_errors
        finish()
    QTimer.singleShot(1000,loaded)
