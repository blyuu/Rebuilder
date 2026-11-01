"""Packaged GUI readback of a completed real RGB-video reconstruction."""
import json
from pathlib import Path
import numpy as np
from PySide6.QtCore import QTimer
from .app_project import atomic_json


def validate(app,window,config):
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    def run():
        try:
            window.video_output=Path(config['rgb_view_run']).resolve()
            window.project_data['input_type']='rgb_video'
            window.load_video_results()
            from .rebuilder_ui import refresh
            refresh(window)
            app.processEvents()
            mesh=window.actors['Mesh'].mapper.dataset
            original=np.asarray(mesh.point_data['RGB']).copy()
            checks={'mesh_has_captured_rgb':len(original)==mesh.n_points and len(np.unique(original,axis=0))>10,
                    'semantic_separate_actor':'Semantics' in window.actors,
                    'solid_default':window.mesh_view_mode.currentText()=='Solid',
                    'partial_coverage_labeled':('부분 형상' in window.scene_summary.text() or
                                                'Partial geometry' in window.scene_summary.text()),
                    'mesh_source':str(window.video_output/'dense/rgb_only_sfm_tsdf_mesh.ply')}
            screenshots={}
            for mode,filename in [('Solid','solid'),('Wireframe','wireframe'),
                                  ('RGB Vertex Color','rgb_vertex'),('Semantic Color','semantic')]:
                window.mesh_view_mode.setCurrentText(mode)
                app.processEvents();window.plotter.render()
                path=output/(filename+'.png');window.plotter.screenshot(str(path))
                screenshots[mode]=str(path)
                checks[filename+'_screenshot']=path.is_file() and path.stat().st_size>10000
                checks[filename+'_mode']=window.mesh_view_mode.currentText()==mode
            window.mesh_view_mode.setCurrentText('RGB Vertex Color')
            shown=window.actors['Mesh'].mapper.dataset
            checks['rgb_actor_scalars']=bool(window.actors['Mesh'].mapper.GetScalarVisibility())
            checks['rgb_unchanged']=np.array_equal(np.asarray(shown.point_data['RGB']),original)
            checks['rgb_and_semantic_distinct']=bool(window.actors['Mesh'].GetVisibility() and not window.actors['Semantics'].GetVisibility())
            before=np.asarray(window.plotter.camera_position[0],float)
            window.plotter.camera.Azimuth(35);window.plotter.render()
            after=np.asarray(window.plotter.camera_position[0],float)
            checks['camera_orbit']=bool(np.linalg.norm(after-before)>1e-4)
            camera=window.plotter.camera
            distance_before=float(camera.GetDistance());camera.Dolly(1.2);window.plotter.render()
            checks['camera_zoom']=bool(camera.GetDistance()<distance_before)
            position_before=np.asarray(camera.GetPosition());focal_before=np.asarray(camera.GetFocalPoint())
            delta=np.array([0.4,0.,0.]);camera.SetPosition(*(position_before+delta));camera.SetFocalPoint(*(focal_before+delta));window.plotter.render()
            checks['camera_pan']=bool(np.allclose(np.asarray(camera.GetPosition())-position_before,delta) and
                                      np.allclose(np.asarray(camera.GetFocalPoint())-focal_before,delta))
            window.fit_scene()
            checks['fit_scene']=bool(np.isfinite(np.asarray([list(row) for row in window.plotter.camera_position],dtype=float)).all())
            checks['arbitrary_sfm_scale_labeled']=window._snapshot()['scale_convention']=='arbitrary_sfm'
            window.right_tabs.setCurrentIndex(1)
            app.processEvents()
            window.grab().save(str(output/'application.png'))
            result={'success':all(v for k,v in checks.items() if isinstance(v,bool)),
                    'checks':checks,'screenshots':screenshots,'camera_before':before.tolist(),
                    'camera_after':after.tolist(),'errors':window.last_errors,
                    'runtime_python':__import__('sys').executable}
            atomic_json(output/'validation.json',result)
            app.exit(0 if result['success'] and not result['errors'] else 2)
        except Exception as exc:
            import traceback
            atomic_json(output/'validation_error.json',{'error':str(exc),'traceback':traceback.format_exc()})
            app.exit(3)
    QTimer.singleShot(700,run)
