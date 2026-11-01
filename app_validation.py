"""Opt-in real packaged GUI integration run, driven by a saved validation config."""
import json
import time
from pathlib import Path
from dataclasses import replace
import psutil
from PySide6.QtCore import QTimer
from .app_project import atomic_json
from .app_runtime import runtime


def validate(app,window,config_path):
    config=json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    output=Path(config['output']);output.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();last=start;gaps=[];state={'step':0,'waiting':False,'record_start':0,'checks':[]}
    window._validation_state=state
    if config.get('photoreal_cancel_video'):
        from .app_validation_photoreal_cancel import validate as validate_cancel
        validate_cancel(app,window,config);return
    if config.get('photoreal_video'):
        from .app_validation_photoreal_workflow import validate as validate_workflow
        validate_workflow(app,window,config);return
    if config.get('photoreal_run'):
        from .app_validation_photoreal import validate as validate_photoreal
        validate_photoreal(app,window,config);return
    if config.get('rgb_view_run'):
        from .app_validation_rgb_view import validate as validate_rgb_view
        validate_rgb_view(app,window,config);return
    if config.get('phase13_video'):
        from .app_validation_phase13 import validate as validate_video
        validate_video(app,window,config);return
    def screenshot(name):
        app.processEvents();window.plotter.render()
        path=output/(name+'.png');okay=window.grab().save(str(path))
        window.plotter.screenshot(str(output/(name+'_viewport.png')))
        return okay and path.stat().st_size>10000

    if config.get('startup_only'):
        def startup_capture():
            if config.get('manifest'):window.load_dataset(Path(config['manifest']))
            if config.get('mesh') and window.paths:
                window.paths=replace(window.paths,mesh=Path(config['mesh']));window.load_mesh()
            result={'branding':window.windowTitle(),'gui_opened':window.isVisible(),'screenshot':screenshot('startup'),
                    'mesh_visible':'Mesh' in window.actors,'executable':__import__('sys').executable,
                    'root':str(window.root),'errors':window.last_errors,'wall_s':time.perf_counter()-start}
            atomic_json(output/'startup.json',result);app.exit(0 if result['gui_opened'] and result['screenshot'] and not result['errors'] else 2)
        QTimer.singleShot(1000,startup_capture);return

    if config.get('restore_project'):
        window.restore_project(Path(config['restore_project']))
        QTimer.singleShot(1500,lambda:finish_restore())
        def finish_restore():
            # Exercise the reorganized viewer and language controls without executing jobs.
            import numpy as np
            from .rebuilder_ui import refresh
            before=window.actors['Mesh'].mapper.dataset.points.copy()
            language=window.language;window.apply_language('en',persist=False);window.apply_language('ko',persist=False);window.apply_language(language,persist=False)
            window.color_mode.setCurrentIndex(2)
            window.plotter.camera.parallel_projection=True
            window.plotter.camera.parallel_projection=False
            window.visible_checks['Mesh'].setChecked(False);window.visible_checks['Mesh'].setChecked(True)
            refresh(window)
            checks={'language_roundtrip':window.language==language,'five_inspector_sections':window.right_tabs.count()==5,
                    'branding':'Rebuilder 3D' in window.windowTitle(),'coordinate_preservation':np.array_equal(before,window.actors['Mesh'].mapper.dataset.points),
                    'export_enabled':window.export_button.isEnabled(),'scale_status':window._snapshot().get('scale_status'),
                    'canonical_inverse_method':window.inverse_method.currentText()=='lambertian_regularized'}
            result={'ui_checks':checks,'branding':window.windowTitle(),'inspector_sections':window.right_tabs.count(),'project_restored':window.project_file is not None,'jobs_executed':len(window.manager.records),
                    'mesh_visible':'Mesh' in window.actors,'screenshot':screenshot('restored'),
                    'errors':window.last_errors,'wall_s':time.perf_counter()-start}
            atomic_json(output/'restore.json',result);app.exit(0 if result['mesh_visible'] and not result['jobs_executed'] and not result['errors'] else 2)
        return

    def load_dataset():
        window.load_dataset(Path(config['manifest']))
        if not window.paths:raise ValueError('Validation dataset failed to load')
        window.paths=replace(window.paths,gaussian_dataset=Path(config['gaussian_dataset']),gaussian_checkpoint=Path(config['gaussian_checkpoint']))
        window.right_tabs.setCurrentIndex(1)
        state['checks'].append({'startup_screenshot':screenshot('startup'),'frames':len(window.scene.frames)})

    def pipeline(name):
        window.output_edit.setText(str(output/name.replace(' ','_')))
        window.pipeline_combo.setCurrentText(name)
        if name=='Depth Anything V2':window.stop_spin.setValue(2)
        else:window.stop_spin.setValue(len(window.scene.frames))
        if name=='Inverse rendering':window.inverse_mesh.setText(str(window.paths.mesh))
        window.run_pipeline()

    def video():
        window.right_tabs.setCurrentIndex(window.right_tabs.count()-1)
        window.video_input.setText(config['video']);window.video_stride.setValue(5);window.video_max_frames.setValue(20)
        window.run_video_pipeline()

    def save():
        state['checks'].append({'reconstruction_screenshot':screenshot('rgb_video')})
        window.save_project_at(output/('validated.rebuilder' if config.get('phase12_1') else 'validated.reconx.json'))

    def invalid():
        path=output/'invalid.mp4';path.write_bytes(b'not a video')
        window._submit('Invalid video',[('decode',runtime(window.root,'video'),['-m','reconx.video_input',str(path),'--output',str(output/'invalid_decode')])],output/'invalid_decode')

    def missing_model():
        code='from reconx.app_runtime import model_path; model_path("deliberately_missing_validation_asset", "missing")'
        window._submit('Missing model',[('check',runtime(window.root,'depth'),['-c',code])],output/'missing_model')

    def failure():
        window._submit('Failed reconstruction',[('tsdf',runtime(window.root,'recon'),['-m','reconx.tsdf',str(output/'missing_scene.json'),'--output',str(output/'failed')])],output/'failed')

    def resources():
        window._submit('Resource guard',[('never allocate',runtime(window.root,'recon'),['-c','raise RuntimeError("must not execute")'])],output/'resources',{'required_free_ram_bytes':psutil.virtual_memory().total*2})

    def cancel():
        pidfile=output/'cancel_child.pid'
        code='import subprocess,sys,time,pathlib; p=subprocess.Popen([sys.executable,"-c","import time;time.sleep(120)"]);pathlib.Path(sys.argv[1]).write_text(str(p.pid));time.sleep(120)'
        r=window._submit('Cancellation',[('parent',runtime(window.root,'recon'),['-c',code,str(pidfile)])],output/'cancel')
        def stop():
            if pidfile.is_file():window.manager.cancel(r['id'])
            else:QTimer.singleShot(100,stop)
        QTimer.singleShot(500,stop)

    actions=[load_dataset,lambda:pipeline('TSDF'),lambda:pipeline('Depth Anything V2'),lambda:pipeline('Gaussian render'),
             lambda:pipeline('Inverse rendering'),lambda:pipeline('RGB-D SLAM'),video,save,invalid,missing_model,failure,resources,cancel]
    expected=['completed']*6+['failed']*4+['cancelled']
    if config.get('phase12_1'):
        from .rebuilder_ui import export_at
        def imported():
            window.import_video_at(Path(config['video']))
        def verify_import():
            info=window.project_data.get('video_metadata',{})
            assert info.get('decoder_verified') and info['width']>0
            assert window.workflow.currentIndex()==0
            assert window.manager.records[-1]['name']=='Inspect video'
            state['checks'].append({'video_import_without_reconstruction':True,'metadata':info})
        def exported(kind,fmt):
            window.export_kind.setCurrentIndex(window.export_kind.findData(kind));window.export_format.setCurrentText(fmt)
            export_at(window,output/('exported_'+kind+'.'+fmt))
        def gaussian_export():
            request=output/'gaussian_request.json'
            atomic_json(request,{'kind':'gaussian','source':config['gaussian_checkpoint'],'destination':str(output/'gaussian.pt'),'snapshot':{}})
            window._submit('Gaussian export',[('native parameter roundtrip',runtime(window.root,'gsplat'),['-m','reconx.app_export',str(request)])],output)
        def invalid_export():
            window.export_kind.setCurrentIndex(0);export_at(window,output/'nonexistent/scene.ply')
        def missing_geometry():
            request=output/'missing_export.json'
            atomic_json(request,{'kind':'mesh','source':str(output/'missing.ply'),'destination':str(output/'missing_export.ply'),'snapshot':{}})
            window._submit('Missing export geometry',[('validate',runtime(window.root,'recon'),['-m','reconx.app_export',str(request)])],output/'missing_export')
        def marigold():
            window.stop_spin.setValue(1);window.pipeline_combo.setCurrentText('Marigold v1.1');window.output_edit.setText(str(output/'marigold'));window.run_pipeline()
        actions=[load_dataset,lambda:pipeline('TSDF'),lambda:pipeline('Depth Anything V2'),marigold,lambda:pipeline('Gaussian render'),
                 lambda:pipeline('Inverse rendering'),lambda:pipeline('RGB-D SLAM'),imported,verify_import,video,save,
                 lambda:exported('mesh','ply'),lambda:exported('mesh','obj'),lambda:exported('mesh','glb'),
                 lambda:exported('cloud','ply'),lambda:exported('semantic','json'),lambda:exported('camera','json'),gaussian_export,
                 invalid,missing_model,failure,resources,cancel,invalid_export,missing_geometry]
        expected=['completed']*15+['failed']*4+['cancelled']+['failed']*2


    def monitor():
        nonlocal last
        now=time.perf_counter();gaps.append(now-last);last=now
        try:
            if now-start>1200:raise TimeoutError('Packaged validation exceeded 20 minutes')
            if window.manager.active or window.manager.queue:
                QTimer.singleShot(50,monitor);return
            if state['step']<len(actions):
                actions[state['step']]();state['step']+=1
                QTimer.singleShot(100,monitor);return
            records=window.manager.records
            statuses=[r['status'] for r in records]
            child=output/'cancel_child.pid';child_alive=psutil.pid_exists(int(child.read_text())) if child.is_file() else None
            by_name={r['name']:Path(r['output']) for r in records}
            def read(path):return json.loads(path.read_text()) if path.is_file() else {}
            tsdf=read(by_name['TSDF']/'run.json');video_root=by_name['RGB video']
            sfm=read(video_root/'sfm/sfm.json');dense=read(video_root/'dense/dense_report.json');semantic=read(video_root/'semantic/semantic_report.json')
            checks={'tsdf_triangles':tsdf.get('mesh_triangles',0),'video_registered_frames':sfm.get('registered_frames',0),
                    'video_sparse_points':sfm.get('landmark_count',0),'video_triangles':dense.get('triangles',0),
                    'video_semantic_voxels':semantic.get('occupied_voxels',0),
                    'depth_predictions':len(list(by_name['Depth Anything V2'].glob('*_raw.npy'))),
                    'gaussian_render':(by_name['Gaussian render']/'last_render.json').is_file(),
                    'inverse_result':(by_name['Inverse rendering']/'results.json').is_file(),
                    'interactive_mesh_actor':'Mesh' in window.actors}
            result={'wall_s':now-start,'max_event_loop_gap_s':max(gaps),'checks':state['checks'],'jobs':records,
                    'gui_peak_rss_bytes':window.gui_peak_rss_bytes,'dataset_load_s':window.dataset_load_s,'mesh_load_s':window.mesh_load_s,
                    'actual_statuses':statuses,'expected_statuses':expected,'cancelled_child_alive':child_alive,
                    'screenshot':screenshot('failure_recovery'),'runtime_python':__import__('sys').executable,
                    'root':str(window.root),'errors':window.last_errors,
                    'artifact_checks':checks,
                    'success':statuses==expected and child_alive is False and all(bool(v) for v in checks.values())}
            atomic_json(output/'validation.json',result);app.exit(0 if result['success'] else 2)
        except Exception as exc:
            import traceback
            atomic_json(output/'validation_error.json',{'error':str(exc),'traceback':traceback.format_exc(),'step':state['step']})
            app.exit(3)
    QTimer.singleShot(100,monitor)
