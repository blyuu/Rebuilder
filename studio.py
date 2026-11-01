"""Production desktop controller layered over the established interactive viewer."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from dataclasses import replace
from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QApplication,QWidget,QVBoxLayout,QLabel,QPushButton,QTextEdit,QFileDialog,QMessageBox,QComboBox,QSpinBox
from .desktop import MainWindow
from .desktop_data import read_metrics,slice_manifest
from .desktop_jobs import build_job
from .app_runtime import VERSION,install_root,runtime,inventory,read_config,resolve
from .app_project import atomic_json,save_project,load_project,explain_error
from .app_jobs import JobManager
from .app_workflows import video_stages,photoreal_stages
from .ui_appearance import QComboBox,apply_theme,localize,read_language,save_language,tr


class StudioWindow(MainWindow):
    def __init__(self,root=None,manifest=None,workspace=None):
        root=Path(root or install_root()).resolve()
        self.workspace=Path(workspace or os.environ.get('RECONX_WORKSPACE',Path.home()/'Documents/RECON-X')).resolve()
        self.workspace.mkdir(parents=True,exist_ok=True)
        self.project_file=None;self.project_data={};self.last_errors=[];self.job_callbacks={};self._inverse_output=None;self.project_job_offset=0
        self.log_file=self.workspace/'application.log'
        self.language=read_language()
        super().__init__(root,manifest or root/'no-default-scene.json')
        self.history=[]
        self.setWindowTitle(f'Rebuilder 3D {VERSION}')
        self.console.document().setMaximumBlockCount(1500)
        self.manager=JobManager(root,self.workspace/'jobs',self)
        self.manager.message.connect(self.log);self.manager.changed.connect(self._job_changed);self.manager.finished.connect(self._finished)
        self.output_edit.setText(str(self.workspace/'runs/tsdf'))
        self.home=self._home_page();self.right_tabs.insertTab(0,self.home,'Home');self.right_tabs.setCurrentIndex(0)
        self.right_tabs.setMinimumWidth(380)
        self.pipeline_combo.removeItem(self.pipeline_combo.findText('Semantic map')) # fixed-scene Phase 7 script is not a generic workflow
        self.pipeline_combo.addItem('RGB-D SLAM')
        self.pipeline_combo.addItem('Gaussian train')
        self.train_iterations=QSpinBox();self.train_iterations.setRange(1,10000);self.train_iterations.setValue(400)
        self.train_gaussians=QSpinBox();self.train_gaussians.setRange(100,100000);self.train_gaussians.setValue(12000)
        page=self.gaussian_info.parentWidget().layout()
        page.addWidget(QLabel('RGB-D-assisted initialization; fixed 160 px images'))
        page.addWidget(QLabel('Training iterations'));page.addWidget(self.train_iterations)
        page.addWidget(QLabel('Maximum Gaussians'));page.addWidget(self.train_gaussians)
        button=QPushButton('Train Gaussian scene');button.clicked.connect(self.train_gaussian);page.addWidget(button)
        menu=self.menuBar().addMenu('Application')
        for title,fn in [('New project',self.new_project),('Open project',self.open_project_dialog),('Save project',self.save_project_dialog),
                         ('Import RGB video results',self.open_video_results),('Export diagnostics',self.export_diagnostics)]:
            action=QAction(title,self);action.triggered.connect(fn);menu.addAction(action)
        self.refresh_assets()
        self.log('Ready. RGB-D uses meters; RGB-only SfM uses arbitrary scale. Accuracy requires reference data.')
        settings=self.menuBar().addMenu('Settings')
        action=QAction('Appearance & language…',self);action.triggered.connect(self.open_settings);settings.addAction(action)
        from .rebuilder_ui import install
        install(self)
        apply_theme(self);self.apply_language(self.language,persist=False)

    def apply_language(self,language,persist=True):
        self.language=language
        if persist:save_language(language)
        localize(self,language);self.refresh_assets()
        if hasattr(self,'manager'):
            self.job_status.setPlainText('\n'.join(f'{tr(r["name"],language)}: {tr(r["status"],language)} ({int(r["progress"]*100)}%)' for r in self.manager.records[-12:]))
        if not getattr(self,'video_output',None):
            self.video_info.setPlainText(tr('Select a video. SfM, relative depth, TSDF and semantics run from RGB only. Camera intrinsics are estimated unless separately calibrated.',language))

    def open_settings(self):
        from PySide6.QtWidgets import QDialog,QDialogButtonBox
        dialog=QDialog(self);dialog.setWindowTitle(tr('Appearance & language',self.language));dialog.setMinimumWidth(440)
        layout=QVBoxLayout(dialog)
        title=QLabel('Rebuilder 3D · Midnight');title.setStyleSheet('font-size:18px;color:#c2f78d;padding:8px;');layout.addWidget(title)
        label=QLabel(tr('Language',self.language));layout.addWidget(label)
        languages=QComboBox();languages.addItem('English','en');languages.addItem('한국어','ko')
        languages.setCurrentIndex(1 if self.language=='ko' else 0);layout.addWidget(languages)
        note=QLabel();note.setWordWrap(True);layout.addWidget(note)
        buttons=QDialogButtonBox(QDialogButtonBox.StandardButton.Close);buttons.rejected.connect(dialog.close);layout.addWidget(buttons)
        def changed():
            self.apply_language(languages.currentData())
            dialog.setWindowTitle(tr('Appearance & language',self.language));label.setText(tr('Language',self.language))
            buttons.button(QDialogButtonBox.StandardButton.Close).setText(tr('Close',self.language))
            note.setText('선택 즉시 적용되며 다음 실행에도 유지됩니다. 모델 이름·지표 키·실행 로그는 원문을 유지합니다.' if self.language=='ko' else 'Applied immediately and remembered on restart. Model names, metric keys and diagnostic logs retain their original form.')
        languages.currentIndexChanged.connect(changed)
        note.setText('언어 선택은 즉시 적용되고 자동 저장됩니다.' if self.language=='ko' else 'Language changes apply immediately and are saved automatically.')
        buttons.button(QDialogButtonBox.StandardButton.Close).setText(tr('Close',self.language));dialog.exec()

    def _home_page(self):
        page=QWidget();layout=QVBoxLayout(page)
        title=QLabel(f'Rebuilder 3D {VERSION}\nFrom 2D Images to 3D Worlds');title.setStyleSheet('font-size:22px;font-weight:600;color:#8fd7ff');layout.addWidget(title)
        for label,callback in [('New project',self.new_project),('Open project…',self.open_project_dialog),('Save project…',self.save_project_dialog)]:
            button=QPushButton(label);button.clicked.connect(callback);layout.addWidget(button)
        layout.addWidget(QLabel('Inputs: TUM RGB-D / MP4, MOV / RECON-X JSON / PLY\nRGB-only reconstruction has arbitrary SfM scale.'))
        self.recent=QComboBox();layout.addWidget(self.recent)
        button=QPushButton('Open recent project');button.clicked.connect(lambda:self.restore_project(Path(self.recent.currentText())) if self.recent.currentText() else None);layout.addWidget(button)
        self.asset_model=QComboBox();self.asset_model.addItems(['depth_anything_v2_small','segformer','marigold_v1_1','rtabmap']);layout.addWidget(self.asset_model)
        button=QPushButton('Configure existing model / RTAB-Map…');button.clicked.connect(self.configure_asset);layout.addWidget(button)
        self.asset_status=QTextEdit();self.asset_status.setReadOnly(True);layout.addWidget(self.asset_status)
        self.job_status=QTextEdit();self.job_status.setReadOnly(True);layout.addWidget(self.job_status)
        self._refresh_recent();return page

    def _refresh_recent(self):
        path=self.workspace/'recent.json'
        try:recent=json.loads(path.read_text()) if path.is_file() else []
        except (OSError,ValueError):recent=[]
        self.recent.clear();self.recent.addItems(recent[:10])

    def refresh_assets(self):
        info=inventory(self.root)
        t=lambda s:tr(s,self.language)
        lines=[t('RUNTIMES')]+[f'{key}: {t("Ready" if value["available"] else "Missing")}' for key,value in info['runtimes'].items()]
        lines+=['',t('MODEL ASSETS')]+[f'{key}: {t("Configured" if value["available"] else "Select model directory")}' for key,value in info['models'].items()]
        lines+=['',t('Weights are not downloaded automatically.'),t('Logs and projects: ')+str(self.workspace)]
        self.asset_status.setPlainText('\n'.join(lines));self.asset_status.setToolTip(json.dumps(info,indent=2))

    def configure_asset(self):
        name=self.asset_model.currentText()
        if name=='rtabmap':path,_=QFileDialog.getOpenFileName(self,'Select rtabmap-rgbd_dataset.exe',str(self.root),'Executable (*.exe)')
        else:path=QFileDialog.getExistingDirectory(self,'Select complete local model directory',str(Path.home()))
        if not path:return
        dest=self.root/'config/local.json'
        data=read_metrics(dest) if dest.is_file() else {}
        data.setdefault('external' if name=='rtabmap' else 'models',{})[name]=path
        try:atomic_json(dest,data);self.refresh_assets()
        except Exception as exc:self._error('Configuration could not be saved',exc)

    def log(self,message):
        super().log(message)
        with self.log_file.open('a',encoding='utf-8') as log:log.write(f'{time.strftime("%Y-%m-%dT%H:%M:%S")} {message}\n')

    def _error(self,title,error):
        diagnostic=(dict(error) if isinstance(error,dict) else explain_error(str(error)))
        diagnostic['title']=title;diagnostic['log']=str(self.log_file)
        self.last_errors.append(diagnostic);self.log(json.dumps(diagnostic))
        if sys.exc_info()[0]:
            with self.log_file.open('a',encoding='utf-8') as f:f.write(traceback.format_exc())
        box=QMessageBox(self);box.setWindowTitle(tr(title,self.language));box.setText(tr(diagnostic['message'],self.language))
        box.setInformativeText(tr(diagnostic['recovery'],self.language)+('\n로그: ' if self.language=='ko' else '\nLog: ')+str(self.log_file));box.setDetailedText(str(error))
        box.setIcon(QMessageBox.Icon.Warning);box.setModal(False);box.show()

    def new_project(self):
        if self.manager.active or self.manager.queue:self._error('Job running','Finish or cancel pending jobs first');return
        if hasattr(self,'dollhouse'):
            self.dollhouse.stop();self.dollhouse.surface=None
        self.project_file=None;self.project_data={};self.paths=None;self.scene=None
        self.project_job_offset=len(self.manager.records)
        self.plotter.clear();self.plotter.add_axes();self.actors.clear();self.dataset_tree.clear()
        self.video_input.clear();self._inverse_output=None;self.video_view=False
        self.video_coverage_note=None
        if hasattr(self,'input_summary'):self.input_summary.setText('No input imported')
        self.inspector.setPlainText('No reconstruction results')
        for name in ('depth_image','gaussian_image','inverse_image'):
            widget=getattr(self,name);widget.clear();widget.setText('No output for this project')
        for name in ('semantic_info','gaussian_info','depth_info','inverse_info'):
            getattr(self,name).setPlainText('No output for this project')
        self.load_evaluation()
        if hasattr(self,'video_output'):del self.video_output
        self.right_tabs.setCurrentIndex(0);self.log('New empty project')

    def open_dataset(self):
        directory=QFileDialog.getExistingDirectory(self,'Select TUM dataset / scene.json directory',str(self.workspace))
        if not directory:return
        folder=Path(directory);manifest=folder/'scene.json'
        try:
            if not manifest.is_file() and all((folder/name).is_file() for name in ('rgb.txt','depth.txt','groundtruth.txt')):
                from .tum import make_tum_manifest
                manifest=self.workspace/'imports'/folder.name/'scene.json';make_tum_manifest(folder,manifest)
            if not manifest.is_file():raise FileNotFoundError('Choose a TUM dataset or a folder containing scene.json')
            self.load_dataset(manifest)
        except Exception as exc:self._error('Dataset import failed',exc)

    def open_mesh(self):
        path,_=QFileDialog.getOpenFileName(self,'Open mesh PLY',str(self.workspace),'PLY (*.ply)')
        if not path:return
        try:
            if self.paths:self.paths=replace(self.paths,mesh=Path(path));self.load_mesh()
            else:
                import pyvista as pv
                mesh=pv.read(path)
                if not mesh.n_cells:raise ValueError('Mesh is empty')
                self._set_actor('Mesh',mesh,color='#9db8c9');self.fit_scene()
                self.project_data.update(input_type='artifacts',mesh=path,scale_convention='unknown')
                self.viewport_status.setText('Imported mesh • units unspecified by PLY')
        except Exception as exc:self._error('Mesh import failed',exc)

    def open_cloud(self):
        path,_=QFileDialog.getOpenFileName(self,'Open point cloud PLY',str(self.workspace),'PLY (*.ply)')
        if not path:return
        try:
            if self.paths:self.paths=replace(self.paths,cloud=Path(path));self.load_cloud()
            else:
                import pyvista as pv
                cloud=pv.read(path)
                if not cloud.n_points:raise ValueError('Point cloud is empty')
                self._set_actor('Cloud',cloud,style='points',point_size=3,color='#7ccab3');self.fit_scene()
                self.project_data.update(input_type='artifacts',cloud=path,scale_convention='unknown')
        except Exception as exc:self._error('Cloud import failed',exc)

    def _snapshot(self):
        video=(self.project_data.get('input_type')=='rgb_video' or bool(getattr(self,'video_view',False)) or
               (bool(self.video_input.text().strip()) and hasattr(self,'workflow') and self.workflow.currentIndex()==0))
        data={'input_type':'rgb_video' if video else 'rgbd','workflow':'RGB video' if video else self.pipeline_combo.currentText(),
              'scale_convention':'arbitrary_sfm' if video else 'meters',
              'parameters':{'voxel':self.voxel_spin.value(),'truncation':self.trunc_spin.value(),
                            'video_stride':self.video_stride.value(),'video_max_frames':self.video_max_frames.value(),
                            'video_overlap':self.video_overlap.value(),'video_model':self.video_depth_model.currentIndex(),
                            'video_voxel':self.video_voxel.value(),'video_depth_trunc':self.video_depth_trunc.value(),
                            'frame_start':self.start_spin.value(),'frame_stop':self.stop_spin.value(),
                            'inverse_regularization':self.inverse_reg.value(),'inverse_pose':self.inverse_pose.currentIndex(),
                            'gaussian_frame':self.gaussian_frame.value()},
              'output':self.output_edit.text(),'execution_status':self.manager.records[-1]['status'] if self.manager.records else 'loaded',
              'jobs':self.project_data.get('jobs',[])+self.manager.records[self.project_job_offset:],'models':inventory(self.root)['models'],
              'camera_state':[[float(x) for x in row] for row in self.plotter.camera_position],
              'visible':{key:check.isChecked() for key,check in self.visible_checks.items()}}
        if self.paths:
            for key in ('manifest','mesh','cloud','slam_manifest','semantic_map','gaussian_checkpoint','gaussian_dataset'):
                value=getattr(self.paths,key);data[key]=str(value) if value else None
            original=read_metrics(self.paths.manifest);data['calibration']=original['intrinsics']
            data['camera_poses']=[{'timestamp_s':f['timestamp_s'],'T_world_camera':f['T_world_camera']} for f in original['frames']]
            data['metrics']=dict(self.project_data.get('metrics',{}))
            for record in self.manager.records[self.project_job_offset:]:
                if record['status']=='completed':
                    for filename in ('run.json','results.json','last_render.json'):
                        result=Path(record['output'])/filename
                        if result.is_file():data['metrics'][record['name']]=read_metrics(result)
        if video:
            data['video']=self.video_input.text();data['intrinsics']=self.video_intrinsics.text() or None
            if hasattr(self,'video_output'):
                data['video_output']=str(self.video_output)
                if (self.video_output/'photoreal/trained/scene.pt').is_file():
                    data['gaussian_checkpoint']=str(self.video_output/'photoreal/trained/scene.pt')
                    data['gaussian_dataset']=str(self.video_output/'photoreal/dataset/dataset.json')
                if (self.video_output/'photoreal/texture/textured.glb').is_file():
                    data['textured_glb']=str(self.video_output/'photoreal/texture/textured.glb')
                if hasattr(self,'mesh_view_mode'):data['mesh_view_mode']=self.mesh_view_mode.currentText()
                sfm=self.video_output/'sfm/sfm.json'
                if sfm.is_file():
                    camera_data=read_metrics(sfm);data['calibration']=camera_data.get('cameras',{})
                    data['camera_poses']=[{'timestamp_s':f.get('timestamp_s'),'T_world_camera':f['T_world_camera']} for f in camera_data['frames']]
                data['scale_status']='arbitrary_sfm_scale'
                summary=self.video_output/'summary.json'
                if summary.is_file():data['metrics']=read_metrics(summary)
        inverse_path=Path(self.inverse_mesh.text()) if self.inverse_mesh.text().strip() else None
        data['inverse_mesh']=str(inverse_path) if inverse_path and inverse_path.is_file() else None
        data['inverse_output']=str(self._inverse_output) if self._inverse_output else None
        if not self.paths and self.project_data.get('input_type')=='artifacts':
            data.update(input_type='artifacts',scale_convention='unknown')
            for key in ('mesh','cloud'):
                if self.project_data.get(key):data[key]=self.project_data[key]
        return data

    def save_project_at(self,path):
        save_project(Path(path),self._snapshot());self.project_file=Path(path).resolve()
        previous=[self.recent.itemText(i) for i in range(self.recent.count())]
        atomic_json(self.workspace/'recent.json',list(dict.fromkeys([str(self.project_file)]+previous))[:10]);self._refresh_recent()
        self.log(f'Project saved: {path}')

    def save_project_dialog(self):
        path=str(self.project_file) if self.project_file else QFileDialog.getSaveFileName(self,'Save Rebuilder 3D project',str(self.workspace/'project.rebuilder'),'Rebuilder project (*.rebuilder);;Legacy JSON (*.json)')[0]
        if path:
            try:self.save_project_at(path)
            except Exception as exc:self._error('Project save failed',exc)

    def save_session(self):self.save_project_dialog()
    def open_session(self):self.open_project_dialog()
    def open_project_dialog(self):
        path,_=QFileDialog.getOpenFileName(self,'Open project or legacy session',str(self.workspace),'Rebuilder project (*.rebuilder *.json)')
        if path:self.restore_project(Path(path))

    def restore_project(self,path):
        try:
            if self.manager.active or self.manager.queue:raise ValueError('Finish or cancel pending workflows before opening a project')
            data=load_project(Path(path),self.root);self.new_project();self.project_data=data;self.project_file=Path(path)
            if data.get('manifest') and Path(data['manifest']).is_file():
                self.load_dataset(Path(data['manifest']))
                if self.paths:
                    values={k:Path(data[k]) for k in ('mesh','cloud','slam_manifest','semantic_map','gaussian_checkpoint','gaussian_dataset') if data.get(k) and Path(data[k]).is_file()}
                    self.paths=replace(self.paths,**values)
                    if self.paths.mesh:self.load_mesh()
            p=data.get('parameters',data)
            for key,widget in [('voxel',self.voxel_spin),('truncation',self.trunc_spin),('video_stride',self.video_stride),
                               ('video_max_frames',self.video_max_frames),('video_overlap',self.video_overlap),('video_voxel',self.video_voxel),('video_depth_trunc',self.video_depth_trunc),
                               ('frame_start',self.start_spin),('frame_stop',self.stop_spin)]:
                if key in p:widget.setValue(p[key])
            self.video_depth_model.setCurrentIndex(p.get('video_model',0))
            self.inverse_reg.setValue(p.get('inverse_regularization',8));self.inverse_pose.setCurrentIndex(p.get('inverse_pose',0))
            self.gaussian_frame.setValue(p.get('gaussian_frame',14))
            self.pipeline_combo.setCurrentText(data.get('workflow','TSDF'))
            if data.get('output'):self.output_edit.setText(data['output'])
            if data.get('inverse_mesh'):self.inverse_mesh.setText(data['inverse_mesh'])
            self._inverse_output=Path(data['inverse_output']) if data.get('inverse_output') else None
            if data.get('video'):self.video_input.setText(data['video'])
            if data.get('intrinsics'):self.video_intrinsics.setText(data['intrinsics'])
            if data.get('video_output') and Path(data['video_output']).is_dir():
                self.video_output=Path(data['video_output']);self.load_video_results();self.right_tabs.setCurrentIndex(0)
                if data.get('mesh_view_mode') and hasattr(self,'mesh_view_mode'):
                    self.mesh_view_mode.setCurrentText(data['mesh_view_mode'])
            if data.get('input_type')=='artifacts':
                import pyvista as pv
                for key,label in [('mesh','Mesh'),('cloud','Cloud')]:
                    if data.get(key) and Path(data[key]).is_file():self._set_actor(label,pv.read(data[key]),color='#9db8c9')
                self.fit_scene()
            for key,value in data.get('visible',{}).items():
                if key in self.visible_checks:self.visible_checks[key].setChecked(value)
            if data.get('camera_state'):self.plotter.camera_position=data['camera_state']
            self.load_evaluation()
            if data['missing_sources']:self._error('Project opened with missing references',json.dumps(data['missing_sources']))
            self.log(f'Opened project without recomputation: {path}')
        except Exception as exc:self._error('Project load failed',exc)

    def open_video_results(self):
        directory=QFileDialog.getExistingDirectory(self,'Choose completed RGB-video run',str(self.workspace))
        if directory:
            self.video_output=Path(directory);self.project_data['input_type']='rgb_video';self.load_video_results()

    def load_video_results(self):
        super().load_video_results()
        meta=self.video_output/'extracted/extraction.json'
        if hasattr(self,'input_summary') and meta.is_file():
            info=read_metrics(meta);v=info['video_metadata']
            self.input_summary.setText(f'{Path(info["source_video"]).name}\n{v["width"]} × {v["height"]} · {v["duration_s_nominal"]:.2f} s\nRGB Video · extracted: {info["frames_extracted"]} frames\nCalibration: estimated unless supplied')
        self.inspector.setPlainText(('RGB 영상에서 추정한 기하 구조입니다.\n절대 스케일·기하 정확도는 검증되지 않았습니다.\n\n결과: ' if self.language=='ko' else 'Geometry estimated from RGB video.\nAbsolute scale and geometric accuracy are not validated.\n\nResults: ')+str(self.video_output))
        self.project_data['input_type']='rgb_video'
        if hasattr(self,'workflow'):self.workflow.setCurrentIndex(0)
        self.video_view=True;self.color_mode.setItemText(1,'Height (arbitrary SfM units)')
        if hasattr(self,'mesh_view_mode'):self.mesh_view_mode.setCurrentText('Solid');self.apply_mesh_view_mode()
        sfm=read_metrics(self.video_output/'sfm/sfm.json')
        duration=float(info['video_metadata']['duration_s_nominal']) if meta.is_file() else 0.
        poses=sfm['frames'];span=poses[-1]['timestamp_s']-poses[0]['timestamp_s'] if len(poses)>1 else 0.
        gap=max((b['timestamp_s']-a['timestamp_s'] for a,b in zip(poses,poses[1:])),default=0.)
        coverage=len(poses)/max(1,sfm['input_frames'])
        self.video_coverage_note=(f'Registered cameras: {len(poses)}/{sfm["input_frames"]} '
            f'({coverage:.0%}); temporal span: {span:.1f}/{duration:.1f} s; largest gap: {gap:.1f} s. '
            'Partial geometry only; complete room model not established.')
        if self.language=='ko':self.video_coverage_note=(f'등록 카메라: {len(poses)}/{sfm["input_frames"]} '
            f'({coverage:.0%}); 시간 범위: {span:.1f}/{duration:.1f}초; 최대 추적 공백: {gap:.1f}초. '
            '부분 형상만 확인됨 · 방 전체 모델은 확인되지 않았습니다.')
        if hasattr(self,'dollhouse'):self.dollhouse.load(self.video_output)
        self.viewport_status.setText('RGB-only scene • arbitrary SfM scale • orbit / pan / zoom / right-click query')
        self.visible_checks['SLAM path'].setText('SfM path')
        localize(self,self.language)

    def load_dataset(self,manifest):
        if hasattr(self,'dollhouse'):
            self.dollhouse.stop();self.dollhouse.surface=None
        self.video_view=False
        self.video_coverage_note=None
        super().load_dataset(manifest)
        self.color_mode.setItemText(1,'Height (meters)');self.visible_checks['SLAM path'].setText('SLAM path')
        self.project_data['input_type']='rgbd'
        if hasattr(self,'workflow'):
            self.workflow.setCurrentIndex(1)
            self.input_summary.setText(f'RGB-D · {len(self.scene.frames)} frames\n{manifest}\nCalibration: manifest intrinsics · meters')
        localize(self,self.language)

    def _refresh_color(self):
        if self.color_mode.currentIndex()==2:
            for name in ('Mesh','Cloud'):
                actor=self.actors.get(name)
                if actor:
                    mesh=actor.mapper.dataset.copy()
                    color=next((key for key in ('RGB','RGBA','red') if key in mesh.point_data and mesh.point_data[key].ndim==2),None)
                    if color:self._set_actor(name,mesh,scalars=color,rgb=True,show_scalar_bar=False,**({'style':'points','point_size':self.point_size.value()} if name=='Cloud' else {}))
            return
        if getattr(self,'video_view',False):
            actor=self.actors.get('Mesh')
            if actor:
                mesh=actor.mapper.dataset.copy()
                if self.color_mode.currentIndex()==1:
                    mesh['height_sfm']=mesh.points[:,2];self._set_actor('Mesh',mesh,scalars='height_sfm',cmap='viridis')
                else:self._set_actor('Mesh',mesh,color='#9db8c9',show_edges=False)
        else:super()._refresh_color()

    def apply_mesh_view_mode(self):
        """Render actual RGB samples separately from class-label colors."""
        if not hasattr(self,'mesh_view_mode'):return
        mode=self.mesh_view_mode.currentText()
        if hasattr(self,'dollhouse') and self.dollhouse.mode(mode):return
        actor=self.actors.get('Mesh')
        if actor is None:return
        mesh=actor.mapper.dataset.copy()
        camera=self.plotter.camera_position
        if mode=='Semantic Color':
            semantic=self.actors.get('Semantics')
            if semantic is None:
                self.viewport_status.setText('Semantic map unavailable for this scene')
                self.mesh_view_mode.blockSignals(True);self.mesh_view_mode.setCurrentText('Solid');self.mesh_view_mode.blockSignals(False)
                mode='Solid'
            else:
                actor.SetVisibility(False);semantic.SetVisibility(True)
                self.viewport_status.setText('Semantic class colors (labels), not captured RGB')
                self.plotter.render();return
        if mode=='RGB Vertex Color':
            rgb=mesh.point_data.get('RGB')
            if rgb is None or len(rgb)!=mesh.n_points or rgb.ndim!=2 or rgb.shape[1]<3:
                self.viewport_status.setText('No RGB vertex colors stored in this mesh')
                self.mesh_view_mode.blockSignals(True);self.mesh_view_mode.setCurrentText('Solid');self.mesh_view_mode.blockSignals(False)
                mode='Solid'
            else:
                self._set_actor('Mesh',mesh,scalars='RGB',rgb=True,show_scalar_bar=False)
                self.viewport_status.setText('Captured RGB vertex colors • SfM geometry accuracy unchanged')
        if mode=='Solid':self._set_actor('Mesh',mesh,color='#9db8c9',show_edges=False)
        elif mode=='Wireframe':self._set_actor('Mesh',mesh,color='#b9d6e7',style='wireframe',line_width=1)
        self.actors['Mesh'].SetVisibility(True)
        semantic=self.actors.get('Semantics')
        if semantic:semantic.SetVisibility(False)
        self.plotter.camera_position=camera
        self.plotter.render()

    def _picked_point(self,xyz):
        if getattr(self,'video_view',False) and xyz is not None:
            self.viewport_status.setText(f'SfM position (arbitrary units): {xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f}')
            self.video_query.setText(', '.join(f'{x:.4f}' for x in xyz))
        else:super()._picked_point(xyz)

    def export_diagnostics(self):
        path,_=QFileDialog.getSaveFileName(self,'Export diagnostics',str(self.workspace/'diagnostics.json'),'JSON (*.json)')
        if path:atomic_json(Path(path),{'runtime':inventory(self.root),'jobs':self.manager.records,'errors':self.last_errors,'application_log':str(self.log_file)})

    def load_evaluation(self):
        # Historical research metrics remain accessible through the legacy research desktop.
        # A new project must not inherit unrelated benchmark results.
        from PySide6.QtWidgets import QTreeWidgetItem
        from .desktop import flatten
        self.metrics.clear()
        self.metrics.addTopLevelItem(QTreeWidgetItem(['Interpretation','Counts measure density; consistency is not GT accuracy.']))
        self.metrics.addTopLevelItem(QTreeWidgetItem(['Accuracy','Unavailable unless this run has reference evaluation.']))
        for key,value in flatten(self.project_data.get('metrics',{})):
            self.metrics.addTopLevelItem(QTreeWidgetItem([key,value]))
        for record in getattr(getattr(self,'manager',None),'records',[])[self.project_job_offset:]:
            if record['status']!='completed':continue
            for name in ('run.json','summary.json','results.json','last_render.json'):
                path=Path(record['output'])/name
                if path.is_file():
                    parent=QTreeWidgetItem([record['name'],str(path)]);self.metrics.addTopLevelItem(parent)
                    for key,value in flatten(read_metrics(path)):parent.addChild(QTreeWidgetItem([key,value]))

    def select_video(self):
        path,_=QFileDialog.getOpenFileName(self,'Import RGB video',str(self.workspace),'Video (*.mp4 *.mov)')
        if path:self.import_video_at(Path(path))

    def import_video_at(self,path):
        if self.manager.active or self.manager.queue:
            self._error('Job running','Finish or cancel pending jobs first');return
        if not path.is_file() or path.suffix.lower() not in ('.mp4','.mov'):
            self._error('Video inspection failed','Expected an existing MP4 or MOV video');return
        self.new_project();self.video_input.setText(str(path));self.project_data['input_type']='rgb_video'
        self.workflow.setCurrentIndex(0);self.inspect_video_ui()

    def inspect_video_ui(self):
        video=Path(self.video_input.text())
        out=self.workspace/'inspection';out.mkdir(exist_ok=True)
        code='from reconx.video_input import inspect_video; import json,sys; print(json.dumps(inspect_video(__import__("pathlib").Path(sys.argv[1])),indent=2))'
        try:self._submit('Inspect video',[('metadata',runtime(self.root,'video'),['-c',code,str(video)])],out)
        except Exception as exc:self._error('Video inspection failed',exc)

    def _submit(self,name,stages,output,metadata=None):
        output.mkdir(parents=True,exist_ok=True)
        record=self.manager.submit(name,stages,output,metadata)
        self.job_callbacks[record['id']]={'name':name,'output':output}
        return record

    def run_video_pipeline(self):
        try:
            video=Path(self.video_input.text()).resolve();output=self.workspace/'runs'/('video_'+time.strftime('%Y%m%d_%H%M%S'))
            model='depth_anything_v2_small' if self.video_depth_model.currentIndex()==0 else 'marigold_v1_1'
            from .app_runtime import model_path
            model_path(model,model);model_path('segformer','nvidia/segformer-b0-finetuned-ade-512-512')
            intrinsics=Path(self.video_intrinsics.text()) if self.video_intrinsics.text().strip() else None
            stages=photoreal_stages(self.root,video,output,candidate_stride=self.video_stride.value(),
                max_frames=max(8,self.video_max_frames.value()),overlap=self.video_overlap.value(),
                model=model,voxel=self.video_voxel.value(),depth_trunc=self.video_depth_trunc.value(),intrinsics=intrinsics)
            self.video_output=output;self.project_data['input_type']='rgb_video'
            record=self._submit('RGB video',stages,output,{'input':str(video),'scale':'arbitrary_sfm','GT_depth_or_pose_used':False})
            atomic_json(output/'pipeline_config.json',{'video':str(video),'parameters':self._snapshot()['parameters'],'stages':record['stages']})
        except Exception as exc:self._error('Video pipeline could not start',exc)

    def run_pipeline(self):
        try:
            if not self.paths:raise ValueError('Load a RGB-D dataset before running this workflow')
            name=self.pipeline_combo.currentText();output=Path(self.output_edit.text()).resolve()
            if name=='Gaussian render':output=self.workspace/'runs/gaussian_render'
            if name=='Inverse rendering':output=self.workspace/'runs'/('inverse_'+time.strftime('%Y%m%d_%H%M%S'));self._inverse_output=output
            output.parent.mkdir(parents=True,exist_ok=True)
            manifest=self.paths.manifest;start,stop=self.start_spin.value(),self.stop_spin.value()
            if name=='Inverse rendering' and self.inverse_pose.currentIndex()==1:
                if not self.paths.slam_manifest:raise ValueError('SLAM trajectory unavailable')
                manifest=self.paths.slam_manifest
            if name=='RGB-D SLAM':
                self._run_slam(output);return
            if name=='Gaussian train':
                dataset=output/'dataset';trained=output/'trained'
                stages=[('convert',runtime(self.root,'recon'),['-m','reconx.neural_dataset',str(manifest),'--output',str(dataset),'--width','160']),
                        ('train',runtime(self.root,'gsplat'),['-m','reconx.gaussian_scene',str(dataset/'dataset.json'),'--output',str(trained),
                         '--iterations',str(self.train_iterations.value()),'--max-gaussians',str(self.train_gaussians.value())])]
                self._submit(name,stages,output,{'depth_initialization':'GT sensor depth; not RGB-only','manifest':str(manifest)})
                return
            if name not in ('Gaussian render','Inverse rendering') and (start!=0 or stop!=len(self.scene.frames)):
                manifest=slice_manifest(manifest,output/'input_subset.json',start,stop)
            job=build_job(name,self.root,manifest,output,voxel_size_m=self.voxel_spin.value(),truncation_m=self.trunc_spin.value(),
                frame_index=self.gaussian_frame.value(),gaussian_dataset=self.paths.gaussian_dataset,gaussian_checkpoint=self.paths.gaussian_checkpoint,
                inverse_mesh=Path(self.inverse_mesh.text()),inverse_regularization=self.inverse_reg.value(),pose_source='GT' if self.inverse_pose.currentIndex()==0 else 'RTAB')
            self.project_data['input_type']='rgbd'
            self._submit(name,[(name,job.program,job.arguments)],output,{'manifest':str(manifest),'frame_range':[start,stop]})
        except Exception as exc:self._error('Workflow could not start',exc)

    def _run_slam(self,output):
        config=read_config(self.root);binary=resolve(self.root,config['external'].get('rtabmap','tools/rtabmap-0.24.2/bin/rtabmap-rgbd_dataset.exe'))
        if not binary.is_file():raise FileNotFoundError('RTAB-Map executable missing. Configure it in Home / Runtime assets.')
        # Prepare only the selected RGB/depth pairs. No reference camera poses are copied to SLAM input.
        p=runtime(self.root,'recon');source=output/'input_freiburg3_sequence';result=output/'rtabmap'
        stages=[('prepare',p,['-m','reconx.app_slam',str(self.paths.manifest),str(source),'--max-frames',str(self.stop_spin.value())]),
                ('RTAB-Map',binary,['--output',str(result),'--output_name','scene','--Rtabmap/DetectionRate','0','--RGBD/LinearUpdate','0','--RGBD/AngularUpdate','0',str(source)])]
        result.mkdir(parents=True,exist_ok=True)
        self._submit('RGB-D SLAM',stages,output,{'ground_truth_provided_to_slam':False,'manifest':str(self.paths.manifest)})

    def inverse_output(self):return self._inverse_output or super().inverse_output()

    def train_gaussian(self):
        self.pipeline_combo.setCurrentText('Gaussian train');self.run_pipeline()

    def _job_changed(self,record):
        self.job_status.setPlainText('\n'.join(f'{tr(r["name"],self.language)}: {tr(r["status"],self.language)} ({int(r["progress"]*100)}%)' for r in self.manager.records[-12:]))
        active=self.manager.active
        self.current_job=(None,self.manager.process,0) if active else None
        self.video_process=self.manager.process if active and active['name']=='RGB video' else None
        self.progress.setRange(0,100);self.progress.setValue(int(record['progress']*100))
        self.cancel_button.setEnabled(bool(active));self.video_cancel.setEnabled(bool(active))
        self.run_button.setEnabled(not active);self.video_run.setEnabled(not active)

    def _finished(self,record):
        self.current_job=None;self.video_process=None;self.run_button.setEnabled(True);self.video_run.setEnabled(True)
        self.cancel_button.setEnabled(False);self.video_cancel.setEnabled(False)
        self.history.append({'name':record['name'],'success':record['status']=='completed','output':record['output'],'elapsed_s':record.get('wall_s')})
        self.log(f'{record["name"]}: {record["status"]}; log: {record["log_dir"]}')
        output=Path(record['output'])
        if record['status']=='completed':
            if record['name']=='TSDF' and self.paths:
                self.paths=replace(self.paths,mesh=output/'mesh.ply',cloud=output/'cloud.ply');self.load_mesh()
            elif record['name']=='RGB video':self.video_output=output;self.load_video_results()
            elif record['name']=='Gaussian train' and self.paths:
                self.paths=replace(self.paths,gaussian_dataset=output/'dataset/dataset.json',gaussian_checkpoint=output/'trained/scene.pt')
                self._load_gaussian_info()
            elif record['name']=='Gaussian render':
                from PySide6.QtGui import QPixmap
                self.gaussian_image.setPixmap(QPixmap(str(output/f'frame_{self.gaussian_frame.value():04d}_render.png')).scaledToWidth(350))
            elif record['name']=='Inverse rendering':self.load_inverse()
            elif record['name']=='Inspect video':
                raw=(Path(record['log_dir'])/'00.log').read_text();self.video_info.setPlainText(raw)
                info=json.loads(raw);self.project_data['video_metadata']=info
                self.input_summary.setText(f'{Path(info["source"]).name}\n{info["width"]} × {info["height"]} · {info["duration_s_nominal"]:.2f} s\nRGB Video · extracted frames: 0 (not run)\nCalibration: supplied' if self.video_intrinsics.text() else f'{Path(info["source"]).name}\n{info["width"]} × {info["height"]} · {info["duration_s_nominal"]:.2f} s\nRGB Video · extracted frames: 0 (not run)\nCalibration: to be estimated')
            for filename in ('run.json','results.json','summary.json','last_render.json'):
                path=output/filename
                if path.is_file():
                    from PySide6.QtWidgets import QTreeWidgetItem
                    from .desktop import flatten
                    item=QTreeWidgetItem([record['name'],str(path)]);self.metrics.addTopLevelItem(item)
                    for key,value in flatten(read_metrics(path)):item.addChild(QTreeWidgetItem([key,value]))
        elif record['status']=='failed':
            if record['name']=='RGB video' and (output/'dense/dense_report.json').is_file():
                self.video_output=output;self.load_video_results()
            failure=dict(record['error']);failure['details']+='\nJob log: '+record['log_dir']
            self._error('Job failed',failure)

    def cancel_pipeline(self):self.manager.cancel()
    def cancel_video_pipeline(self):self.manager.cancel()
    def closeEvent(self,event):
        if hasattr(self,'dollhouse'):self.dollhouse.stop()
        self.manager.shutdown();self.current_job=None;self.video_process=None;super().closeEvent(event)


def main(argv=None):
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',type=Path);parser.add_argument('--project',type=Path)
    parser.add_argument('--workspace',type=Path);parser.add_argument('--validate',type=Path)
    args=parser.parse_args(argv);app=QApplication(sys.argv[:1]);app.setApplicationName('Rebuilder 3D')
    window=StudioWindow(manifest=args.manifest,workspace=args.workspace);window.show()
    if args.project:QTimer.singleShot(0,lambda:window.restore_project(args.project))
    if args.validate:
        from .app_validation import validate
        QTimer.singleShot(1000,lambda:validate(app,window,args.validate))
    return app.exec()


if __name__=='__main__':raise SystemExit(main())
