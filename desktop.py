"""RECON-X Windows research desktop. Algorithms run in their existing environments."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault('QT_API','pyside6')
import numpy as np
import psutil
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtGui import QAction, QColor, QIcon, QImage, QPixmap
from PySide6.QtWidgets import (QApplication, QCheckBox, QComboBox, QDockWidget,
    QFileDialog, QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QSpinBox, QDoubleSpinBox, QSplitter,
    QTabWidget, QTextEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from .desktop_data import (ProjectPaths, cached_semantic, camera_frustums, discover, load_session, metric_files, metric_summary,
    preview_frame, read_metrics, save_session, scene_summary, semantic_geometry,
    semantic_query, slice_manifest, trajectory)
from .desktop_jobs import build_job
from .io import load_scene
from .semantic_map import SemanticVoxelMap, palette
from .tum import make_tum_manifest
from .ui_appearance import QComboBox


ROOT=Path(__file__).resolve().parents[2]


def image_label(array: np.ndarray, max_width: int=300) -> QPixmap:
    data=np.ascontiguousarray(array.astype(np.uint8))
    h,w=data.shape[:2]
    if data.ndim==2:
        image=QImage(data.data,w,h,w,QImage.Format.Format_Grayscale8).copy()
    else:
        image=QImage(data.data,w,h,3*w,QImage.Format.Format_RGB888).copy()
    return QPixmap.fromImage(image).scaledToWidth(max_width,Qt.TransformationMode.SmoothTransformation)


def flatten(data, prefix=''):
    if isinstance(data,dict):
        for key,value in data.items():yield from flatten(value,f'{prefix}.{key}' if prefix else str(key))
    elif isinstance(data,list):
        if len(data)<=6 and all(not isinstance(v,(dict,list)) for v in data):
            yield prefix,', '.join(str(x) for x in data)
        else:yield prefix,f'{len(data)} entries'
    else:yield prefix,'missing' if data is None else str(data)


class MainWindow(QMainWindow):
    def __init__(self, root: Path=ROOT, manifest: Path | None=None):
        super().__init__()
        self.root=root.resolve();self.paths:ProjectPaths | None=None;self.scene=None
        self.actors={};self.current_job=None
        history_path=root/'outputs/phase9/history.json'
        try:self.history=read_metrics(history_path) if history_path.is_file() else []
        except (ValueError,OSError):self.history=[]
        self.dataset_load_s=None;self.mesh_load_s=None
        self._gui_process=psutil.Process();self._gui_process.cpu_percent();self.gui_peak_rss_bytes=0
        self.setWindowTitle('RECON-X  |  3D Reconstruction Studio')
        self.resize(1550,920)
        self.setStyleSheet('''QWidget{background:#171c24;color:#e5edf6;font-size:12px}
            QTreeWidget,QTextEdit,QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox{background:#202936;border:1px solid #38485a;padding:4px}
            QPushButton{background:#294057;border:1px solid #4d6980;padding:7px;border-radius:3px}
            QPushButton:hover{background:#365b78} QTabWidget::pane{border:1px solid #344355}
            QTabBar::tab{background:#233142;padding:8px 12px} QTabBar::tab:selected{background:#365b78}
            QProgressBar{border:1px solid #4d6980;text-align:center} QProgressBar::chunk{background:#55a9d8}''')
        self._build_ui();self._build_menu()
        self.memory_timer=QTimer(self);self.memory_timer.timeout.connect(self._update_memory);self.memory_timer.start(1000)
        if manifest is None:manifest=self.root/'outputs/tum_fr3_structure_notexture_far/scene.json'
        if manifest.is_file():QTimer.singleShot(0,lambda:self.load_dataset(manifest))

    def _build_ui(self):
        splitter=QSplitter(Qt.Orientation.Horizontal)
        left=QWidget();self.dataset_panel=left;left_layout=QVBoxLayout(left)
        heading=QLabel('DATASET  /  PIPELINES');heading.setStyleSheet('font-size:15px;font-weight:bold;color:#8fd7ff')
        left_layout.addWidget(heading)
        self.dataset_tree=QTreeWidget();self.dataset_tree.setHeaderLabels(['Asset','Value'])
        self.dataset_tree.itemClicked.connect(self._tree_click)
        left_layout.addWidget(self.dataset_tree,3)
        open_button=QPushButton('Open TUM manifest / directory');open_button.clicked.connect(self.open_dataset)
        left_layout.addWidget(open_button)
        self.frame_spin=QSpinBox();self.frame_spin.valueChanged.connect(self.show_frame)
        left_layout.addWidget(QLabel('Frame index'));left_layout.addWidget(self.frame_spin)
        row=QHBoxLayout();self.start_spin=QSpinBox();self.stop_spin=QSpinBox()
        row.addWidget(QLabel('Range'));row.addWidget(self.start_spin);row.addWidget(QLabel('to'));row.addWidget(self.stop_spin)
        left_layout.addLayout(row)
        self.rgb_preview=QLabel('RGB frame');self.depth_preview=QLabel('Sensor depth')
        for label in (self.rgb_preview,self.depth_preview):
            label.setAlignment(Qt.AlignmentFlag.AlignCenter);label.setMinimumHeight(120)
            left_layout.addWidget(label)
        left_layout.addWidget(QLabel('Pipeline'))
        self.pipeline_combo=QComboBox();self.pipeline_combo.addItems(['TSDF','Depth Anything V2','Marigold v1.1','Semantic map','Gaussian render','Inverse rendering'])
        left_layout.addWidget(self.pipeline_combo)
        self.voxel_spin=QDoubleSpinBox();self.voxel_spin.setRange(.005,.1);self.voxel_spin.setSingleStep(.005);self.voxel_spin.setValue(.03);self.voxel_spin.setDecimals(3)
        self.trunc_spin=QDoubleSpinBox();self.trunc_spin.setRange(.02,.3);self.trunc_spin.setSingleStep(.01);self.trunc_spin.setValue(.09);self.trunc_spin.setDecimals(3)
        row=QHBoxLayout();row.addWidget(QLabel('Voxel m'));row.addWidget(self.voxel_spin);row.addWidget(QLabel('Trunc m'));row.addWidget(self.trunc_spin)
        left_layout.addLayout(row)
        self.output_edit=QLineEdit(str(self.root/'outputs/phase9/tsdf_gui'))
        left_layout.addWidget(self.output_edit)
        browse=QPushButton('Output directory…');browse.clicked.connect(self.choose_output);left_layout.addWidget(browse)
        row=QHBoxLayout();self.run_button=QPushButton('▶ Run');self.cancel_button=QPushButton('■ Cancel')
        self.run_button.clicked.connect(self.run_pipeline);self.cancel_button.clicked.connect(self.cancel_pipeline)
        self.cancel_button.setEnabled(False);row.addWidget(self.run_button);row.addWidget(self.cancel_button);left_layout.addLayout(row)
        splitter.addWidget(left)

        center=QWidget();center_layout=QVBoxLayout(center)
        toolbar=QHBoxLayout()
        for text,method in [('Fit scene',self.fit_scene),('Reset camera',self.reset_camera),('Load mesh',self.load_mesh),('Load cloud',self.load_cloud),('GT path',lambda:self.load_path(False)),('SLAM path',lambda:self.load_path(True)),('Frustums',self.load_frustums),('Semantics',self.load_semantics)]:
            button=QPushButton(text);button.clicked.connect(method);toolbar.addWidget(button)
        center_layout.addLayout(toolbar)
        self.plotter=QtInteractor(center)
        self.plotter.set_background('#111923')
        self.plotter.add_axes()
        self.plotter.enable_point_picking(callback=self._picked_point,show_message=False,
                                           left_clicking=False)
        center_layout.addWidget(self.plotter,1)
        self.viewport_status=QLabel('Orbit: left drag  •  Pan: middle drag  •  Zoom: wheel  •  Pick: right click')
        center_layout.addWidget(self.viewport_status)
        toggle=QHBoxLayout();self.visible_checks={}
        for name in ('Mesh','Cloud','GT path','SLAM path','Frustums','Semantics','Inverse material','Axes'):
            check=QCheckBox(name);check.setChecked(True);check.toggled.connect(lambda value,n=name:self._set_visible(n,value))
            self.visible_checks[name]=check;toggle.addWidget(check)
        self.point_size=QSpinBox();self.point_size.setRange(1,12);self.point_size.setValue(3)
        self.point_size.valueChanged.connect(self._refresh_points)
        toggle.addWidget(QLabel('Point size'));toggle.addWidget(self.point_size)
        center_layout.addLayout(toggle)
        self.color_mode=QComboBox();self.color_mode.addItems(['Default color','Height (meters)'])
        self.color_mode.currentIndexChanged.connect(self._refresh_color)
        center_layout.addWidget(self.color_mode)
        splitter.addWidget(center)

        right=QTabWidget();self.right_tabs=right;right.setMinimumWidth(350)
        self.inspector=QTextEdit();self.inspector.setReadOnly(True);right.addTab(self.inspector,'Inspector')
        right.addTab(self._depth_page(),'Depth')
        right.addTab(self._gaussian_page(),'3DGS')
        right.addTab(self._semantic_page(),'Semantics')
        right.addTab(self._evaluation_page(),'Evaluation')
        right.addTab(self._inverse_page(),'Inverse')
        right.addTab(self._video_page(),'RGB Video')
        right.currentChanged.connect(lambda index:self.dataset_panel.setVisible(index!=right.count()-1))
        splitter.addWidget(right)
        splitter.setSizes([285,890,375]);self.setCentralWidget(splitter)

        dock=QDockWidget('Console / Progress / Performance',self)
        bottom=QWidget();layout=QVBoxLayout(bottom)
        self.progress=QProgressBar();self.progress.setRange(0,1);self.progress.setValue(0)
        self.console=QTextEdit();self.console.setReadOnly(True);self.console.setMaximumHeight(120)
        self.performance=QLabel('Idle')
        layout.addWidget(self.progress);layout.addWidget(self.performance);layout.addWidget(self.console)
        dock.setWidget(bottom);self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea,dock)

    def _depth_page(self):
        page=QWidget();layout=QVBoxLayout(page)
        self.depth_model=QComboBox();self.depth_model.addItems(['Depth Anything V2 Small (relative)','Marigold v1.1 (affine-invariant)'])
        self.depth_model.currentIndexChanged.connect(self.show_depth_result)
        self.depth_method=QComboBox();self.depth_method.addItems(['Raw model output','Robust aligned meters (oracle GT, Phase 4)','Sparse aligned meters (RTAB RGB-D, Phase 6)','GT sensor depth (meters)','Robust held-out absolute error (meters)'])
        self.depth_method.currentIndexChanged.connect(self.show_depth_result)
        layout.addWidget(self.depth_model);layout.addWidget(self.depth_method)
        self.depth_image=QLabel('No depth output');self.depth_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.depth_image)
        self.depth_info=QTextEdit();self.depth_info.setReadOnly(True);layout.addWidget(self.depth_info,1)
        return page

    def _gaussian_page(self):
        page=QWidget();layout=QVBoxLayout(page)
        self.gaussian_info=QTextEdit();self.gaussian_info.setReadOnly(True)
        self.gaussian_frame=QSpinBox();self.gaussian_frame.setRange(0,15);self.gaussian_frame.setValue(14)
        button=QPushButton('Render selected pose on CUDA');button.clicked.connect(self.render_gaussian)
        self.gaussian_image=QLabel('Render a saved scene at a selected camera pose')
        self.gaussian_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.gaussian_info);layout.addWidget(QLabel('Camera frame'));layout.addWidget(self.gaussian_frame)
        layout.addWidget(button);layout.addWidget(self.gaussian_image)
        return page

    def _semantic_page(self):
        page=QWidget();layout=QVBoxLayout(page)
        self.semantic_class=QComboBox();self.semantic_class.currentIndexChanged.connect(self.load_semantics)
        self.semantic_info=QTextEdit();self.semantic_info.setReadOnly(True)
        self.query_edit=QLineEdit('-3.7, 1.7, 0.02')
        query=QPushButton('Query 3D world position');query.clicked.connect(self.query_semantics)
        self.query_result=QLabel('No query')
        layout.addWidget(QLabel('Semantic class filter'));layout.addWidget(self.semantic_class)
        layout.addWidget(self.semantic_info);layout.addWidget(self.query_edit);layout.addWidget(query);layout.addWidget(self.query_result)
        return page

    def _evaluation_page(self):
        page=QWidget();layout=QVBoxLayout(page)
        self.metrics=QTreeWidget();self.metrics.setHeaderLabels(['Metric / condition','Value'])
        button=QPushButton('Refresh actual experiment files');button.clicked.connect(self.load_evaluation)
        layout.addWidget(button);layout.addWidget(self.metrics)
        return page

    def _inverse_page(self):
        page=QWidget();layout=QVBoxLayout(page)
        self.inverse_pose=QComboBox();self.inverse_pose.addItems(['GT pose + GT TSDF','RTAB pose + RTAB TSDF'])
        self.inverse_mesh=QLineEdit(str(self.root/'outputs/tum_fr3_structure_notexture_far/tsdf/mesh.ply'))
        self.inverse_pose.currentIndexChanged.connect(self._inverse_pose_changed)
        self.inverse_reg=QDoubleSpinBox();self.inverse_reg.setRange(0,100);self.inverse_reg.setValue(8)
        self.inverse_reg.setDecimals(2)
        self.inverse_illumination=QComboBox();self.inverse_illumination.addItem('Fixed ambient + directional light')
        self.inverse_method=QComboBox();self.inverse_method.addItems(['lambertian_regularized','lambertian_unregularized','observed_appearance','constant_albedo'])
        self.inverse_mode=QComboBox();self.inverse_mode.addItems(['original','normals','albedo','shading','reconstructed','error','mask'])
        self.inverse_mode.currentIndexChanged.connect(self.show_inverse_image)
        self.inverse_frame=QSpinBox();self.inverse_frame.setRange(0,15);self.inverse_frame.setValue(14)
        self.inverse_frame.valueChanged.connect(self.show_inverse_image)
        self.inverse_method.currentIndexChanged.connect(self.show_inverse_image)
        for label,widget in [('Pose / geometry',self.inverse_pose),('TSDF mesh',self.inverse_mesh),
                             ('Illumination model',self.inverse_illumination),('Regularization',self.inverse_reg),
                             ('Method',self.inverse_method),('View',self.inverse_mode),('Frame',self.inverse_frame)]:
            layout.addWidget(QLabel(label));layout.addWidget(widget)
        row=QHBoxLayout()
        run=QPushButton('Optimize in background');run.clicked.connect(self.run_inverse)
        load=QPushButton('Load results');load.clicked.connect(self.load_inverse)
        material=QPushButton('Show 3D material');material.clicked.connect(self.show_inverse_material)
        row.addWidget(run);row.addWidget(load);row.addWidget(material);layout.addLayout(row)
        self.inverse_light=[QDoubleSpinBox() for _ in range(3)]
        row=QHBoxLayout();row.addWidget(QLabel('3D light XYZ'))
        for spin,value in zip(self.inverse_light,(.25,-.1,.5)):
            spin.setRange(-1.5,1.5);spin.setSingleStep(.05);spin.setValue(value)
            spin.valueChanged.connect(self.show_inverse_material)
            row.addWidget(spin)
        layout.addLayout(row)
        self.inverse_image=QLabel('Run or load Phase 10 results');self.inverse_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inverse_image.setMinimumHeight(180);layout.addWidget(self.inverse_image)
        self.inverse_info=QTextEdit();self.inverse_info.setReadOnly(True);layout.addWidget(self.inverse_info,1)
        return page

    def _video_page(self):
        page=QWidget();layout=QVBoxLayout(page)
        layout.addWidget(QLabel('Phase 11 · RGB video only · SfM scale is arbitrary'))
        self.video_input=QLineEdit()
        browse=QPushButton('Select MP4 / MOV');browse.clicked.connect(self.select_video)
        layout.addWidget(self.video_input);layout.addWidget(browse)
        inspect=QPushButton('Inspect video metadata');inspect.clicked.connect(self.inspect_video_ui)
        layout.addWidget(inspect)
        self.video_intrinsics=QLineEdit();self.video_intrinsics.setPlaceholderText('Optional calibrated intrinsics JSON path')
        layout.addWidget(self.video_intrinsics)
        self.video_stride=QSpinBox();self.video_stride.setRange(1,30);self.video_stride.setValue(3)
        self.video_max_frames=QSpinBox();self.video_max_frames.setRange(2,500);self.video_max_frames.setValue(60)
        self.video_overlap=QSpinBox();self.video_overlap.setRange(1,30);self.video_overlap.setValue(10)
        self.video_depth_model=QComboBox();self.video_depth_model.addItems(['Depth Anything V2 Small','Marigold v1.1'])
        self.video_voxel=QDoubleSpinBox();self.video_voxel.setRange(.01,10.);self.video_voxel.setDecimals(2);self.video_voxel.setValue(.4)
        self.video_depth_trunc=QDoubleSpinBox();self.video_depth_trunc.setRange(1.,60.);self.video_depth_trunc.setValue(35.)
        for title,widget in [('Frame stride',self.video_stride),('Maximum frames',self.video_max_frames)]:
            layout.addWidget(QLabel(title));layout.addWidget(widget)
        layout.addWidget(QLabel('SfM neighbor overlap'));layout.addWidget(self.video_overlap)
        layout.addWidget(QLabel('Relative depth model'));layout.addWidget(self.video_depth_model)
        layout.addWidget(QLabel('TSDF voxel length (SfM units)'));layout.addWidget(self.video_voxel)
        layout.addWidget(QLabel('TSDF depth truncation (SfM units)'));layout.addWidget(self.video_depth_trunc)
        self.video_run=QPushButton('Run RGB video pipeline')
        self.video_run.clicked.connect(self.run_video_pipeline)
        self.video_cancel=QPushButton('Cancel video pipeline')
        self.video_cancel.clicked.connect(self.cancel_video_pipeline)
        self.video_cancel.setEnabled(False)
        self.video_load=QPushButton('Load Phase 11 results')
        self.video_load.clicked.connect(self.load_video_results)
        layout.addWidget(self.video_run);layout.addWidget(self.video_cancel);layout.addWidget(self.video_load)
        self.video_query=QLineEdit();self.video_query.setPlaceholderText('3D query XYZ in SfM units, e.g. 0, 0, 10')
        query=QPushButton('Query video semantic map');query.clicked.connect(self.query_video_semantics)
        layout.addWidget(self.video_query);layout.addWidget(query)
        self.video_info=QTextEdit();self.video_info.setReadOnly(True)
        self.video_info.setPlainText('Select a video. SfM, relative depth, TSDF and semantics run from RGB only. Camera intrinsics are estimated unless separately calibrated.')
        layout.addWidget(self.video_info,1)
        return page

    def select_video(self):
        path,_=QFileDialog.getOpenFileName(self,'Select RGB video',str(self.root),'Video (*.mp4 *.mov)')
        if path:self.video_input.setText(path)

    def inspect_video_ui(self):
        import subprocess
        video=Path(self.video_input.text())
        if not video.is_file():self._error('Video unavailable','Select an existing video');return
        try:
            result=subprocess.run(['ffprobe','-v','error','-select_streams','v:0',
                '-show_entries','stream=width,height,r_frame_rate,nb_frames,codec_name:format=duration',
                '-of','json',str(video)],capture_output=True,text=True,check=True)
            self.video_info.setPlainText(json.dumps(json.loads(result.stdout),indent=2))
        except Exception as exc:self._error('Video inspection failed',exc)

    def run_video_pipeline(self):
        if self.current_job or getattr(self,'video_process',None):
            self._error('Pipeline busy','A pipeline is already running');return
        video=Path(self.video_input.text())
        if not video.is_file() or video.suffix.lower() not in ('.mp4','.mov'):
            self._error('Video unavailable','Select an existing MP4 or MOV video');return
        stamp=time.strftime('%Y%m%d_%H%M%S')
        self.video_output=self.root/'outputs/phase11/gui_runs'/f'{video.stem}_{stamp}'
        self.video_output.mkdir(parents=True,exist_ok=False)
        source_sidecar=video.parent/'source.json'
        if source_sidecar.is_file():
            import shutil
            shutil.copyfile(source_sidecar,self.video_output/'source.json')
        intrinsics=Path(self.video_intrinsics.text()) if self.video_intrinsics.text().strip() else None
        if intrinsics and not intrinsics.is_file():
            self._error('Intrinsics unavailable',intrinsics);return
        calibrated=['--intrinsics-json',str(intrinsics)] if intrinsics else []
        model='depth_anything_v2_small' if self.video_depth_model.currentIndex()==0 else 'marigold_v1_1'
        self.video_stages=[
            ('extract',self.root/'.venv-video/Scripts/python.exe',
             ['-m','reconx.video_input',str(video),'--output',str(self.video_output/'extracted'),
              '--stride',str(self.video_stride.value()),'--max-frames',str(self.video_max_frames.value()),*calibrated]),
            ('SfM',self.root/'.venv-video/Scripts/python.exe',
             ['-m','reconx.video_sfm',str(self.video_output/'extracted'),'--output',str(self.video_output/'sfm'),
              '--overlap',str(self.video_overlap.value()),*calibrated]),
            ('depth',self.root/'.venv-depth/Scripts/python.exe',
             ['-m','reconx.video_depth',str(self.video_output/'extracted'),
              str(self.video_output/'sfm/sfm.json'),'--output',str(self.video_output/'depth'),'--model',model]),
            ('multiview',self.root/'.venv/Scripts/python.exe',
             ['-m','reconx.video_multiview',str(self.video_output/'sfm/sfm.json'),
              str(self.video_output/'depth'),'--output',str(self.video_output/'multiview')]),
            ('TSDF',self.root/'.venv/Scripts/python.exe',
             ['-m','reconx.video_dense',str(self.video_output/'extracted'),
              str(self.video_output/'sfm/sfm.json'),str(self.video_output/'depth'),
              '--output',str(self.video_output/'dense'),'--voxel-length',str(self.video_voxel.value()),
              '--depth-trunc',str(self.video_depth_trunc.value())]),
            ('semantics',self.root/'.venv-depth/Scripts/python.exe',
             ['-m','reconx.video_semantic',str(self.video_output/'extracted'),
              str(self.video_output/'sfm/sfm.json'),str(self.video_output/'depth'),
              '--output',str(self.video_output/'semantic')]),
            ('summary',self.root/'.venv/Scripts/python.exe',
             ['-m','reconx.video_summary',str(self.video_output)])]
        if any(not program.is_file() for _,program,_ in self.video_stages):
            self._error('Environment missing','Phase 11 requires .venv-video, .venv-depth and .venv');return
        (self.video_output/'pipeline_config.json').write_text(json.dumps({
            'video':str(video.resolve()),'stride':self.video_stride.value(),
            'max_frames':self.video_max_frames.value(),'pose_source':'RGB-only SfM',
            'sfm_overlap':self.video_overlap.value(),'intrinsics_json':str(intrinsics) if intrinsics else None,
            'depth_model':model,'depth_source':'relative monocular + SfM sparse alignment',
            'voxel_length_sfm':self.video_voxel.value(),'depth_trunc_sfm':self.video_depth_trunc.value(),
            'coordinate_unit':'SfM arbitrary units, not meters'},indent=2),encoding='utf-8')
        self.video_stage_index=0;self.video_run.setEnabled(False)
        self.video_cancel.setEnabled(True)
        self.progress.setRange(0,len(self.video_stages));self.progress.setValue(0)
        self.video_info.setPlainText(f'Output: {self.video_output}\nStarting RGB-only pipeline…')
        self._start_video_stage()

    def _start_video_stage(self):
        name,program,args=self.video_stages[self.video_stage_index]
        self.progress.setValue(self.video_stage_index)
        process=QProcess(self);self.video_process=process
        process.setProgram(str(program));process.setArguments(args)
        process.setWorkingDirectory(str(self.root))
        process.readyReadStandardOutput.connect(lambda:self._read_video_process(process,False,name))
        process.readyReadStandardError.connect(lambda:self._read_video_process(process,True,name))
        process.finished.connect(lambda code,status:self._video_stage_finished(name,code,status))
        process.errorOccurred.connect(lambda error:self._video_process_error(name,error))
        self.log(f'Phase 11 stage {self.video_stage_index+1}/{len(self.video_stages)}: {name}')
        process.start()

    def _read_video_process(self,process,stderr,name):
        data=bytes(process.readAllStandardError() if stderr else process.readAllStandardOutput())
        message=data.decode('utf-8',errors='replace')
        with (self.video_output/f'{name}.log').open('a',encoding='utf-8') as log:
            log.write(message)
        for line in message.splitlines():self.log(line)

    def _video_process_error(self,name,error):
        self.log(f'Phase 11 {name} process error: {error}')
        if error==QProcess.ProcessError.FailedToStart:
            self.video_process=None;self.video_run.setEnabled(True);self.video_cancel.setEnabled(False)
            self.video_info.append(f'{name} failed to start')

    def _video_stage_finished(self,name,code,status):
        self.video_process=None
        if code!=0 or status!=QProcess.ExitStatus.NormalExit:
            self.video_run.setEnabled(True);self.video_cancel.setEnabled(False)
            self.video_info.append(f'{name} FAILED (exit {code}); inspect console and {self.video_output}')
            return
        self.video_info.append(f'{name} completed')
        self.video_stage_index+=1
        self.progress.setValue(self.video_stage_index)
        if self.video_stage_index<len(self.video_stages):
            self._start_video_stage()
        else:
            self.video_run.setEnabled(True);self.video_cancel.setEnabled(False)
            self.load_video_results()

    def cancel_video_pipeline(self):
        process=getattr(self,'video_process',None)
        if process:
            process.terminate();self.log('Phase 11 cancellation requested')
            QTimer.singleShot(3000,lambda:process.kill() if process.state()!=QProcess.ProcessState.NotRunning else None)

    def load_video_results(self):
        output=getattr(self,'video_output',self.root/'outputs/phase11/public_tum_rgb')
        try:
            sfm=read_metrics(output/'sfm/sfm.json')
            dense=read_metrics(output/'dense/dense_report.json')
            if not dense['triangles'] and (output/'dense_uniform/dense_report.json').is_file():
                dense=read_metrics(output/'dense_uniform/dense_report.json')
            semantic=read_metrics(output/'semantic/semantic_report.json') if (output/'semantic/semantic_report.json').is_file() else {'occupied_voxels':0,'total_inference_s':None}
            self.progress.setRange(0,1);self.progress.setValue(1)
            multiview=read_metrics(output/'multiview/multiview.json') if (output/'multiview/multiview.json').is_file() else {}
            self.video_info.setPlainText(json.dumps({
                'coordinate_unit':'SfM arbitrary units, not meters',
                'registered_frames':sfm['registered_frames'],'input_frames':sfm['input_frames'],
                'sparse_points':sfm['landmark_count'],'tsdf_triangles':dense['triangles'],
                'dense_points':dense['points'],'semantic_voxels':semantic['occupied_voxels'],
                'multiview_median_relative_residual':multiview.get('median_pair_relative_residual'),
                'sfm_time_s':sfm['timing'],'depth_inference_s':read_metrics(output/'depth/depth_report.json')['total_inference_s'],
                'tsdf_fusion_s':dense['fusion_s'],'semantic_inference_s':semantic['total_inference_s'],
                'output':str(output)},indent=2))
            mesh=pv.read(dense['mesh'])
            self._set_actor('Mesh',mesh,color='#9db8c9',show_edges=False)
            if hasattr(self,'mesh_view_mode'):self.apply_mesh_view_mode()
            path=np.asarray([row['T_world_camera'] for row in sfm['frames']])[:,:3,3]
            self._set_actor('SLAM path',pv.lines_from_points(path),color='#ffb66f',line_width=4)
            semantic_cloud=pv.read(output/'semantic/semantic_points.ply') if (output/'semantic/semantic_points.ply').is_file() else pv.PolyData()
            if semantic_cloud.n_points:
                self._set_actor('Semantics',semantic_cloud,style='points',point_size=3,
                    render_points_as_spheres=True,scalars='RGB',rgb=True,show_scalar_bar=False)
            self.fit_scene();self.log(f'Phase 11 RGB video scene loaded: {dense["triangles"]} triangles')
        except Exception as exc:self._error('Phase 11 result loading failed',exc)

    def query_video_semantics(self):
        try:
            output=getattr(self,'video_output',self.root/'outputs/phase11/public_tum_rgb')
            query=np.asarray([float(v.strip()) for v in self.video_query.text().split(',')])
            result=SemanticVoxelMap.load(output/'semantic/semantic_map.json').query(query)
            self.video_info.append(json.dumps({'query_sfm':query.tolist(),'result':result},indent=2))
        except Exception as exc:self._error('Video semantic query failed',exc)

    def _build_menu(self):
        menu=self.menuBar().addMenu('Project')
        for title,callback in [('Open dataset',self.open_dataset),('Open session',self.open_session),
                               ('Save session',self.save_session),('Exit',self.close)]:
            action=QAction(title,self);action.triggered.connect(callback);menu.addAction(action)
        artifacts=self.menuBar().addMenu('Artifacts')
        for title,callback in [('Open TSDF mesh',self.open_mesh),('Open point cloud',self.open_cloud),
                               ('Open trajectory manifest',self.open_trajectory),
                               ('Open semantic map',self.open_semantic_map),
                               ('Open Gaussian checkpoint and dataset',self.open_gaussian_scene)]:
            action=QAction(title,self);action.triggered.connect(callback);artifacts.addAction(action)

    def log(self,message):
        self.console.append(f'[{time.strftime("%H:%M:%S")}] {message}')

    def _error(self,title,error):
        self.log(f'ERROR {title}: {error}')
        QMessageBox.warning(self,title,str(error))

    def open_dataset(self):
        directory=QFileDialog.getExistingDirectory(self,'Choose TUM dataset or output directory',str(self.root/'outputs'))
        if not directory:return
        folder=Path(directory)
        manifest=folder/'scene.json'
        if not manifest.is_file():
            candidates=list(folder.glob('*/scene.json'))
            if len(candidates)==1:manifest=candidates[0]
        if not manifest.is_file() and all((folder/name).is_file() for name in ('rgb.txt','depth.txt','groundtruth.txt')):
            try:
                manifest=self.root/'outputs/phase9/imports'/folder.name/'scene.json'
                make_tum_manifest(folder,manifest)
                self.log(f'Imported raw TUM sequence: {folder}')
            except Exception as exc:self._error('TUM import failed',exc);return
        if not manifest.is_file():
            self._error('Dataset not ready','Choose a RECON-X scene.json directory or raw TUM folder with rgb.txt, depth.txt and groundtruth.txt.')
            return
        self.load_dataset(manifest)

    def open_mesh(self):
        if not self.paths:return
        path,_=QFileDialog.getOpenFileName(self,'Open TSDF mesh',str(self.root/'outputs'),'PLY (*.ply)')
        if path:self.paths=replace(self.paths,mesh=Path(path));self.load_mesh()

    def open_cloud(self):
        if not self.paths:return
        path,_=QFileDialog.getOpenFileName(self,'Open point cloud',str(self.root/'outputs'),'PLY (*.ply)')
        if path:self.paths=replace(self.paths,cloud=Path(path));self.load_cloud()

    def open_trajectory(self):
        if not self.paths:return
        path,_=QFileDialog.getOpenFileName(self,'Open trajectory manifest',str(self.root/'outputs'),'JSON (*.json)')
        if not path:return
        try:
            candidate=load_scene(Path(path))
            if len(candidate.frames)!=len(self.scene.frames) or any(abs(a.timestamp_s-b.timestamp_s)>.002 for a,b in zip(candidate.frames,self.scene.frames)):
                raise ValueError('Trajectory timestamps do not match current RGB-D frames')
            self.paths=replace(self.paths,slam_manifest=Path(path));self.load_path(True)
        except Exception as exc:self._error('Trajectory rejected',exc)

    def open_semantic_map(self):
        if not self.paths:return
        path,_=QFileDialog.getOpenFileName(self,'Open semantic map',str(self.root/'outputs'),'JSON (*.json)')
        if not path:return
        try:
            SemanticVoxelMap.load(Path(path))
            self.paths=replace(self.paths,semantic_map=Path(path))
            self._init_semantic_classes();self.load_semantics()
        except Exception as exc:self._error('Semantic map rejected',exc)

    def open_gaussian_scene(self):
        if not self.paths:return
        checkpoint,_=QFileDialog.getOpenFileName(self,'Open Gaussian checkpoint',str(self.root/'outputs'),'PyTorch (*.pt)')
        if not checkpoint:return
        dataset,_=QFileDialog.getOpenFileName(self,'Open Gaussian dataset',str(self.root/'outputs'),'JSON (*.json)')
        if not dataset:return
        try:
            info=read_metrics(Path(dataset))
            if not info.get('frames'):raise ValueError('Gaussian dataset has no frames')
            self.paths=replace(self.paths,gaussian_checkpoint=Path(checkpoint),gaussian_dataset=Path(dataset))
            self.gaussian_frame.setRange(0,len(info['frames'])-1)
            self._load_gaussian_info()
        except Exception as exc:self._error('Gaussian scene rejected',exc)

    def load_dataset(self,manifest:Path):
        start=time.perf_counter()
        try:
            paths=discover(self.root,manifest)
            scene=load_scene(paths.manifest)
        except Exception as exc:self._error('Dataset loading failed',exc);return
        self.paths=paths;self.scene=scene
        self.dataset_tree.clear()
        info=scene_summary(scene)
        root=QTreeWidgetItem(['TUM RGB-D',paths.manifest.parent.name]);self.dataset_tree.addTopLevelItem(root)
        for key,value in info.items():root.addChild(QTreeWidgetItem([key,str(value)]))
        frames=QTreeWidgetItem(['RGB / depth / pose frames',str(len(scene.frames))]);root.addChild(frames)
        for i,frame in enumerate(scene.frames):
            item=QTreeWidgetItem([f'Frame {i:04d}',f'{frame.timestamp_s:.6f}'])
            item.setData(0,Qt.ItemDataRole.UserRole,i);frames.addChild(item)
        for name,path in [('TSDF mesh',paths.mesh),('Point cloud',paths.cloud),('RTAB trajectory',paths.slam_manifest),
                          ('Semantic voxels',paths.semantic_map),('Gaussian scene',paths.gaussian_checkpoint)]:
            root.addChild(QTreeWidgetItem([name,str(path) if path else 'missing']))
        root.setExpanded(True);frames.setExpanded(False)
        self.frame_spin.setRange(0,len(scene.frames)-1);self.show_frame(0)
        self.start_spin.setRange(0,len(scene.frames)-1)
        self.stop_spin.setRange(1,len(scene.frames));self.stop_spin.setValue(len(scene.frames))
        self.inspector.setPlainText(json.dumps(info,indent=2))
        self._init_semantic_classes();self.load_evaluation();self._load_gaussian_info()
        if paths.gaussian_dataset:
            try:self.gaussian_frame.setRange(0,len(read_metrics(paths.gaussian_dataset)['frames'])-1)
            except (KeyError,ValueError):self.gaussian_frame.setRange(0,len(scene.frames)-1)
        self.dataset_load_s=time.perf_counter()-start
        self.log(f'Loaded {len(scene.frames)} TUM frames in {self.dataset_load_s:.3f}s')

    def _tree_click(self,item,_column):
        index=item.data(0,Qt.ItemDataRole.UserRole)
        if index is not None:self.frame_spin.setValue(int(index))
        elif item.text(0)=='TSDF mesh':self.load_mesh()
        elif item.text(0)=='Point cloud':self.load_cloud()
        elif item.text(0)=='RTAB trajectory':self.load_path(True)
        elif item.text(0)=='Semantic voxels':self.load_semantics()

    def show_frame(self,index):
        if self.scene is None:return
        try:
            rgb,depth=preview_frame(self.scene,index)
            self.rgb_preview.setPixmap(image_label(rgb))
            valid=np.isfinite(depth)&(depth>.1)
            gray=np.uint8(np.clip(depth/4,0,1)*255);gray[~valid]=0
            self.depth_preview.setPixmap(image_label(gray))
            self.show_depth_result()
        except Exception as exc:self._error('Frame preview failed',exc)

    def _set_actor(self,name,data,**kwargs):
        if name in self.actors:self.plotter.remove_actor(self.actors[name]);del self.actors[name]
        actor=self.plotter.add_mesh(data,name=name,**kwargs)
        self.actors[name]=actor
        actor.SetVisibility(self.visible_checks.get(name,QCheckBox()).isChecked())
        self.plotter.render();return actor

    def _set_visible(self,name,value):
        if name=='Axes':
            if value:self.plotter.show_axes()
            else:self.plotter.hide_axes()
            return
        actor=self.actors.get(name)
        if actor:actor.SetVisibility(value);self.plotter.render()

    def _refresh_points(self):
        for name in ('Cloud','Semantics'):
            actor=self.actors.get(name)
            if actor:actor.GetProperty().SetPointSize(self.point_size.value())
        self.plotter.render()

    def _refresh_color(self):
        if self.paths and self.paths.mesh and 'Mesh' in self.actors:self.load_mesh()
        if self.paths and self.paths.cloud and 'Cloud' in self.actors:self.load_cloud()

    def load_mesh(self):
        if not self.paths or not self.paths.mesh:self._error('Mesh unavailable','No TSDF mesh for this dataset');return
        start=time.perf_counter()
        try:
            mesh=pv.read(self.paths.mesh)
            if mesh.n_cells==0:raise ValueError('Mesh is empty')
            if self.color_mode.currentIndex()==1:
                mesh['height_m']=mesh.points[:,2]
                self._set_actor('Mesh',mesh,scalars='height_m',cmap='viridis',show_edges=False)
            else:self._set_actor('Mesh',mesh,color='#9db8c9',show_edges=False)
            self.mesh_load_s=time.perf_counter()-start
            self.inspector.setPlainText(json.dumps({'source':str(self.paths.mesh),'vertices':mesh.n_points,
                'triangles':mesh.n_cells,'bounds_m':mesh.bounds,'load_s':self.mesh_load_s},indent=2))
            self.fit_scene();self.log(f'TSDF mesh: {mesh.n_points} vertices, {mesh.n_cells} triangles')
        except Exception as exc:self._error('Mesh loading failed',exc)

    def load_cloud(self):
        if not self.paths or not self.paths.cloud:self._error('Cloud unavailable','No reconstructed point cloud');return
        try:
            cloud=pv.read(self.paths.cloud)
            kwargs={'style':'points','point_size':self.point_size.value(),'render_points_as_spheres':True}
            if self.color_mode.currentIndex()==1:
                cloud['height_m']=cloud.points[:,2]
                kwargs.update(scalars='height_m',cmap='viridis')
            else:kwargs['color']='#7ccab3'
            self._set_actor('Cloud',cloud,**kwargs)
            self.log(f'Point cloud: {cloud.n_points} points')
        except Exception as exc:self._error('Cloud loading failed',exc)

    def load_path(self,estimated=False):
        if not self.paths:return
        source=self.paths.slam_manifest if estimated else self.paths.manifest
        if source is None:self._error('SLAM unavailable','RTAB-Map trajectory is missing; GT was not substituted');return
        try:
            xyz=trajectory(source)
            line=pv.lines_from_points(xyz)
            name='SLAM path' if estimated else 'GT path'
            self._set_actor(name,line,color='#ffb66f' if estimated else '#4fcfff',line_width=4)
            self.log(f'{name}: {len(xyz)} camera positions')
            self.fit_scene()
        except Exception as exc:self._error('Trajectory loading failed',exc)

    def load_frustums(self):
        if not self.scene:return
        try:
            vertices,lines=camera_frustums(self.scene)
            geometry=pv.PolyData(vertices);geometry.lines=lines
            self._set_actor('Frustums',geometry,color='#d096f4',line_width=2)
            self.log(f'GT camera frustums: {len(vertices)//5}')
        except Exception as exc:self._error('Frustum loading failed',exc)

    def _init_semantic_classes(self):
        self.semantic_class.blockSignals(True);self.semantic_class.clear();self.semantic_class.addItem('All classes',None)
        if self.paths and self.paths.semantic_map:
            semantic=cached_semantic(self.paths.semantic_map)
            present={int(i) for i in semantic.statistics()['voxels_per_class']}
            colors=palette(len(semantic.id_to_label))
            for cid in sorted(present):
                swatch=QPixmap(14,14)
                swatch.fill(QColor(*[int(x) for x in colors[cid]]))
                self.semantic_class.addItem(QIcon(swatch),f'{cid}: {semantic.id_to_label[cid]}',cid)
        self.semantic_class.blockSignals(False)

    def load_semantics(self):
        if not self.paths or not self.paths.semantic_map:self._error('Semantics unavailable','Phase 7 semantic map missing');return
        try:
            cid=self.semantic_class.currentData()
            xyz,colors,stats=semantic_geometry(self.paths.semantic_map,cid)
            if len(xyz):
                mesh=pv.PolyData(xyz);mesh['rgb']=colors
                self._set_actor('Semantics',mesh,scalars='rgb',rgb=True,style='points',
                                point_size=self.point_size.value(),render_points_as_spheres=True)
            elif 'Semantics' in self.actors:
                self.plotter.remove_actor(self.actors.pop('Semantics'))
            self.semantic_info.setPlainText(json.dumps({'displayed_voxels':len(xyz),
                'consistency_not_accuracy':True,'map_statistics':stats},indent=2))
            self.log(f'Semantics: {len(xyz)} voxels, class={cid if cid is not None else "all"}')
            self.fit_scene()
        except Exception as exc:self._error('Semantic map failed',exc)

    def query_semantics(self):
        if not self.paths or not self.paths.semantic_map:return
        try:
            xyz=np.array([float(x) for x in self.query_edit.text().split(',')])
            if xyz.shape!=(3,):raise ValueError('Enter x, y, z in meters')
            result=semantic_query(self.paths.semantic_map,xyz)
            if result is None:text='Unknown / unoccupied voxel'
            else:
                taxonomy=cached_semantic(self.paths.semantic_map).id_to_label
                text=f'{taxonomy[result["label_id"]]} (ID {result["label_id"]}), confidence {result["distribution"][result["label_id"]]:.3f}, {result["frames_observed"]} frames'
            self.query_result.setText(text);self.log(f'Semantic query {xyz.tolist()}: {text}')
        except Exception as exc:self._error('Semantic query failed',exc)

    def _picked_point(self,xyz):
        if xyz is None:return
        self.viewport_status.setText(f'Picked world position: {xyz[0]:.3f}, {xyz[1]:.3f}, {xyz[2]:.3f} m')
        self.query_edit.setText(f'{xyz[0]:.4f}, {xyz[1]:.4f}, {xyz[2]:.4f}')

    def show_depth_result(self):
        if not self.paths or not self.scene:return
        i=self.frame_spin.value();model='depth_anything_v2_small' if self.depth_model.currentIndex()==0 else 'marigold_v1_1'
        kind=self.depth_method.currentIndex()
        if kind==0:
            image=self.root/'outputs/depth_phase4'/model/f'frame_{i:04d}_raw.png'
            explanation='Raw model output; relative / affine-invariant units, not meters.'
        elif kind==1:
            image=self.root/'outputs/depth_phase4/evaluation'/model/f'frame_{i:04d}_robust_meters.png'
            explanation='Oracle robust alignment used GT sensor depth; meters, not independent metric prediction.'
        elif kind==2:
            image=self.root/'outputs/phase6'/model/f'frame_{i:04d}_sparse_m.npy'
            explanation='Per-frame sparse alignment uses RTAB RGB-D geometry; meters, not RGB-only.'
        elif kind==3:
            image=self.scene.frames[i].depth_path
            explanation='TUM RGB-D sensor depth; meters. Used only as a reference for evaluation.'
        else:
            image=self.root/'outputs/depth_phase4/evaluation'/model/f'frame_{i:04d}_robust_heldout_abs_error_m.npy'
            explanation='Absolute error on Phase 4 held-out pixels after oracle GT alignment; meters.'
        if kind==3:
            _,values=preview_frame(self.scene,i)
            gray=np.uint8(np.clip(values/4,0,1)*255)
            gray[values<=0]=0
            self.depth_image.setPixmap(image_label(gray,310))
        elif image.is_file() and image.suffix=='.npy':
            values=np.load(image,mmap_mode='r')
            valid=np.isfinite(values)&(values>0)
            gray=np.uint8(np.clip(values/(1 if kind==4 else 4),0,1)*255)
            gray[~valid]=0
            self.depth_image.setPixmap(image_label(gray,310))
        elif image.is_file():
            self.depth_image.setPixmap(QPixmap(str(image)).scaledToWidth(310,Qt.TransformationMode.SmoothTransformation))
        else:self.depth_image.setText(f'Missing {image.name}')
        self.depth_info.setPlainText(f'{explanation}\n\nSource: {image}\nFrame: {i}\n'
            'See Evaluation tab for actual AbsRel / RMSE / δ1 records.')

    def _load_gaussian_info(self):
        if not self.paths:return
        training=self.root/'outputs/phase5/gt_pose_scene/training.json'
        evaluation=self.root/'outputs/phase5/gt_pose_scene/rendering/evaluation.json'
        text={'checkpoint':str(self.paths.gaussian_checkpoint) if self.paths.gaussian_checkpoint else 'missing',
              'dataset':str(self.paths.gaussian_dataset) if self.paths.gaussian_dataset else 'missing'}
        for name,path in [('training',training),('heldout_evaluation',evaluation)]:
            if path.is_file():text[name]=read_metrics(path)
        self.gaussian_info.setPlainText(json.dumps(text,indent=2))

    def render_gaussian(self):
        if not self.paths:return
        self.pipeline_combo.setCurrentText('Gaussian render')
        self.run_pipeline()

    def _show_rendered(self):
        path=self.root/'outputs/phase9/gaussian_render'/f'frame_{self.gaussian_frame.value():04d}_render.png'
        if path.is_file():
            self.gaussian_image.setPixmap(QPixmap(str(path)).scaledToWidth(310,Qt.TransformationMode.SmoothTransformation))
            self.log(f'Actual CUDA Gaussian render: {path}')

    def _inverse_pose_changed(self):
        if self.inverse_pose.currentIndex()==0:
            mesh=self.paths.mesh if self.paths and self.paths.mesh else self.root/'outputs/tum_fr3_structure_notexture_far/tsdf/mesh.ply'
        else:mesh=self.root/'outputs/slam/fr3_structure_notexture_far/comparison/slam_tsdf/mesh.ply'
        self.inverse_mesh.setText(str(mesh))

    def inverse_output(self):
        return self.root/'outputs/phase10'/('tum_gt' if self.inverse_pose.currentIndex()==0 else 'tum_rtab')

    def run_inverse(self):
        self.pipeline_combo.setCurrentText('Inverse rendering')
        self.run_pipeline()

    def load_inverse(self):
        output=self.inverse_output();path=output/'results.json'
        if not path.is_file():self._error('Inverse results missing',path);return
        try:
            result=read_metrics(path)
            test=result['metrics']['test']
            self.inverse_info.setPlainText(json.dumps({'pose_source':result['config']['pose_source'],
                'train_correspondences':result['train_correspondences'],
                'test_frames':[{k:v for k,v in row.items() if k!='methods'}|{'methods':row['methods']} for row in test],
                'light_regularized':result['light_regularized'],
                'caution':'Albedo and light are gauge constrained; TUM has no material/light ground truth.'},indent=2))
            for spin,value in zip(self.inverse_light,result['light_regularized']):spin.blockSignals(True);spin.setValue(value);spin.blockSignals(False)
            self.show_inverse_image()
            self.log(f'Loaded actual inverse rendering results: {path}')
        except Exception as exc:self._error('Inverse results invalid',exc)

    def show_inverse_image(self):
        if not hasattr(self,'inverse_image'):return
        output=self.inverse_output();frame=self.inverse_frame.value();mode=self.inverse_mode.currentText()
        method=self.inverse_method.currentText()
        filename={'original':'original.png','normals':'normals.png','mask':'mask.png'}.get(mode,f'{method}_{mode}.png')
        path=output/f'frame_{frame:04d}'/filename
        if path.is_file():
            self.inverse_image.setPixmap(QPixmap(str(path)).scaledToWidth(310,Qt.TransformationMode.SmoothTransformation))
        else:self.inverse_image.setText(f'No saved result: {filename}')

    def show_inverse_material(self):
        output=self.inverse_output();saved=output/'scene_material.npz';mesh_path=Path(self.inverse_mesh.text())
        if not saved.is_file() or not mesh_path.is_file():return
        try:
            with np.load(saved) as record:
                albedo=np.asarray(record['albedo']);normals=np.asarray(record['normal'])
            if self.inverse_method.currentText()=='lambertian_unregularized':
                self.log('3D material preview uses saved regularized albedo')
            mesh=pv.read(mesh_path)
            if mesh.n_cells!=len(albedo):raise ValueError('Saved albedo does not match mesh triangle count')
            light=np.array([spin.value() for spin in self.inverse_light])
            shading=.6+np.maximum(0,normals@light)
            linear=np.clip(albedo*shading[:,None],0,1)
            srgb=np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)
            mesh.cell_data['material_rgb']=np.uint8(np.clip(srgb*255,0,255))
            self._set_actor('Inverse material',mesh,scalars='material_rgb',rgb=True)
            if 'Mesh' in self.actors:self.visible_checks['Mesh'].setChecked(False)
            self.fit_scene();self.log(f'3D inverse material: {len(albedo)} triangles; light {light.tolist()}')
        except Exception as exc:self._error('3D inverse material failed',exc)

    def load_evaluation(self):
        self.metrics.clear()
        for name,path in metric_files(self.root):
            parent=QTreeWidgetItem([name,'missing' if not path.is_file() else 'loading']);self.metrics.addTopLevelItem(parent)
            if not path.is_file():continue
            try:
                data=read_metrics(path)
                parent.setText(1,metric_summary(name,data))
                parent.addChild(QTreeWidgetItem(['source',str(path)]))
                for key,value in flatten(data):
                    parent.addChild(QTreeWidgetItem([key,value]))
            except Exception as exc:parent.addChild(QTreeWidgetItem(['ERROR',str(exc)]))

    def fit_scene(self):
        if self.actors:self.plotter.reset_camera();self.plotter.render()

    def reset_camera(self):
        self.plotter.view_isometric();self.fit_scene()

    def choose_output(self):
        directory=QFileDialog.getExistingDirectory(self,'Select output directory',str(self.root/'outputs'))
        if directory:self.output_edit.setText(directory)

    def run_pipeline(self):
        if self.current_job is not None:self._error('Pipeline busy','Cancel or finish the current process first');return
        if not self.paths:return
        name=self.pipeline_combo.currentText()
        output=Path(self.output_edit.text()).expanduser()
        if name=='Gaussian render':output=self.root/'outputs/phase9/gaussian_render'
        if name=='Inverse rendering':output=self.inverse_output()
        output.parent.mkdir(parents=True,exist_ok=True)
        try:
            start,stop=self.start_spin.value(),self.stop_spin.value()
            if not 0<=start<stop<=len(self.scene.frames):raise ValueError('Frame range must have start < stop')
            manifest=self.paths.manifest
            if name=='Inverse rendering' and self.inverse_pose.currentIndex()==1:
                if not self.paths.slam_manifest:raise FileNotFoundError('RTAB trajectory unavailable for this dataset')
                manifest=self.paths.slam_manifest
            if name not in ('Gaussian render','Inverse rendering') and (start!=0 or stop!=len(self.scene.frames)):
                manifest=slice_manifest(manifest,output/'input_subset.json',start,stop)
            job=build_job(name,self.root,manifest,output,
                voxel_size_m=self.voxel_spin.value(),truncation_m=self.trunc_spin.value(),
                frame_index=self.gaussian_frame.value(),gaussian_dataset=self.paths.gaussian_dataset,
                gaussian_checkpoint=self.paths.gaussian_checkpoint,
                inverse_mesh=Path(self.inverse_mesh.text()),inverse_regularization=self.inverse_reg.value(),
                pose_source='GT' if self.inverse_pose.currentIndex()==0 else 'RTAB')
        except Exception as exc:self._error('Pipeline dependency missing',exc);return
        output.mkdir(parents=True,exist_ok=True)
        process=QProcess(self);process.setProgram(str(job.program));process.setArguments(list(job.arguments))
        process.setWorkingDirectory(str(self.root))
        process.readyReadStandardOutput.connect(lambda:self._read_process(process,False))
        process.readyReadStandardError.connect(lambda:self._read_process(process,True))
        process.finished.connect(lambda code,status:self._job_finished(job,code,status))
        process.errorOccurred.connect(lambda error:self.log(f'Process error: {error}'))
        self.current_job=(job,process,time.perf_counter())
        self.run_button.setEnabled(False);self.cancel_button.setEnabled(True)
        self.progress.setRange(0,0)
        self.log(f'Running {name}: {job.program} {" ".join(job.arguments)}')
        process.start()

    def _read_process(self,process,stderr):
        data=bytes(process.readAllStandardError() if stderr else process.readAllStandardOutput())
        for line in data.decode('utf-8',errors='replace').splitlines():self.log(line)

    def cancel_pipeline(self):
        if self.current_job:
            self.current_job[1].terminate();self.log('Cancellation requested')
            QTimer.singleShot(3000,lambda:self.current_job[1].kill() if self.current_job and self.current_job[1].state()!=QProcess.ProcessState.NotRunning else None)

    def _job_finished(self,job,code,status):
        elapsed=time.perf_counter()-self.current_job[2] if self.current_job else 0
        success=code==0 and status==QProcess.ExitStatus.NormalExit
        self.history.append({'name':job.name,'output':str(job.output),'elapsed_s':elapsed,
                             'exit_code':code,'success':success})
        history=self.root/'outputs/phase9/history.json';history.parent.mkdir(parents=True,exist_ok=True)
        history.write_text(json.dumps(self.history,indent=2),encoding='utf-8')
        self.log(f'{job.name} {"completed" if success else "FAILED"}: exit={code}, {elapsed:.1f}s')
        self.current_job=None;self.run_button.setEnabled(True);self.cancel_button.setEnabled(False)
        self.progress.setRange(0,1);self.progress.setValue(1 if success else 0)
        if success:
            if job.name=='TSDF':
                self.paths=ProjectPaths(self.paths.root,self.paths.manifest,self.paths.slam_manifest,
                    job.output/'mesh.ply',job.output/'cloud.ply',self.paths.semantic_map,
                    self.paths.gaussian_checkpoint,self.paths.gaussian_dataset)
                self.load_mesh()
            elif job.name=='Gaussian render':self._show_rendered()
            elif job.name=='Inverse rendering':self.load_inverse()
            self.load_evaluation()
        else:self._error('Pipeline failed',f'{job.name} returned exit code {code}. Inspect the console and {job.output}.')

    def _update_memory(self):
        p=self._gui_process;rss=p.memory_info().rss;self.gui_peak_rss_bytes=max(self.gui_peak_rss_bytes,rss)
        self.performance.setText(f'GUI RSS: {rss/1e6:.1f} MB   •   CPU: {p.cpu_percent():.0f}%   •   Jobs: {len(self.history)}')

    def save_session(self):
        if not self.paths:return
        path=self.root/'outputs/phase9/session.json'
        save_session(path,{'manifest':str(self.paths.manifest),'mesh':str(self.paths.mesh) if self.paths.mesh else None,
            'cloud':str(self.paths.cloud) if self.paths.cloud else None,
            'slam_manifest':str(self.paths.slam_manifest) if self.paths.slam_manifest else None,
            'semantic_map':str(self.paths.semantic_map) if self.paths.semantic_map else None,
            'gaussian_checkpoint':str(self.paths.gaussian_checkpoint) if self.paths.gaussian_checkpoint else None,
            'gaussian_dataset':str(self.paths.gaussian_dataset) if self.paths.gaussian_dataset else None,
            'frame_index':self.frame_spin.value(),'workspace':'main','point_size':self.point_size.value(),
            'voxel_size_m':self.voxel_spin.value(),'truncation_m':self.trunc_spin.value(),
            'frame_start':self.start_spin.value(),'frame_stop':self.stop_spin.value(),
            'color_mode':self.color_mode.currentIndex(),
            'output':self.output_edit.text(),
            'camera_state':[[float(x) for x in row] for row in self.plotter.camera_position],
            'visible':{key:check.isChecked() for key,check in self.visible_checks.items()}},self.root)
        self.log(f'Saved session: {path}')

    def open_session(self):
        path,_=QFileDialog.getOpenFileName(self,'Open RECON-X session',str(self.root/'outputs/phase9'),'JSON (*.json)')
        if not path:return
        try:
            data=load_session(Path(path),self.root)
            self.load_dataset(Path(data['manifest']))
            if self.paths:
                paths=self.paths
                self.paths=ProjectPaths(paths.root,paths.manifest,
                    Path(data['slam_manifest']) if data.get('slam_manifest') else paths.slam_manifest,
                    Path(data['mesh']) if data.get('mesh') else paths.mesh,
                    Path(data['cloud']) if data.get('cloud') else paths.cloud,
                    Path(data['semantic_map']) if data.get('semantic_map') else paths.semantic_map,
                    Path(data['gaussian_checkpoint']) if data.get('gaussian_checkpoint') else paths.gaussian_checkpoint,
                    Path(data['gaussian_dataset']) if data.get('gaussian_dataset') else paths.gaussian_dataset)
            self.frame_spin.setValue(int(data.get('frame_index',0)))
            self.point_size.setValue(int(data.get('point_size',3)))
            self.voxel_spin.setValue(float(data.get('voxel_size_m',.03)))
            self.trunc_spin.setValue(float(data.get('truncation_m',.09)))
            self.start_spin.setValue(int(data.get('frame_start',0)))
            self.stop_spin.setValue(int(data.get('frame_stop',len(self.scene.frames))))
            self.color_mode.setCurrentIndex(int(data.get('color_mode',0)))
            self.output_edit.setText(data.get('output',str(self.root/'outputs/phase9/tsdf_gui')))
            for key,value in data.get('visible',{}).items():
                if key in self.visible_checks:self.visible_checks[key].setChecked(bool(value))
            if data.get('camera_state'):
                self.plotter.camera_position=data['camera_state'];self.plotter.render()
            self.log(f'Opened session: {path}')
        except Exception as exc:self._error('Session loading failed',exc)

    def closeEvent(self,event):
        if self.current_job:self.cancel_pipeline()
        if getattr(self,'video_process',None):self.cancel_video_pipeline()
        self.plotter.close();super().closeEvent(event)


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=Path)
    p.add_argument('--screenshot',type=Path,help='Save a screenshot of the running window and exit')
    p.add_argument('--run-tsdf-smoke',action='store_true')
    p.add_argument('--phase11-output',type=Path,help='Load a completed RGB video run')
    args=p.parse_args(argv)
    app=QApplication(sys.argv[:1])
    window=MainWindow(ROOT,args.manifest);window.show()
    if args.screenshot:
        def capture():
            try:
                if args.phase11_output:
                    window.video_output=args.phase11_output.resolve()
                    window.right_tabs.setCurrentIndex(window.right_tabs.count()-1)
                    window.load_video_results()
                else:
                    window.load_mesh();window.load_path(False);window.load_path(True);window.load_semantics()
                    window.load_evaluation()
                app.processEvents();window.plotter.render()
                args.screenshot.parent.mkdir(parents=True,exist_ok=True)
                if not window.grab().save(str(args.screenshot)):
                    raise RuntimeError(f'Screenshot failed: {args.screenshot}')
                print(f'GUI_SCREENSHOT={args.screenshot.resolve()}',flush=True)
                if args.run_tsdf_smoke:
                    window.pipeline_combo.setCurrentText('TSDF')
                    window.output_edit.setText(str(ROOT/'outputs/phase9/smoke_tsdf'))
                    window.run_pipeline()
                    def monitor():
                        if window.current_job is None:
                            app.exit(0 if window.history and window.history[-1]['name']=='TSDF' and window.history[-1]['success'] else 3)
                        else:QTimer.singleShot(1000,monitor)
                    QTimer.singleShot(1000,monitor)
                else:app.quit()
            except Exception as exc:
                print(f'GUI_SMOKE_ERROR={exc}',file=sys.stderr,flush=True);app.exit(2)
        QTimer.singleShot(3500,capture)
    elif args.phase11_output:
        window.video_output=args.phase11_output.resolve()
        QTimer.singleShot(0,lambda:window.right_tabs.setCurrentIndex(window.right_tabs.count()-1))
        QTimer.singleShot(0,window.load_video_results)
    return app.exec()


if __name__=='__main__':raise SystemExit(main())
