"""User-oriented shell; established viewer and workflow widgets are reused."""
from pathlib import Path
import json
import time
from PySide6.QtCore import Qt,QTimer,QUrl
from PySide6.QtGui import QAction,QDesktopServices
from PySide6.QtWidgets import QWidget,QVBoxLayout,QHBoxLayout,QLabel,QPushButton,QTabWidget,QScrollArea,QDockWidget,QCheckBox,QFileDialog,QMessageBox
from .ui_appearance import QComboBox,KO,localize
from .app_export import FORMATS
from .app_project import atomic_json
from .app_runtime import runtime,VERSION

KO.update({'Project':'프로젝트','Import Video':'영상 가져오기','Import RGB-D Dataset':'RGB-D 데이터셋 가져오기',
 'Scene':'장면','Layers':'레이어','Appearance':'외형','Analysis':'분석','Export':'내보내기',
 'Advanced Settings':'고급 설정','Start Reconstruction':'복원 시작','Detailed logs':'상세 로그',
 'Perspective':'원근 투영','Open output folder':'출력 폴더 열기','Export selected output':'선택 항목 내보내기',
 'No input imported':'아직 입력이 없습니다','Runtime assets / recent projects':'실행 환경 / 최근 프로젝트','No export yet':'내보낸 결과가 없습니다','Native world coordinates are preserved. No texture export.':'원본 월드 좌표를 유지합니다. 텍스처는 내보내지 않습니다.','From 2D Images to 3D Worlds':'2D 이미지에서 3D 세계로','RGB Video → 3D':'RGB 영상 → 3D','RGB-D Reconstruction':'RGB-D 복원'})

def panel():
    w=QWidget();return w,QVBoxLayout(w)

def button(layout,text,callback):
    b=QPushButton(text);b.clicked.connect(callback);layout.addWidget(b);return b

def scroll(widget):
    area=QScrollArea();area.setWidgetResizable(True);area.setWidget(widget);return area

def install(window):
    w=window;split=w.centralWidget();old_left=w.dataset_panel;center=split.widget(1);right=w.right_tabs
    right.currentChanged.disconnect()
    pages={right.tabText(i):right.widget(i) for i in range(right.count())}
    while right.count():right.removeTab(0)
    for page in pages.values():page.show()
    left,ll=panel();ll.addWidget(QLabel('Rebuilder 3D'))
    for title,fn in [('New project',w.new_project),('Import Video',w.select_video),('Import RGB-D Dataset',w.open_dataset),('Open project',w.open_project_dialog),('Save project',w.save_project_dialog)]:button(ll,title,fn)
    w.input_summary=QLabel('No input imported');w.input_summary.setWordWrap(True);ll.addWidget(w.input_summary)
    w.workflow=QComboBox();w.workflow.addItems(['RGB Video → 3D','RGB-D Reconstruction']);ll.addWidget(w.workflow)
    w.start_reconstruction=button(ll,'Start Reconstruction',lambda:w.run_video_pipeline() if w.workflow.currentIndex()==0 else w.run_pipeline())
    button(ll,'■ Cancel',w.cancel_pipeline)
    advanced=QTabWidget();advanced.addTab(scroll(old_left),'RGB-D');advanced.addTab(scroll(pages['RGB Video']),'RGB Video')
    expand=QCheckBox('Advanced Settings');expand.toggled.connect(advanced.setVisible);ll.addWidget(expand);ll.addWidget(advanced,1);advanced.hide()
    w.stage_summary=QLabel('Ready');w.stage_summary.setWordWrap(True);ll.addWidget(w.stage_summary);ll.addStretch()
    # Replace first splitter slot while retaining the old controls in Advanced.
    split.insertWidget(0,scroll(left));w.dataset_panel=left
    scene,sl=panel();w.scene_summary=QLabel('No reconstruction results');w.scene_summary.setWordWrap(True);w.scene_summary.setAlignment(Qt.AlignmentFlag.AlignTop);sl.addWidget(w.scene_summary,0)
    sl.addWidget(pages['Inspector'])
    assets=QCheckBox('Runtime assets / recent projects');sl.addWidget(assets);sl.addWidget(scroll(pages['Home']));assets.toggled.connect(sl.itemAt(sl.count()-1).widget().setVisible);sl.itemAt(sl.count()-1).widget().hide()
    layers,ly=panel();appearance=QTabWidget();analysis=pages['Evaluation']
    for name in ('Depth','3DGS','Inverse'):appearance.addTab(scroll(pages[name]),name)
    appearance.addTab(scroll(pages['Semantics']),'Semantics')
    # Move layer loading buttons and toggles out of the viewport.
    toolbar=center.layout().itemAt(0).layout()
    for i in range(toolbar.count()-1,1,-1):
        item=toolbar.takeAt(i)
        if item.widget():ly.insertWidget(0,item.widget())
    for check in w.visible_checks.values():ly.addWidget(check)
    w.mesh_view_mode=QComboBox();w.mesh_view_mode.addItems(['Solid','Wireframe','RGB Vertex Color','Semantic Color'])
    w.mesh_view_mode.currentIndexChanged.connect(w.apply_mesh_view_mode)
    ly.addWidget(QLabel('Mesh display'));ly.addWidget(w.mesh_view_mode)
    from .photoreal_viewer import Dollhouse
    w.dollhouse=Dollhouse(w,ly)
    center.layout().removeWidget(w.color_mode);w.color_mode.hide()
    grid=QCheckBox('Reference grid');grid.toggled.connect(lambda visible:w.plotter.show_grid(color='#8591a6') if visible else w.plotter.remove_bounds_axes());ly.addWidget(grid)
    ly.addWidget(QLabel('Point size'));ly.addWidget(w.point_size);ly.addStretch()
    perspective=QCheckBox('Perspective');perspective.setChecked(True)
    perspective.toggled.connect(lambda value:setattr(w.plotter.camera,'parallel_projection',not value));toolbar.addWidget(perspective)
    w.welcome,wel=panel();title=QLabel('Rebuilder 3D\nFrom 2D Images to 3D Worlds');title.setAlignment(Qt.AlignmentFlag.AlignCenter);title.setStyleSheet('font-size:24px;padding:24px;color:#ddd9ff');wel.addWidget(title)
    button(wel,'Import Video',w.select_video);button(wel,'Open project',w.open_project_dialog);center.layout().insertWidget(1,w.welcome)
    w.welcome.setMinimumHeight(260)
    export,el=panel();w.export_kind=QComboBox()
    for label,key in [('3D Mesh','mesh'),('Textured Mesh','textured_mesh'),('Point Cloud','cloud'),('Semantic Data','semantic'),('Gaussian Scene','gaussian'),('Camera & Analysis Data','camera')]:w.export_kind.addItem(label,key)
    w.export_format=QComboBox();el.addWidget(w.export_kind);el.addWidget(w.export_format)
    w.export_note=QLabel('Native world coordinates are preserved. No texture export.');w.export_note.setWordWrap(True);el.addWidget(w.export_note)
    w.export_button=button(el,'Export selected output',lambda:export_dialog(w))
    w.export_status=QLabel('No export yet');w.export_status.setWordWrap(True);el.addWidget(w.export_status)
    w.export_folder=button(el,'Open output folder',lambda:QDesktopServices.openUrl(QUrl.fromLocalFile(str(w.last_export.parent))))
    w.export_folder.setEnabled(False);el.addStretch()
    def export_note():
        kind=w.export_kind.currentData();fmt=w.export_format.currentText()
        if kind=='mesh' and fmt=='obj':note='OBJ exports geometry only; captured RGB is omitted.'
        elif kind=='mesh' and fmt in ('ply','glb'):note='PLY/GLB preserve captured RGB vertex colors; no UV texture.'
        elif kind=='textured_mesh':note='UV GLB preserves captured RGB texture; geometry accuracy unchanged.'
        else:note='Native world coordinates are preserved. No texture export.'
        shown=localize_note(note,w.language)
        w.export_note.setProperty('_i18n_source_text',note)
        w.export_note.setProperty('_i18n_last_text',shown)
        w.export_note.setText(shown)
    def formats():w.export_format.clear();w.export_format.addItems(FORMATS[w.export_kind.currentData()]);export_note();refresh(w)
    w.export_kind.currentIndexChanged.connect(formats);w.export_format.currentIndexChanged.connect(export_note);formats()
    # Remove the now-empty layer row from the center.
    for i in range(center.layout().count()-1,-1,-1):
        item=center.layout().itemAt(i)
        if item.layout() and item.layout()!=toolbar:
            while item.layout().count():
                child=item.layout().takeAt(0)
                if child.widget():child.widget().hide()
    for name,page in [('Scene',scene),('Layers',layers),('Appearance',appearance),('Analysis',analysis),('Export',export)]:right.addTab(page,name)
    right.setMinimumWidth(300);right.setMaximumWidth(600);split.setSizes([300,850,400])
    for dock in w.findChildren(QDockWidget):
        dock.setWindowTitle('Status');w.console.hide();toggle=QCheckBox('Detailed logs');toggle.toggled.connect(w.console.setVisible);dock.widget().layout().insertWidget(0,toggle)
    about=QAction('About Rebuilder 3D',w);about.triggered.connect(lambda:QMessageBox.information(w,'Rebuilder 3D',f'Rebuilder 3D {VERSION}\nFrom 2D Images to 3D Worlds\nRGB Video-Based 3D Scene Reconstruction & Understanding\nRGB-only scale is arbitrary. Research outputs are not physical ground truth.'));w.menuBar().addAction(about)
    w.shell_timer=QTimer(w);w.shell_timer.timeout.connect(lambda:refresh(w));w.shell_timer.start(750)
    localize(w,w.language)

def localize_note(note,language):
    if language!='ko':return note
    return {'OBJ exports geometry only; captured RGB is omitted.':'OBJ는 형상만 내보냅니다. 실제 RGB 색상은 포함되지 않습니다.',
            'PLY/GLB preserve captured RGB vertex colors; no UV texture.':'PLY/GLB는 실제 RGB 정점 색상을 보존합니다. UV 텍스처는 없습니다.',
            'Native world coordinates are preserved. No texture export.':'원본 월드 좌표를 유지합니다. 텍스처는 내보내지 않습니다.',
            'UV GLB preserves captured RGB texture; geometry accuracy unchanged.':'UV GLB는 실제 RGB 텍스처를 보존하며 형상 정확도는 바뀌지 않습니다.'}[note]

def sources(w):
    data=w._snapshot()
    if getattr(w,'video_view',False) and getattr(w,'video_output',None):
        root=w.video_output;report=root/'dense/dense_report.json'
        if report.is_file():
            dense=json.loads(report.read_text(encoding='utf-8'));data['mesh']=dense['mesh'];data['cloud']=dense.get('point_cloud')
        data['semantic_map']=str(root/'semantic/semantic_map.json')
        data['textured_glb']=str(root/'photoreal/texture/textured.glb')
    return data

def export_at(w,destination):
    data=sources(w);kind=w.export_kind.currentData();key={'mesh':'mesh','textured_mesh':'textured_glb','cloud':'cloud','semantic':'semantic_map','gaussian':'gaussian_checkpoint'}.get(kind)
    source=data.get(key) if key else None
    if kind!='camera' and (not source or not Path(source).is_file()):raise ValueError('No exportable '+kind+' data')
    out=w.workspace/'exports'/str(time.time_ns());out.mkdir(parents=True)
    request=out/'request.json';atomic_json(request,{'kind':kind,'source':source,'destination':str(destination),'snapshot':data})
    w.last_export=Path(destination);w.export_folder.setEnabled(False);w.export_status.setText('Exporting and validating…')
    return w._submit('Export',[('write and independently validate',runtime(w.root,'gsplat' if kind=='gaussian' else 'recon'),['-m','reconx.app_export',str(request)])],out)

def export_dialog(w):
    fmt=w.export_format.currentText();path,_=QFileDialog.getSaveFileName(w,'Export',str(w.workspace/('scene.'+fmt)),f'{fmt.upper()} (*.{fmt})')
    if path:
        try:export_at(w,Path(path))
        except Exception as exc:w._error('Export failed',exc)

def refresh(w):
    if not hasattr(w,'welcome'):return
    has_scene=any(k!='Axes' for k in w.actors);w.welcome.setVisible(not has_scene)
    w.plotter.setVisible(has_scene)
    for key,check in w.visible_checks.items():check.setEnabled(key=='Axes' or key in w.actors)
    data=sources(w);kind=w.export_kind.currentData();key={'mesh':'mesh','textured_mesh':'textured_glb','cloud':'cloud','semantic':'semantic_map','gaussian':'gaussian_checkpoint'}.get(kind)
    w.export_button.setEnabled(not w.manager.active and (bool(data.get('camera_poses') or data.get('video_output')) if kind=='camera' else bool(data.get(key) and Path(data[key]).is_file())))
    w.start_reconstruction.setEnabled(not w.manager.active and (bool(w.video_input.text().strip()) if w.workflow.currentIndex()==0 else w.scene is not None))
    mesh=w.actors.get('Mesh');dataset=mesh.mapper.dataset if mesh else None
    scale='Arbitrary (SfM Units)' if data['input_type']=='rgb_video' else data.get('scale_convention','unknown')
    frames=len(w.scene.frames) if w.scene and data['input_type']=='rgbd' else None;registered=None
    if data.get('video_output'):
        sfm=Path(data['video_output'])/'sfm/sfm.json'
        if sfm.is_file():r=json.loads(sfm.read_text(encoding='utf-8'));frames=r.get('input_frames');registered=r.get('registered_frames')
    records=w.manager.records[w.project_job_offset:] or w.project_data.get('jobs',[])
    latest=records[-1] if records else None
    samples=getattr(w.manager,'gpu_samples',[])
    if samples and w.manager.active:w.performance.setText(w.performance.text().split(' · GPU')[0]+f' · GPU {samples[-1]["device_used_MiB"]:.0f} MiB (whole device)')
    wall=latest.get('wall_s') if latest else None
    wall_text=f'{wall:.2f}' if isinstance(wall,(int,float)) else '—'
    point_actor=w.actors.get('Cloud') or w.actors.get('Semantics')
    point_count=point_actor.mapper.dataset.n_points if point_actor else 0
    w.scene_summary.setText(f'{w.project_file.stem if w.project_file else "Untitled project"}\nInput: {data["input_type"]}\nFrames: {frames if frames is not None else "—"} · Registered: {registered if registered is not None else "—"}\nPoints: {point_count}\nMesh vertices: {dataset.n_points if dataset else 0}\nTriangles: {dataset.n_cells if dataset else 0}\nScale: {scale}\nLast job time: {wall_text} s')
    if data['input_type']=='rgb_video' and getattr(w,'video_coverage_note',None):w.scene_summary.setText(w.scene_summary.text()+'\n'+w.video_coverage_note)
    if w.language=='ko':
        text=w.scene_summary.text()
        for en,ko in [('Untitled project','이름 없는 프로젝트'),('Input:','입력:'),('Frames:','프레임:'),('Registered:','등록 카메라:'),('Mesh vertices:','메시 정점:'),('Points:','포인트:'),('Triangles:','삼각형:'),('Scale:','스케일:'),('Last job time:','최근 작업 시간:'),('Arbitrary (SfM Units)','임의 스케일 (SfM 단위)')]:text=text.replace(en,ko)
        w.scene_summary.setText(text)
    if latest:
        states=[]
        for i,stage in enumerate(latest['stages']):
            state='completed' if i<latest['stage_index'] else ('running' if latest['status']=='running' else latest['status']) if i==latest['stage_index'] else ('skipped' if latest['status'] in ('failed','cancelled') else 'pending')
            states.append(f'{stage["name"]}: {state}')
        w.stage_summary.setText('\n'.join(states));w.progress.setFormat('%p% stages completed')
    else:
        w.stage_summary.setText('사용 가능' if w.language=='ko' else 'Ready');w.progress.setValue(0)
    if latest and latest['name']=='Export' and latest['status'] in ('completed','failed','cancelled'):
        result=Path(latest['output'])/'export_result.json'
        if latest['status']=='completed' and result.is_file():
            r=json.loads(result.read_text(encoding='utf-8'));w.export_status.setText(f'Validated · {r["bytes"]:,} bytes\n{r["path"]}');w.export_folder.setEnabled(True)
        else:w.export_status.setText('Export '+latest['status']+' — see detailed logs')
