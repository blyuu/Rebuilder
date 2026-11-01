"""Dollhouse controls for observed geometry and interactive RGB-only Gaussian rendering."""
import json,shutil
from pathlib import Path
import numpy as np
from PySide6.QtCore import QObject,QProcess,QProcessEnvironment,QTimer,Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel,QPushButton,QCheckBox,QSlider,QFileDialog
from .app_runtime import runtime,child_environment
from .ui_appearance import KO

KO.update({'Textured Mesh':'사진 텍스처 메시','Gaussian Scene':'가우시안 장면','Dollhouse / Fit':'공간 전체 보기',
 'Captured camera view':'촬영 위치에서 보기','Cutaway (view direction)':'단면 보기 (시선 방향)','Export textured GLB':'텍스처 GLB 내보내기',
 'Photorealistic full workflow':'사진 외형 전체 복원','Hide ceiling-facing surfaces':'천장 방향 표면 숨기기'})


def optical_pose(position,target,up):
    position,target,up=map(lambda x:np.asarray(x,float),(position,target,up))
    z=target-position;z/=np.linalg.norm(z)
    x=np.cross(z,up);x/=np.linalg.norm(x);y=np.cross(z,x)
    pose=np.eye(4);pose[:3,:3]=np.column_stack([x,y,z]);pose[:3,3]=position
    return pose


class Dollhouse(QObject):
    def __init__(self,w,layout):
        super().__init__(w);self.w=w;self.process=None;self.ready=False;self.busy=False;self.pending=False;self.enabled=False;self.buffer='';self.surface=None
        self.timer=QTimer(self);self.timer.setSingleShot(True);self.timer.timeout.connect(self.request)
        self.overlay=QLabel(w.plotter);self.overlay.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents);self.overlay.hide()
        w.plotter.camera.AddObserver('ModifiedEvent',lambda *_:self.timer.start(180) if self.enabled else None)
        self.cut=QCheckBox('Cutaway (view direction)');layout.addWidget(self.cut);self.cut.toggled.connect(self.apply_clip)
        self.slider=QSlider(Qt.Orientation.Horizontal);self.slider.setRange(0,100);self.slider.setValue(50);layout.addWidget(self.slider);self.slider.valueChanged.connect(self.apply_clip)
        for title,fn in [('Dollhouse / Fit',w.fit_scene),('Captured camera view',self.captured),('Export textured GLB',self.export)]:
            b=QPushButton(title);b.clicked.connect(fn);layout.addWidget(b)
        w.mesh_view_mode.addItems(['Textured Mesh','Gaussian Scene'])
        w.destroyed.connect(self.stop)

    def load(self,run):
        self.stop();self.run=Path(run);self.surface=None
        texture=self.run/'photoreal/texture';self.texture_dir=texture
        if (texture/'surface.npz').is_file():
            import pyvista as pv
            data=np.load(texture/'surface.npz');self.surface=pv.PolyData(data['vertices'],np.column_stack([np.full(len(data['faces']),3),data['faces']]).ravel())
            self.surface.active_texture_coordinates=data['uv'];self.texture=pv.read_texture(texture/'atlas.png')
        self.dataset_path=self.run/'photoreal/dataset/dataset.json';self.checkpoint=self.run/'photoreal/trained/scene.pt'
        if self.dataset_path.is_file():self.data=json.loads(self.dataset_path.read_text());self.w.gaussian_frame.setRange(0,len(self.data['frames'])-1)
        self.w.gaussian_info.setPlainText(('RGB-only Gaussian checkpoint available; select Gaussian Scene and orbit/pan/zoom.\n' if self.checkpoint.is_file() else 'No trained RGB-only Gaussian scene in this run.\n')+'Geometry completeness and accuracy are not established.')
        if self.surface is not None:self.w.mesh_view_mode.setCurrentText('Textured Mesh');self.mode('Textured Mesh')

    def mode(self,mode):
        if mode!='Gaussian Scene' and self.process is not None:self.stop()
        self.enabled=mode=='Gaussian Scene';self.overlay.setVisible(self.enabled)
        if mode=='Textured Mesh':
            if self.surface is None:self.w.viewport_status.setText('No UV-textured mesh in this run');return True
            self.w._set_actor('Mesh',self.surface.copy(),texture=self.texture,show_scalar_bar=False,lighting=False)
            self.w.viewport_status.setText('Captured RGB texture • unobserved texels gray • geometry accuracy unchanged');self.apply_clip();return True
        if self.enabled:
            if not getattr(self,'checkpoint',Path()).is_file():
                self.enabled=False;self.overlay.hide();self.w.viewport_status.setText('No RGB-only Gaussian checkpoint');return True
            self.w.plotter.camera.parallel_projection=False
            if self.process is None:
                p=QProcess(self);self.process=p;env=QProcessEnvironment()
                for k,v in child_environment(self.w.root).items():env.insert(k,v)
                p.setProcessEnvironment(env);p.setProgram(str(runtime(self.w.root,'gsplat')))
                p.setArguments(['-m','reconx.photoreal_render',str(self.checkpoint),str(self.dataset_path),str(self.run/'photoreal/free_view')])
                p.readyReadStandardOutput.connect(self.read);p.readyReadStandardError.connect(lambda:self.w.log(bytes(p.readAllStandardError()).decode('utf8',errors='replace')))
                p.finished.connect(lambda *_:self.failed('Gaussian renderer stopped'))
                p.errorOccurred.connect(lambda *_:self.failed(p.errorString()));p.start()
            self.timer.start(100);return True
        return False

    def failed(self,message):
        self.ready=False;self.busy=False;self.w.viewport_status.setText(message)
        if self.process and self.process.state()==QProcess.ProcessState.NotRunning:
            self.process.deleteLater();self.process=None

    def request(self):
        if not self.enabled or not self.ready:return
        if self.busy:self.pending=True;return
        cam=self.w.plotter.camera;pose=optical_pose(cam.position,cam.focal_point,cam.up)
        width=min(640,max(64,self.w.plotter.width()));height=max(64,round(width*self.w.plotter.height()/max(1,self.w.plotter.width())))
        height=min(height,1080);f=height/(2*np.tan(np.deg2rad(cam.view_angle)/2))
        request={'T_world_camera':pose.tolist(),'width':width,'height':height,'intrinsics':{'fx':f,'fy':f,'cx':width/2,'cy':height/2}}
        self.busy=True;self.process.write((json.dumps(request)+'\n').encode())

    def read(self):
        self.buffer+=bytes(self.process.readAllStandardOutput()).decode('utf8',errors='replace')
        while '\n' in self.buffer:
            line,self.buffer=self.buffer.split('\n',1)
            try:data=json.loads(line)
            except ValueError:self.w.log(line);continue
            if data.get('ready'):self.ready=True;self.request()
            elif data.get('image'):
                self.busy=False;self.last_result=data
                self.overlay.setGeometry(self.w.plotter.rect());self.overlay.setPixmap(QPixmap(data['image']).scaled(self.overlay.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation));self.overlay.raise_();self.overlay.setVisible(self.enabled)
                self.w.viewport_status.setText(f'RGB-only Gaussian free view • CUDA {data["render_s"]*1000:.1f} ms • partial observed scene')
                if self.pending:self.pending=False;self.timer.start(50)
            elif data.get('error'):self.failed(data['error'])

    def captured(self):
        if not hasattr(self,'data'):return
        frame=self.data['frames'][min(self.w.gaussian_frame.value(),len(self.data['frames'])-1)];pose=np.asarray(frame['T_world_camera']);k=frame['intrinsics']
        self.w.plotter.camera_position=[pose[:3,3],pose[:3,3]+pose[:3,2],-pose[:3,1]]
        self.w.plotter.camera.view_angle=float(np.rad2deg(2*np.arctan(self.data['height']/(2*k['fy']))))
        self.w.plotter.render();self.timer.start(100)

    def apply_clip(self,*_):
        actor=self.w.actors.get('Mesh')
        if actor is None:return
        actor.mapper.RemoveAllClippingPlanes()
        if self.cut.isChecked():
            from vtkmodules.vtkCommonDataModel import vtkPlane
            normal=np.asarray(self.w.plotter.camera.direction);bounds=np.asarray(actor.GetBounds()).reshape(3,2);center=bounds.mean(axis=1)
            origin=center+normal*np.linalg.norm(bounds[:,1]-bounds[:,0])*(self.slider.value()/100-.5)
            plane=vtkPlane();plane.SetNormal(*normal);plane.SetOrigin(*origin);actor.mapper.AddClippingPlane(plane)
        self.w.plotter.render()

    def export(self):
        source=getattr(self,'texture_dir',Path())/'textured.glb'
        if not source.is_file():self.w.viewport_status.setText('No textured GLB available');return
        target,_=QFileDialog.getSaveFileName(self.w,'Export textured GLB',str(self.w.workspace/'textured.glb'),'GLB (*.glb)')
        if target:shutil.copy2(source,target);self.w.viewport_status.setText(str(target))

    def stop(self,*_):
        self.enabled=False;self.overlay.hide();self.timer.stop();self.ready=False;self.busy=False
        if self.process:
            self.process.blockSignals(True);self.process.kill();self.process.waitForFinished(2000);self.process.deleteLater();self.process=None
