"""Studio palette and reversible UI localization; workflow identifiers stay stable."""
import json
import os
from pathlib import Path
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction,QFont
from PySide6.QtWidgets import (QComboBox as QtComboBox,QLabel,QAbstractButton,QTabWidget,
    QDockWidget,QTreeWidget,QLineEdit,QWidget)
from .app_project import atomic_json

KO = {
 'RECON-X 0.12.0\nReconstruction Studio':'RECON-X 0.12.0\n3D 복원 스튜디오',
 'Insufficient memory':'사용 가능한 메모리가 부족합니다',
 'Model asset unavailable or invalid':'모델 파일이 없거나 올바르지 않습니다',
 'Camera reconstruction failed':'카메라 포즈 복원에 실패했습니다',
 'Depth alignment failed':'카메라 복원 결과와 깊이 정합에 실패했습니다',
 'Video could not be read':'영상을 읽을 수 없습니다',
 'Invalid camera calibration':'카메라 보정값이 올바르지 않습니다',
 'Required file unavailable':'필요한 파일을 찾을 수 없습니다',
 'GPU runtime unavailable':'GPU 실행 환경을 사용할 수 없습니다',
 'Invalid project or manifest':'프로젝트 또는 매니페스트가 올바르지 않습니다',
 'Workflow could not complete':'작업을 완료하지 못했습니다','Job failed':'작업 실패',
 'Close other GPU applications; reduce resolution or frame count and retry.':'다른 GPU 프로그램을 닫고 해상도 또는 프레임 수를 줄여 다시 실행하세요.',
 'Select the complete local model directory in Home / Runtime assets.':'홈의 모델 경로 설정에서 필요한 파일이 모두 있는 모델 폴더를 선택하세요.',
 'Use a sharper, textured video with overlap between views; inspect stage logs.':'선명하고 무늬가 풍부하며 인접 시점이 겹치는 영상을 사용하세요. 단계별 로그도 확인하세요.',
 'Use a valid MP4/MOV with a supported codec; check input metadata.':'지원되는 코덱의 MP4/MOV 영상을 사용하고 영상 정보를 확인하세요.',
 'Inspect the per-frame depth report and SfM camera registration. Capture a slower video with textured surfaces and camera translation; check calibration if available.':'프레임별 깊이 보고서와 카메라 등록 결과를 확인하세요. 질감이 있는 물체를 천천히 이동하며 촬영하고, 보정값이 있다면 확인하세요.',
 'Inspect SfM registration and camera intrinsics. Capture overlapping textured views with camera translation.':'카메라 등록 결과와 내부 파라미터를 확인하세요. 겹치는 장면을 유지하며 카메라를 이동해 촬영하세요.',
 'Check MP4/MOV codec, orientation and first-frame decode; use a compatible conversion if needed.':'MP4/MOV 코덱, 회전 정보 및 첫 프레임 디코딩을 확인하고 필요하면 호환 형식으로 변환하세요.',
 'Check positive focal lengths and the calibration image resolution.':'초점 거리가 양수인지, 보정 영상의 해상도가 일치하는지 확인하세요.',
 'Restore or select the file again; check project references.':'파일을 복구하거나 다시 선택하고 프로젝트의 참조 경로를 확인하세요.',
 'Check the NVIDIA driver and the selected runtime pack.':'NVIDIA 드라이버와 선택한 실행 환경을 확인하세요.',
 'Open a supported RECON-X JSON file or restore its backup.':'지원되는 RECON-X JSON 파일을 열거나 백업을 복원하세요.',
 'Review the diagnostic log, correct the input or runtime configuration, and retry.':'진단 로그에서 원인을 확인하고 입력 또는 실행 환경을 수정한 뒤 다시 실행하세요.',
 'Project':'프로젝트','Artifacts':'결과물','Application':'애플리케이션','Settings':'설정',
 'Appearance & language…':'화면 및 언어…','Appearance & language':'화면 및 언어',
 'Language':'언어','Close':'닫기','Home':'홈','Inspector':'정보','Depth':'깊이',
 'Semantics':'의미 지도','Evaluation':'평가','Inverse':'역렌더링','RGB Video':'RGB 영상',
 'New project':'새 프로젝트','Open project':'프로젝트 열기','Save project':'프로젝트 저장',
 'Open project…':'프로젝트 열기…','Save project…':'프로젝트 저장…','Open recent project':'최근 프로젝트 열기',
 'Open dataset':'데이터셋 열기','Open session':'이전 세션 열기','Save session':'세션 저장','Exit':'종료',
 'Import RGB video results':'RGB 영상 복원 결과 열기','Export diagnostics':'진단 정보 내보내기',
 'Open TSDF mesh':'TSDF 메시 열기','Open point cloud':'포인트 클라우드 열기',
 'Open trajectory manifest':'궤적 매니페스트 열기','Open semantic map':'의미 지도 열기',
 'Open Gaussian checkpoint and dataset':'가우시안 체크포인트·데이터셋 열기',
 'DATASET  /  PIPELINES':'데이터셋 / 복원 작업','Asset':'항목','Value':'값',
 'Open TUM manifest / directory':'TUM 데이터셋 / 폴더 열기','Frame index':'프레임 번호',
 'Range':'범위','to':'~','RGB frame':'RGB 프레임','Sensor depth':'센서 깊이','Pipeline':'복원 작업',
 'Voxel m':'복셀 (m)','Trunc m':'절단 (m)','Output directory…':'출력 폴더…',
 '▶ Run':'▶ 실행','■ Cancel':'■ 취소','Fit scene':'장면 맞춤','Reset camera':'시점 초기화',
 'Load mesh':'메시 불러오기','Load cloud':'점군 불러오기','GT path':'GT 궤적','SLAM path':'SLAM 궤적',
 'SfM path':'SfM 궤적','Frustums':'카메라 시야','Mesh':'메시','Cloud':'점군','Axes':'좌표축',
 'Inverse material':'복원 재질','Point size':'점 크기','Default color':'기본 색상',
 'Mesh display':'메시 표시','Solid':'단색','Wireframe':'와이어프레임',
 'RGB Vertex Color':'실제 RGB 정점 색상','Semantic Color':'의미 레이블 색상',
 'Height (meters)':'높이 (미터)','Height (arbitrary SfM units)':'높이 (임의 SfM 단위)',
 'Console / Progress / Performance':'실행 로그 / 진행 상태 / 성능','Idle':'대기 중',
 'Depth Anything V2 Small (relative)':'Depth Anything V2 Small (상대 깊이)',
 'Marigold v1.1 (affine-invariant)':'Marigold v1.1 (아핀 불변 깊이)',
 'Raw model output':'모델 원본 출력','Robust aligned meters (oracle GT, Phase 4)':'강건 정합 · 미터 (GT 사용, Phase 4)',
 'Sparse aligned meters (RTAB RGB-D, Phase 6)':'희소 기하 정합 · 미터 (RTAB RGB-D, Phase 6)',
 'GT sensor depth (meters)':'GT 센서 깊이 (미터)',
 'Robust held-out absolute error (meters)':'강건 정합 · 평가 픽셀 절대 오차 (미터)',
 'No depth output':'깊이 결과 없음','Render selected pose on CUDA':'선택한 시점을 CUDA로 렌더링',
 'Render a saved scene at a selected camera pose':'저장된 장면을 선택한 카메라 시점에서 렌더링합니다',
 'Camera frame':'카메라 프레임','Semantic class filter':'의미 클래스 필터',
 'Query 3D world position':'3D 좌표 의미 조회','No query':'조회 결과 없음','All classes':'모든 클래스',
 'Metric / condition':'지표 / 조건','Refresh actual experiment files':'실행 결과 새로고침',
 'GT pose + GT TSDF':'GT 포즈 + GT TSDF','RTAB pose + RTAB TSDF':'RTAB 포즈 + RTAB TSDF',
 'Fixed ambient + directional light':'고정 환경광 + 방향광','Pose / geometry':'포즈 / 기하',
 'TSDF mesh':'TSDF 메시','Illumination model':'조명 모델','Regularization':'정규화',
 'Method':'방법','View':'보기','Frame':'프레임','Optimize in background':'백그라운드 최적화',
 'Load results':'결과 불러오기','Show 3D material':'3D 재질 표시','3D light XYZ':'3D 조명 XYZ',
 'Run or load Phase 10 results':'역렌더링을 실행하거나 결과를 불러오세요',
 'original':'원본','normals':'법선','albedo':'알베도','shading':'음영','reconstructed':'복원 영상','error':'오차','mask':'마스크',
 'lambertian_regularized':'Lambertian · 정규화 적용','lambertian_unregularized':'Lambertian · 정규화 없음',
 'observed_appearance':'관측 외관','constant_albedo':'상수 알베도',
 'Phase 11 · RGB video only · SfM scale is arbitrary':'RGB 전용 영상 복원 · SfM 스케일은 임의 단위입니다',
 'Select MP4 / MOV':'MP4 / MOV 선택','Inspect video metadata':'영상 정보 확인',
 'Optional calibrated intrinsics JSON path':'카메라 내부 파라미터 JSON 경로 (선택)',
 'Frame stride':'프레임 추출 간격','Maximum frames':'최대 프레임 수','SfM neighbor overlap':'SfM 인접 프레임 수',
 'Relative depth model':'상대 깊이 모델','TSDF voxel length (SfM units)':'TSDF 복셀 크기 (SfM 단위)',
 'TSDF depth truncation (SfM units)':'TSDF 깊이 절단 거리 (SfM 단위)',
 'Run RGB video pipeline':'RGB 영상 복원 실행','Cancel video pipeline':'영상 복원 취소',
 'Load Phase 11 results':'RGB 영상 결과 불러오기','Query video semantic map':'영상 의미 지도 조회',
 '3D query XYZ in SfM units, e.g. 0, 0, 10':'조회할 XYZ 좌표 (SfM 단위). 예: 0, 0, 10',
 'Configure existing model / RTAB-Map…':'모델 / RTAB-Map 경로 설정…',
 'RGB-D-assisted initialization; fixed 160 px images':'RGB-D 깊이를 사용한 초기화 · 영상 폭 160 px',
 'Training iterations':'학습 반복 횟수','Maximum Gaussians':'최대 가우시안 수','Train Gaussian scene':'가우시안 장면 학습',
 'Gaussian train':'가우시안 학습','Gaussian render':'가우시안 렌더링','Inverse rendering':'역렌더링',
 'Inputs: TUM RGB-D / MP4, MOV / RECON-X JSON / PLY\nRGB-only reconstruction has arbitrary SfM scale.':
 '입력: TUM RGB-D / MP4, MOV / RECON-X JSON / PLY\nRGB 전용 복원은 임의의 SfM 스케일을 사용합니다.',
 'Orbit: left drag  •  Pan: middle drag  •  Zoom: wheel  •  Pick: right click':
 '회전: 왼쪽 드래그  ·  이동: 가운데 드래그  ·  확대: 휠  ·  선택: 오른쪽 클릭',
 'RGB-only scene • arbitrary SfM scale • orbit / pan / zoom / right-click query':
 'RGB 전용 장면 · 임의 SfM 스케일 · 회전 / 이동 / 확대 / 오른쪽 클릭 조회',
 'Interpretation':'지표 해석','Accuracy':'정확도','Counts measure density; consistency is not GT accuracy.':'개수는 밀도 지표입니다. 일관성은 GT 정확도가 아닙니다.',
 'Unavailable unless this run has reference evaluation.':'현재 실행에 기준 데이터 평가가 없으면 정확도를 제공하지 않습니다.',
 'queued':'대기','running':'실행 중','completed':'완료','failed':'실패','cancelled':'취소됨',
 'Ready':'사용 가능','Missing':'없음','Configured':'설정됨','Select model directory':'모델 폴더 선택 필요',
 'RUNTIMES':'실행 환경','MODEL ASSETS':'모델 파일','Weights are not downloaded automatically.':'모델 파일은 자동으로 다운로드하지 않습니다.',
 'Logs and projects: ':'로그 및 프로젝트: ',
 'Select a video. SfM, relative depth, TSDF and semantics run from RGB only. Camera intrinsics are estimated unless separately calibrated.':
 '영상을 선택하세요. RGB 영상만으로 SfM·상대 깊이·TSDF·의미 지도를 생성합니다. 별도의 보정값이 없으면 카메라 내부 파라미터를 추정합니다.',
}

def tr(text,language='en'):
    return KO.get(text,text) if language=='ko' else text

class QComboBox(QtComboBox):
    """Display translated choices without changing algorithm keys/currentText contracts."""
    SOURCE_ROLE=int(Qt.ItemDataRole.UserRole)+71
    def currentText(self):
        return self.currentData(self.SOURCE_ROLE) or super().currentText()
    def setItemText(self,index,text):
        self.setItemData(index,text,self.SOURCE_ROLE)
        super().setItemText(index,text)
    def findText(self,text,*args):
        for i in range(self.count()):
            if self.itemData(i,self.SOURCE_ROLE)==text:return i
        return super().findText(text,*args)
    def setCurrentText(self,text):
        index=self.findText(text)
        if index>=0:self.setCurrentIndex(index)
        else:super().setCurrentText(text)

STYLE='''
QWidget { background:#171b24; color:#e1e7ed; font-family:"Segoe UI","Malgun Gothic"; font-size:13px; }
QMainWindow { background:#141821; }
QMenuBar { background:#13191d; border-bottom:1px solid #303444; padding:4px; }
QMenuBar::item { padding:6px 12px; background:transparent; }
QMenuBar::item:selected,QMenu::item:selected { background:#34334d; color:#c2f78d; }
QMenu { background:#20232d; border:1px solid #414355; padding:5px; }
QMenu::item { padding:7px 24px; }
QLabel { background:transparent; }
QTreeWidget,QTextEdit,QLineEdit,QComboBox,QSpinBox,QDoubleSpinBox { background:#131a21; border:1px solid #343948; border-radius:4px; padding:5px; selection-background-color:#42445c; selection-color:#f4f7fa; }
QLineEdit:focus,QComboBox:focus,QSpinBox:focus,QDoubleSpinBox:focus { border:1px solid #a5dc7d; }
QTreeWidget::item { padding:4px; border-bottom:1px solid #232b35; }
QTreeWidget::item:selected { background:#31394a; color:#c2f78d; }
QHeaderView::section { background:#202733; border:0; border-bottom:1px solid #3b4052; padding:6px; color:#aab6c5; }
QPushButton { background:#252d36; border:1px solid #3e4853; padding:8px 10px; border-radius:5px; color:#dce5ed; }
QPushButton:hover { background:#34374b; border-color:#8e81b8; }
QPushButton:pressed { background:#41425a; }
QPushButton:disabled { color:#697580; background:#1c232b; border-color:#29323d; }
QPushButton[primary="true"] { background:#b1ed82; border:1px solid #b1ed82; color:#14221a; font-weight:600; }
QPushButton[primary="true"]:hover { background:#c5fa9d; }
QPushButton[primary="true"]:disabled { background:#374538; color:#84917b; border-color:#3e5142; }
QTabWidget::pane { border:1px solid #34394a; background:#191e29; border-radius:5px; }
QTabBar::tab { background:#191f28; color:#aab6c5; padding:10px 12px; border-bottom:2px solid transparent; }
QTabBar::tab:selected { background:#2b2d41; color:#c2f78d; border-bottom:2px solid #b1ed82; }
QTabBar::tab:hover { color:#f2f4f8; background:#282c3c; }
QDockWidget::title { background:#202431; padding:8px; color:#b6bad2; }
QProgressBar { background:#11191f; border:1px solid #353e49; border-radius:4px; text-align:center; min-height:16px; color:#e5f0e8; }
QProgressBar::chunk { background:#547b53; border-radius:3px; }
QCheckBox { spacing:5px; background:transparent; }
QCheckBox::indicator { width:12px; height:12px; border:1px solid #617064; border-radius:3px; background:#17221d; }
QCheckBox::indicator:checked { background:#b1ed82; border-color:#b1ed82; }
QScrollBar:vertical { background:#171d25; width:9px; }
QScrollBar:horizontal { background:#171d25; height:9px; }
QScrollBar::handle { background:#4a4b62; border-radius:4px; min-height:24px; min-width:24px; }
QScrollBar::add-line,QScrollBar::sub-line { width:0; height:0; }
QSplitter::handle { background:#252b39; }
QToolTip { background:#303344; color:#edf3f5; border:1px solid #74708c; padding:5px; }
'''

def settings_path():
    return Path(os.environ.get('LOCALAPPDATA',Path.home()))/'RECON-X/preferences.json'

def read_language(path=None):
    try:
        value=json.loads(Path(path or settings_path()).read_text(encoding='utf-8')).get('language','en')
        return value if value in ('en','ko') else 'en'
    except (OSError,ValueError,AttributeError):return 'en'

def save_language(language,path=None):
    if language not in ('en','ko'):raise ValueError('Unsupported language')
    atomic_json(Path(path or settings_path()),{'schema_version':1,'language':language})

def apply_theme(window):
    for button in (window.run_button,window.video_run):button.setProperty('primary',True)
    window.setStyleSheet(STYLE)
    window.setFont(QFont('Malgun Gothic',9))
    for label in window.findChildren(QLabel):
        if 'font-size:22px' in label.styleSheet():
            label.setStyleSheet('font-size:22px;font-weight:600;color:#c2f78d;padding:8px 0;')
        elif 'font-size:15px' in label.styleSheet():
            label.setStyleSheet('font-size:14px;font-weight:600;color:#b8b4db;padding:4px 0;')
    window.plotter.set_background('#141d24',top='#28233c')

def localize(window,language):
    def update(obj,key,current,setter):
        last=obj.property('_i18n_last_'+key);source=obj.property('_i18n_source_'+key)
        if source is None or (last is not None and current!=last):source=current
        obj.setProperty('_i18n_source_'+key,source)
        target=tr(source,language);setter(target);obj.setProperty('_i18n_last_'+key,target)
    for obj in window.findChildren(QWidget):
        if isinstance(obj,(QLabel,QAbstractButton)):
            update(obj,'text',obj.text(),obj.setText)
        if isinstance(obj,QDockWidget):update(obj,'title',obj.windowTitle(),obj.setWindowTitle)
        if isinstance(obj,QLineEdit):update(obj,'placeholder',obj.placeholderText(),obj.setPlaceholderText)
        if isinstance(obj,QTabWidget):
            for i in range(obj.count()):update(obj.widget(i),'tab',obj.tabText(i),lambda text,j=i:obj.setTabText(j,text))
        if isinstance(obj,QComboBox):
            blocked=obj.blockSignals(True)
            for i in range(obj.count()):
                source=obj.itemData(i,obj.SOURCE_ROLE) or obj.itemText(i)
                obj.setItemData(i,source,obj.SOURCE_ROLE);QtComboBox.setItemText(obj,i,tr(source,language))
            obj.blockSignals(blocked)
        if isinstance(obj,QTreeWidget):
            header=obj.headerItem()
            for i in range(obj.columnCount()):
                source=header.data(i,int(Qt.ItemDataRole.UserRole)+72) or header.text(i)
                header.setData(i,int(Qt.ItemDataRole.UserRole)+72,source);header.setText(i,tr(source,language))
    for action in window.findChildren(QAction):update(action,'text',action.text(),action.setText)
    window.setWindowTitle('Rebuilder 3D 0.12.1 | '+('3D 복원 스튜디오' if language=='ko' else 'Reconstruction Studio'))
