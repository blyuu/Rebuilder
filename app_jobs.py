"""One serial QProcess queue for CPU/GPU workflows, with process-tree cleanup."""
from __future__ import annotations
import json
import time
import uuid
import threading
import subprocess
from pathlib import Path
import psutil
from PySide6.QtCore import QObject, QProcess, QProcessEnvironment, QTimer, Signal
from .app_project import atomic_json, explain_error
from .app_runtime import VERSION, child_environment
from .app_job_state import TRANSITIONS,transition


class JobManager(QObject):
    changed=Signal(dict)
    message=Signal(str)
    finished=Signal(dict)

    def __init__(self,root:Path,logs:Path,parent=None):
        super().__init__(parent);self.root=Path(root);self.logs=Path(logs)
        self.queue=[];self.active=None;self.process=None;self.records=[];self._children={}
        self.timer=QTimer(self);self.timer.setInterval(100);self.timer.timeout.connect(self._sample)

    def submit(self,name,stages,output,metadata=None):
        record={'id':uuid.uuid4().hex,'version':VERSION,'name':name,'status':'queued',
                'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'output':str(output),
                'metadata':metadata or {},'stages':[{'name':n,'program':str(p),'arguments':list(a)} for n,p,a in stages],
                'stage_index':0,'progress':0,'peak_tree_rss_bytes':0,'sampled_cpu_seconds':0,
                'log_dir':str(self.logs/uuid.uuid4().hex),'stage_results':[]}
        Path(record['log_dir']).mkdir(parents=True,exist_ok=True)
        self.records.append(record);self.queue.append(record);self._persist(record)
        QTimer.singleShot(0,self._next);return record

    def _persist(self,record):
        atomic_json(Path(record['log_dir'])/'job.json',record);self.changed.emit(dict(record))

    def _next(self):
        if self.active or not self.queue:return
        self.active=self.queue.pop(0);transition(self.active,'running')
        self.started=time.perf_counter()
        self.gpu_stop=threading.Event();self.gpu_samples=[]
        required=self.active['metadata'].get('required_free_ram_bytes',128*1024*1024)
        if psutil.virtual_memory().available < required:
            self._finish('failed',f'Insufficient resources: requires {required} bytes of free RAM');return
        self.gpu_thread=threading.Thread(target=self._gpu_sample,args=(self.gpu_stop,self.gpu_samples,self.started),daemon=True);self.gpu_thread.start()
        self._start_stage()

    def _gpu_sample(self,stop,samples,started):
        while not stop.is_set():
            try:
                r=subprocess.run(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],
                    capture_output=True,text=True,timeout=3,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                if r.returncode==0:
                    values=r.stdout.splitlines()[0].split(',')
                    samples.append({'elapsed_s':time.perf_counter()-started,'device_used_MiB':float(values[0]),'device_utilization_percent':float(values[1])})
            except (OSError,ValueError,IndexError,subprocess.TimeoutExpired):pass
            stop.wait(.5)

    def _start_stage(self):
        r=self.active;stage=r['stages'][r['stage_index']]
        p=QProcess(self);self.process=p;self.stage_started=time.perf_counter();self._tail='';self._children={};self.stage_cpu=0
        p.setProgram(stage['program']);p.setArguments(stage['arguments']);p.setWorkingDirectory(str(self.root))
        env=QProcessEnvironment()
        for key,value in child_environment(self.root).items():env.insert(key,value)
        p.setProcessEnvironment(env)
        p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        p.readyReadStandardOutput.connect(self._read)
        p.finished.connect(self._stage_finished)
        p.errorOccurred.connect(self._error)
        self.message.emit(f'{r["name"]}: stage {r["stage_index"]+1}/{len(r["stages"])} {stage["name"]}')
        self._persist(r);p.start();self.timer.start()

    def _read(self):
        if not self.process or not self.active:return
        text=bytes(self.process.readAllStandardOutput()).decode('utf-8',errors='replace')
        self._tail=(self._tail+text)[-12000:]
        with (Path(self.active['log_dir'])/f'{self.active["stage_index"]:02d}.log').open('a',encoding='utf-8') as f:f.write(text)
        for line in text.splitlines():self.message.emit(line)

    def _sample(self):
        if not self.process or not self.active:return
        try:
            proc=psutil.Process(int(self.process.processId()));procs=[proc]+proc.children(recursive=True)
            rss=cpu=0
            for child in procs:
                self._children[child.pid]=child
                try:rss+=child.memory_info().rss;ct=child.cpu_times();cpu+=ct.user+ct.system
                except psutil.Error:pass
            self.active['peak_tree_rss_bytes']=max(self.active['peak_tree_rss_bytes'],rss)
            self.active['sampled_cpu_seconds']=max(self.active['sampled_cpu_seconds'],cpu)
            self.stage_cpu=max(self.stage_cpu,cpu)
        except (psutil.Error,ValueError):pass

    def _error(self,error):
        if error==QProcess.ProcessError.FailedToStart and self.active:
            self._finish('failed',self.process.errorString())

    def _stage_finished(self,code,status):
        if not self.active:return
        self._read();r=self.active
        r['stage_results'].append({'name':r['stages'][r['stage_index']]['name'],'exit_code':code,
                                   'wall_s':time.perf_counter()-self.stage_started,'sampled_cpu_seconds':self.stage_cpu})
        if r.get('cancel_requested'):self._finish('cancelled');return
        if code!=0 or status!=QProcess.ExitStatus.NormalExit:
            self._finish('failed',self._tail or f'Process exit code {code}');return
        r['stage_index']+=1;r['progress']=r['stage_index']/len(r['stages'])
        if r['stage_index']==len(r['stages']):self._finish('completed')
        else:
            self.process.deleteLater();self.process=None;self._start_stage()

    def _finish(self,state,error=None):
        self.timer.stop();r=self.active
        if not r:return
        transition(r,state);r['wall_s']=time.perf_counter()-self.started
        r['sampled_cpu_seconds']=sum(s['sampled_cpu_seconds'] for s in r['stage_results'])
        r['mean_cpu_core_equivalents']=r['sampled_cpu_seconds']/max(r['wall_s'],1e-9)
        if hasattr(self,'gpu_stop'):
            self.gpu_stop.set()
            r['gpu_samples']=list(self.gpu_samples)
            r['peak_device_vram_MiB']=max((x['device_used_MiB'] for x in self.gpu_samples),default=None)
            r['gpu_measurement_scope']='whole device, includes desktop and other processes; not Torch allocated peak'
        if error:r['error']=explain_error(error,r['stages'][min(r['stage_index'],len(r['stages'])-1)]['name'])
        self._persist(r)
        if self.process:self.process.deleteLater()
        self.process=None;self.active=None
        self.finished.emit(dict(r));QTimer.singleShot(0,self._next)

    def cancel(self,job_id=None):
        for r in list(self.queue):
            if job_id is None or r['id']==job_id:
                self.queue.remove(r);transition(r,'cancelled');self._persist(r);self.finished.emit(dict(r))
        if self.active and (job_id is None or self.active['id']==job_id):
            self.active['cancel_requested']=True;self._sample()
            # Kill descendants before the parent; Windows terminate is forceful and releases CUDA contexts.
            for child in reversed(list(self._children.values())):
                try:child.kill()
                except psutil.Error:pass
            if self.process:self.process.kill()
            self._persist(self.active)

    def shutdown(self):
        self.cancel()
        if self.process:self.process.waitForFinished(2000)
