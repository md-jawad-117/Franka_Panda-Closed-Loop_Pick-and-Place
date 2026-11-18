"""Sensor-only learned adapter and persistent cross-Python CPU worker client."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,TimeoutError
import io,json,os,struct,subprocess,time
import numpy as np

from .config import resolve
from .dataset_geometry import camera_transform,intrinsics
from .observation_selection import select_observation


class WorkerClient:
    def __init__(self,config,on_wait=None):
        self.timeout=config['worker_timeout_s'];self.pool=ThreadPoolExecutor(max_workers=1)
        self.on_wait=on_wait
        command=[str(resolve(config['worker_python'])),'-u',str(resolve(config.get('worker_script','training/learned_pose_worker.py'))),
                 '--candidate',str(resolve(config['candidate'])), '--protocol',str(resolve(config['protocol'])),
                 '--candidate-sha',config['candidate_sha256'],'--protocol-sha',config['protocol_sha256'],
                 '--threads',str(config['worker_threads'])]
        if config.get('fixture_protocol'):
            command+=['--fixture-protocol',str(resolve(config['fixture_protocol'])),
                      '--fixture-protocol-sha',config['fixture_protocol_sha256']]
        if config.get('capsule_protocol'):
            command+=['--capsule-protocol',str(resolve(config['capsule_protocol'])),
                      '--capsule-protocol-sha',config['capsule_protocol_sha256']]
        environment=os.environ.copy();environment.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS=str(config['worker_threads']),MKL_NUM_THREADS=str(config['worker_threads']))
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=None,env=environment,
                                      creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        try:
            self.metadata=self._await(self._read)
            if not self.metadata.get('ready'):raise RuntimeError('Learned worker did not initialize')
        except Exception:self.close();raise

    def _read(self):
        line=self.process.stdout.readline()
        if not line:raise RuntimeError('Learned inference worker exited; no oracle fallback is permitted')
        return json.loads(line)

    def _await(self,fn):
        future=self.pool.submit(fn);deadline=time.monotonic()+self.timeout
        while True:
            remaining=deadline-time.monotonic()
            if remaining<=0:
                self.process.kill();self.process.wait();raise RuntimeError('Learned inference worker timed out')
            try:return future.result(timeout=min(.025,remaining) if self.on_wait is not None else remaining)
            except TimeoutError:
                if self.on_wait is not None:self.on_wait()

    def predict(self,arrays):
        if self.process.poll() is not None:raise RuntimeError('Learned inference worker exited; simulation stopped without oracle fallback')
        stream=io.BytesIO();np.savez(stream,**arrays);payload=stream.getvalue()
        def request():
            self.process.stdin.write(struct.pack('!I',len(payload))+payload);self.process.stdin.flush()
            return self._read()
        return self._await(request)

    def close(self):
        if hasattr(self,'process'):
            if self.process.poll() is None:
                try:
                    self.process.stdin.write(struct.pack('!I',0));self.process.stdin.flush();self.process.wait(timeout=3)
                except (OSError,subprocess.TimeoutExpired):self.process.kill();self.process.wait()
            self.process.stdin.close();self.process.stdout.close()
        self.pool.shutdown(wait=True,cancel_futures=True)


class LearnedPerception:
    """No Sim reference: sensor frames -> fixed geometry -> temporal confirmation.

    Inference is synchronous lockstep simulation, not a real-time latency model.
    Rejected observations never become zero/default target measurements.
    """
    def __init__(self,sensor,config,rng,worker=None,on_wait=None):
        self.sensor=sensor;self.config=config;self.rng=rng
        self.worker=worker or WorkerClient(config,on_wait);self.metadata=self.worker.metadata
        self.reset()

    def reset(self):
        self.next_capture=-float('inf');self.pending={};self.tracks={};self.latest={};self.sample_t=None
        self.sample_index=0;self.counts=Counter();self.history=[];self.wall_seconds=[];self.last_masks={}

    def update(self,t):
        if t+1e-9<self.next_capture:return None
        arrays={};start=time.perf_counter()
        for camera in ('wrist_cam','top_cam'):
            frame=self.sensor.capture(camera)
            if abs(frame.timestamp-t)>1e-7:raise RuntimeError('Camera timestamp differs from controller simulation time')
            noise=self.config['depth_noise'];depth=frame.depth
            valid=np.isfinite(depth)&(depth>0)&(depth<noise['max_range_m'])
            valid &= self.rng.random(depth.shape)>=noise['dropout']
            observed=depth+self.rng.normal(0,noise['sigma_m'],depth.shape);valid &= observed>0
            observed=np.where(valid,observed,0).astype(np.float32)
            arrays.update({f'rgb_{camera}':frame.rgb,f'depth_{camera}':observed,f'valid_{camera}':valid.astype(np.uint8),
                           f'K_{camera}':intrinsics(depth.shape[1],depth.shape[0],frame.fovy_deg),
                           f'T_{camera}':camera_transform(frame.world_rotation,frame.world_position)})
        response=self.worker.predict(arrays);measurements={};status={}
        evidence={target:{'capture_t':float(t),'motion_pending':self.latest.get(target,{}).get('motion_pending',False)}
                  for target in ('capsule','fixture')}
        for target in ('capsule','fixture'):
            selected,routing=select_observation(response['views'],target,self.config.get('observation_selection'),self.tracks.get('fixture'))
            if self.config.get('observation_selection'):evidence[target]['routing']=routing
            if self.config.get('reacquisition_evidence'):
                evidence[target]['wrist_world_from_camera']=arrays['T_wrist_cam'].tolist()
            if selected is None:
                self.pending.pop(target,None);status[target]='geometry_rejected'
                self.counts[target+'_geometry_rejected']+=1;continue
            position=np.array(selected['position_m']);axis=np.array(selected['axis_world']);previous=self.tracks.get(target)
            wrist=arrays['T_wrist_cam'];local_position=(position-wrist[:3,3])@wrist[:3,:3];local_axis=axis@wrist[:3,:3]
            # Existing stable tracks accept small innovations. A changed/new target
            # requires two consistent, distinct captures before it is trusted.
            def consistent(entry):
                angle=np.cos(np.radians(self.config['confirmation_axis_deg']));distance=self.config['confirmation_position_m']
                world_ok=np.linalg.norm(position-entry['position'])<=distance and abs(float(axis@entry['axis']))>=angle
                wrist_ok=(target=='capsule' and self.config['allow_wrist_frame_confirmation'] and
                          np.linalg.norm(local_position-entry['local_position'])<=distance and abs(float(local_axis@entry['local_axis']))>=angle)
                return world_ok or wrist_ok
            current={'position':position,'axis':axis,'local_position':local_position,'local_axis':local_axis}
            if previous is not None and consistent(previous):confirmed=True
            else:
                pending=self.pending.get(target);n=pending['n']+1 if pending is not None and consistent(pending) else 1
                self.pending[target]={**current,'n':n}
                confirmed=n>=self.config['confirmation_samples']
            if not confirmed:
                # A plausible innovation blocks commitment, but remains untrusted
                # as a robot target. Missing geometry does not erase this warning.
                evidence[target]['motion_pending'] = previous is not None
                evidence[target]['candidate_position_m'] = position.tolist()
                evidence[target]['candidate_axis_world'] = axis.tolist()
                status[target]='awaiting_confirmation';self.counts[target+'_awaiting_confirmation']+=1;continue
            evidence[target]['motion_pending']=False
            self.pending.pop(target,None);self.tracks[target]=current
            measurements[target]=(position,float(np.arctan2(axis[1],axis[0])),selected['source'],axis,selected['quality'])
            status[target]='accepted';self.counts[target+'_accepted']+=1
        for target in evidence:evidence[target]['status']=status[target]
        self.latest=evidence
        self.next_capture=t+1/self.config['capture_hz'];self.sample_t=t;self.sample_index+=1
        wall=time.perf_counter()-start;self.wall_seconds.append(wall)
        self.history.append({'t':float(t),'sample_index':self.sample_index,'status':status,'evidence':evidence,
                             'sources':{target:value[2] for target,value in measurements.items()},
                             'measurements':{target:{'position_m':value[0].tolist(),'yaw_rad':value[1],'source':value[2]}
                                             for target,value in measurements.items()},
                             'worker_seconds':response['wall_seconds'],'total_capture_inference_seconds':wall,
                             'rejections':{camera:{target:r['reasons'] for target,r in views.items()}
                                           for camera,views in response['views'].items()}})
        return measurements

    def summary(self):
        return {'metadata':self.metadata,'sample_count':self.sample_index,'counts':dict(self.counts),
                'mean_wall_seconds_per_capture':float(np.mean(self.wall_seconds)) if self.wall_seconds else None,
                'timing_model':'synchronous CPU inference in lockstep simulation; wall time is not simulated sensor delay',
                'depth_noise':self.config['depth_noise'],'capture_hz_simulation':self.config['capture_hz']}

    def close(self):
        self.sensor.close();self.worker.close()
