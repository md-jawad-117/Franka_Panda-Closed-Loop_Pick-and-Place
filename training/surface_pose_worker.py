"""Optional surface-aware fixture worker; same frozen network/sensor boundary."""
import argparse,io,json,struct,sys,time
from pathlib import Path
import numpy as np
from learned_pose_worker import InferenceService,read_exact,sha,CAMERAS
from pose_evaluation.geometry import Calibration
from pose_evaluation.fixture_surface_v2 import estimate_fixture,SETTINGS


class SurfaceInferenceService(InferenceService):
    def __init__(self,*args,fixture_protocol,fixture_protocol_sha,**kwargs):
        super().__init__(*args,**kwargs)
        if sha(fixture_protocol)!=fixture_protocol_sha:raise ValueError('Surface protocol hash mismatch')
        self.surface=json.loads(fixture_protocol.read_text())
        if self.surface['base_protocol_sha256']!=self.metadata['protocol_sha256']:raise ValueError('Base protocol mismatch')
        implementation=Path(__file__).parent/'pose_evaluation/fixture_surface_v2.py'
        if self.surface['implementation_sha256']!=sha(implementation):raise ValueError('Surface implementation changed')
        if self.surface['settings']!=SETTINGS:raise ValueError('Surface settings differ from frozen implementation')
        self.masks={};self.hooks=[]
        for camera,model in self.models.items():
            def capture(module,inputs,output,camera=camera):self.masks[camera]=output.argmax(1)[0].numpy().astype(np.uint8)
            self.hooks.append(model.register_forward_hook(capture))
        self.metadata.update(base_worker_sha256=self.metadata['worker_sha256'],worker_sha256=sha(__file__),
                             fixture_estimator=SETTINGS['version'],fixture_estimator_sha256=sha(implementation),
                             fixture_protocol_sha256=fixture_protocol_sha,fixture_settings=SETTINGS)

    def predict(self,arrays):
        started=time.perf_counter();response=super().predict(arrays)
        original={}
        for camera in CAMERAS:
            original[camera]=response['views'][camera]['fixture']
            calibration=Calibration(arrays['K_'+camera],arrays['T_'+camera])
            response['views'][camera]['fixture']=estimate_fixture(self.masks[camera]==2,arrays['depth_'+camera],
                arrays['valid_'+camera],calibration,self.protocol,self.surface['settings'])
        response['diagnostic_v1_fixture']=original
        response['wall_seconds']=time.perf_counter()-started
        return response


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','protocol','fixture-protocol'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('candidate-sha','protocol-sha','fixture-protocol-sha'):p.add_argument('--'+name,required=True)
    p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    service=SurfaceInferenceService(a.candidate,a.protocol,a.candidate_sha,a.protocol_sha,threads=a.threads,
                                    fixture_protocol=a.fixture_protocol,fixture_protocol_sha=a.fixture_protocol_sha)
    print(json.dumps(service.metadata,allow_nan=False),flush=True)
    while True:
        header=sys.stdin.buffer.read(4)
        if not header:return
        if len(header)!=4:raise EOFError('Truncated header')
        size=struct.unpack('!I',header)[0]
        if size>8*1024*1024:raise ValueError('Observation packet too large')
        if not size:return
        with np.load(io.BytesIO(read_exact(sys.stdin.buffer,size)),allow_pickle=False) as archive:
            arrays={key:archive[key] for key in archive.files}
        print(json.dumps(service.predict(arrays),allow_nan=False),flush=True)


if __name__=='__main__':main()
