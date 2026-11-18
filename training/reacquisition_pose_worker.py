"""Frozen-network worker with an optional observed-floor capsule surface filter."""
from pathlib import Path
import argparse,io,json,struct,sys,time
import numpy as np
from surface_pose_worker import SurfaceInferenceService
from learned_pose_worker import sha,read_exact,CAMERAS
from pose_evaluation.geometry import Calibration
from pose_evaluation.capsule_surface_v4 import estimate_capsule_with_fallback,SETTINGS


class ReacquisitionInferenceService(SurfaceInferenceService):
    def __init__(self,*args,capsule_protocol,capsule_protocol_sha,**kwargs):
        super().__init__(*args,**kwargs)
        if sha(capsule_protocol)!=capsule_protocol_sha:raise ValueError('Capsule protocol hash mismatch')
        protocol=json.loads(capsule_protocol.read_text());implementation=Path(__file__).parent/'pose_evaluation/capsule_surface_v4.py'
        if protocol['implementation_sha256']!=sha(implementation):raise ValueError('Capsule estimator changed')
        if protocol['base_protocol_sha256']!=self.metadata['protocol_sha256']:raise ValueError('Base capsule protocol changed')
        if protocol['fixture_protocol_sha256']!=self.metadata['fixture_protocol_sha256']:raise ValueError('Fixture protocol changed')
        if protocol['settings']!=SETTINGS:raise ValueError('Capsule settings changed')
        self.metadata.update(surface_worker_sha256=self.metadata['worker_sha256'],worker_sha256=sha(__file__),
                             capsule_estimator=SETTINGS['version'],capsule_estimator_sha256=sha(implementation),
                             capsule_protocol_sha256=capsule_protocol_sha,capsule_settings=SETTINGS)

    def predict(self,arrays):
        started=time.perf_counter();response=super().predict(arrays)
        floors=[v['fixture']['position_m'][2] for v in response['views'].values() if v['fixture']['accepted']]
        floor=min(floors) if floors else None;original={}
        for camera in CAMERAS:
            original[camera]=response['views'][camera]['capsule']
            calibration=Calibration(arrays['K_'+camera],arrays['T_'+camera])
            response['views'][camera]['capsule']=estimate_capsule_with_fallback(self.masks[camera]==1,arrays['depth_'+camera],
                arrays['valid_'+camera],calibration,self.protocol,floor)
        response['diagnostic_v1_capsule']=original
        response['wall_seconds']=time.perf_counter()-started
        return response


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['candidate','protocol','fixture-protocol','capsule-protocol']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['candidate-sha','protocol-sha','fixture-protocol-sha','capsule-protocol-sha']:p.add_argument('--'+name,required=True)
    p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    service=ReacquisitionInferenceService(a.candidate,a.protocol,a.candidate_sha,a.protocol_sha,threads=a.threads,
        fixture_protocol=a.fixture_protocol,fixture_protocol_sha=a.fixture_protocol_sha,
        capsule_protocol=a.capsule_protocol,capsule_protocol_sha=a.capsule_protocol_sha)
    print(json.dumps(service.metadata,allow_nan=False),flush=True)
    while True:
        header=sys.stdin.buffer.read(4)
        if not header:return
        if len(header)!=4:raise EOFError('Truncated header')
        size=struct.unpack('!I',header)[0]
        if size>8*1024*1024:raise ValueError('Observation packet too large')
        if not size:return
        with np.load(io.BytesIO(read_exact(sys.stdin.buffer,size)),allow_pickle=False) as archive:
            arrays={k:archive[k] for k in archive.files}
        print(json.dumps(service.predict(arrays),allow_nan=False),flush=True)


if __name__=='__main__':main()
