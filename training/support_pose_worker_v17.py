"""Frozen networks plus observed-support geometry; no simulator dependency."""
from pathlib import Path
import argparse,copy,io,json,struct,sys,time
import numpy as np
from reacquisition_pose_worker import ReacquisitionInferenceService
from learned_pose_worker import sha,read_exact,CAMERAS
from pose_evaluation.geometry import Calibration
from pose_evaluation.observed_support_v17 import SETTINGS,observed_planes,estimate_fixture
from pose_evaluation.capsule_surface_v4 import estimate_capsule_with_fallback


class SupportInferenceService(ReacquisitionInferenceService):
    def __init__(self,*args,prefer_top=False,entrypoint=None,**kwargs):
        super().__init__(*args,**kwargs)
        path=Path(__file__).parent/'support_plane_protocol_v17.json';protocol=json.loads(path.read_text())
        implementation=Path(__file__).parent/'pose_evaluation/observed_support_v17.py'
        if protocol['implementation_sha256']!=sha(implementation) or protocol['settings']!=SETTINGS:raise ValueError('Support estimator/protocol changed')
        if protocol['base_protocol_sha256']!=self.metadata['protocol_sha256']:raise ValueError('Support base protocol mismatch')
        self.prefer_top=prefer_top
        self.metadata.update(reacquisition_worker_sha256=self.metadata['worker_sha256'],worker_sha256=sha(entrypoint or __file__),
            support_worker_sha256=sha(__file__),support_estimator_sha256=sha(implementation),support_protocol_sha256=sha(path),
            support_settings=SETTINGS,fixture_camera_priority='top_then_wrist' if prefer_top else 'original_selection')

    def predict(self,arrays):
        started=time.perf_counter();response=super().predict(arrays)
        top=Calibration(arrays['K_top_cam'],arrays['T_top_cam'])
        planes=observed_planes(arrays['depth_top_cam'],arrays['valid_top_cam'],self.masks['top_cam']==0,top)
        response['diagnostic_original_fixture']={camera:copy.deepcopy(response['views'][camera]['fixture']) for camera in CAMERAS}
        if planes:
            for camera in CAMERAS:
                calibration=Calibration(arrays['K_'+camera],arrays['T_'+camera])
                response['views'][camera]['fixture']=estimate_fixture(self.masks[camera]==2,arrays['depth_'+camera],arrays['valid_'+camera],calibration,self.protocol,planes)
        response['diagnostic_supported_fixture']={camera:copy.deepcopy(response['views'][camera]['fixture']) for camera in CAMERAS}
        if self.prefer_top and response['views']['top_cam']['fixture']['accepted'] and response['views']['wrist_cam']['fixture']['accepted']:
            response['views']['wrist_cam']['fixture']['accepted']=False
            response['views']['wrist_cam']['fixture']['reasons']=['top_fixture_view_preferred']
        floors=[v['fixture']['position_m'][2] for v in response['views'].values() if v['fixture']['accepted']]
        floor=min(floors) if floors else None
        for camera in CAMERAS:
            calibration=Calibration(arrays['K_'+camera],arrays['T_'+camera])
            response['views'][camera]['capsule']=estimate_capsule_with_fallback(self.masks[camera]==1,arrays['depth_'+camera],arrays['valid_'+camera],calibration,self.protocol,floor)
        response['observed_support_planes']=planes;response['wall_seconds']=time.perf_counter()-started
        return response


def main(prefer_top=False,entrypoint=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['candidate','protocol','fixture-protocol','capsule-protocol']:p.add_argument('--'+name,type=Path,required=True)
    for name in ['candidate-sha','protocol-sha','fixture-protocol-sha','capsule-protocol-sha']:p.add_argument('--'+name,required=True)
    p.add_argument('--threads',type=int,default=2);a=p.parse_args()
    service=SupportInferenceService(a.candidate,a.protocol,a.candidate_sha,a.protocol_sha,threads=a.threads,
        fixture_protocol=a.fixture_protocol,fixture_protocol_sha=a.fixture_protocol_sha,
        capsule_protocol=a.capsule_protocol,capsule_protocol_sha=a.capsule_protocol_sha,prefer_top=prefer_top,entrypoint=entrypoint)
    print(json.dumps(service.metadata,allow_nan=False),flush=True)
    while True:
        header=sys.stdin.buffer.read(4)
        if not header:return
        if len(header)!=4:raise EOFError('Truncated header')
        size=struct.unpack('!I',header)[0]
        if size>8*1024*1024:raise ValueError('Observation packet too large')
        if not size:return
        with np.load(io.BytesIO(read_exact(sys.stdin.buffer,size)),allow_pickle=False) as data:arrays={k:data[k] for k in data.files}
        print(json.dumps(service.predict(arrays),allow_nan=False),flush=True)


if __name__=='__main__':main()
