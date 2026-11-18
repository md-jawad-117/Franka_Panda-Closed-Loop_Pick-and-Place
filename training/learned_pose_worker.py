"""Persistent CPU inference service. Binary NPZ observations in, JSON poses out.

Run only from the simulation client. There is no simulator, label or recipe API.
Stdout is reserved for protocol responses; failures terminate with stderr traceback.
"""
import argparse,hashlib,io,json,struct,sys,time
from pathlib import Path
import numpy as np
import torch
from perception_training.model import TinyUNet,MODEL_VERSION
from pose_evaluation.geometry import Calibration,estimate_pose

CAMERAS=('wrist_cam','top_cam')
FIELDS={f'{field}_{camera}' for camera in CAMERAS for field in ('rgb','depth','valid','K','T')}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class InferenceService:
    def __init__(self,candidate_path,protocol_path,candidate_sha,protocol_sha,threads=2):
        if sha(candidate_path)!=candidate_sha or sha(protocol_path)!=protocol_sha:raise ValueError('Frozen candidate/protocol hash mismatch')
        self.candidate=json.loads(candidate_path.read_text());self.protocol=json.loads(protocol_path.read_text())
        root=candidate_path.resolve().parents[1];self.models={}
        torch.set_num_threads(threads)
        for camera,entry in self.candidate['models'].items():
            path=(root/entry['checkpoint_relative_to_research_root']).resolve()
            if not path.is_relative_to(root) or sha(path)!=entry['sha256']:raise ValueError('Checkpoint hash/path mismatch')
            for name,digest in entry['source_sha256'].items():
                if sha(root/'training/perception_training'/name)!=digest:raise ValueError('Frozen network source changed')
            saved=torch.load(path,map_location='cpu',weights_only=True);cfg=saved['config']
            if cfg['model_version']!=MODEL_VERSION or cfg['camera_filter']!=camera:raise ValueError('Incorrect model/routing')
            model=TinyUNet(cfg['input_channels'],cfg['base']);model.load_state_dict(saved['model_state']);model.eval()
            self.models[camera]=model
        self.metadata={'ready':True,'candidate_id':self.candidate['candidate_id'],'candidate_sha256':candidate_sha,
                       'protocol_sha256':protocol_sha,'torch':str(torch.__version__),'numpy':str(np.__version__),
                       'python':sys.version.split()[0],'threads':threads,'device':'cpu',
                       'worker_sha256':sha(__file__),'geometry_sha256':sha(Path(__file__).parent/'pose_evaluation/geometry.py'),
                       'checkpoint_sha256':{c:e['sha256'] for c,e in self.candidate['models'].items()}}

    def predict(self,arrays):
        if set(arrays)!=FIELDS:raise ValueError('Unexpected/missing fields; only sensor observations and calibration are permitted')
        result={};start=time.perf_counter()
        with torch.inference_mode():
            for camera in CAMERAS:
                rgb=arrays[f'rgb_{camera}'];depth=arrays[f'depth_{camera}'];valid=arrays[f'valid_{camera}']
                if rgb.shape!=(240,320,3) or rgb.dtype!=np.uint8 or depth.shape!=(240,320) or valid.shape!=depth.shape:
                    raise ValueError('Unsupported image shape/type')
                if not np.isfinite(depth).all() or np.any(depth<0) or np.any(depth[valid==0]!=0):raise ValueError('Invalid observed depth')
                calibration=Calibration(arrays[f'K_{camera}'],arrays[f'T_{camera}'])
                x=np.concatenate([rgb.astype(np.float32).transpose(2,0,1)/255,
                                  np.clip(depth,0,2.5)[None]/2.5,valid[None].astype(np.float32)])
                pred=self.models[camera](torch.from_numpy(x[None])).argmax(1)[0].numpy()
                result[camera]={target:estimate_pose(pred==cls,depth,valid,calibration,target,self.protocol)
                                for cls,target in ((1,'capsule'),(2,'fixture'))}
        return {'views':result,'wall_seconds':time.perf_counter()-start}


def read_exact(stream,size):
    chunks=[]
    while size:
        chunk=stream.read(size)
        if not chunk:raise EOFError('Truncated observation packet')
        chunks.append(chunk);size-=len(chunk)
    return b''.join(chunks)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True);p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--candidate-sha',required=True);p.add_argument('--protocol-sha',required=True)
    p.add_argument('--threads',type=int,default=2);args=p.parse_args()
    service=InferenceService(args.candidate,args.protocol,args.candidate_sha,args.protocol_sha,args.threads)
    print(json.dumps(service.metadata,allow_nan=False),flush=True)
    while True:
        header=sys.stdin.buffer.read(4)
        if not header:return
        if len(header)!=4:raise EOFError('Truncated header')
        size=struct.unpack('!I',header)[0]
        if size>8*1024*1024:raise ValueError('Observation packet too large')
        if size==0:return
        with np.load(io.BytesIO(read_exact(sys.stdin.buffer,size)),allow_pickle=False) as archive:
            arrays={key:archive[key] for key in archive.files}
        print(json.dumps(service.predict(arrays),allow_nan=False),flush=True)


if __name__=='__main__':main()
