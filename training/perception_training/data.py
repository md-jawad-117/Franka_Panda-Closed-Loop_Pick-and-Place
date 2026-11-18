"""Observation-only inputs, supervised visible masks, and scene split preflight."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

MODES={'rgb':3,'rgbd':5}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def safe_path(root,relative):
    root=Path(root).resolve(); path=(root/relative).resolve()
    if not path.is_relative_to(root): raise ValueError('Dataset path escapes its root')
    return path


def read_manifest(root):
    root=Path(root)
    meta=json.loads((root/'dataset.json').read_text())
    rows=[json.loads(s) for s in (root/'manifest.jsonl').read_text().splitlines()]
    if meta.get('schema_version')!='rgbd_capsule_fixture_v1': raise ValueError('Unsupported dataset schema')
    if not meta.get('generation_complete'): raise ValueError('Dataset generation is incomplete')
    if len(rows)!=meta['samples'] or len(rows)!=meta['expected_samples']: raise ValueError('Dataset count mismatch')
    if len({r['sample_id'] for r in rows})!=len(rows): raise ValueError('Duplicate sample IDs')
    scenes=[i for ids in meta['splits'].values() for i in ids]
    if len(set(scenes))!=len(scenes) or set(scenes)!=set(range(meta['scene_count'])):
        raise ValueError('Overlapping or incomplete scene splits')
    mapping={f'scene_{i:05d}':split for split,ids in meta['splits'].items() for i in ids}
    inventory=Counter()
    for row in rows:
        if mapping.get(row['scene_id'])!=row['split']: raise ValueError('Scene split leakage')
        if row['camera'] not in ('wrist_cam','top_cam'): raise ValueError('Unknown camera')
        if not 0<=row['frame_index']<len(meta['stages']): raise ValueError('Unknown stage index')
        sid=row['sample_id']
        if sid!=f'{row["scene_id"]}_f{row["frame_index"]:02d}_{row["camera"]}': raise ValueError('Sample identity mismatch')
        inventory[(row['scene_id'],row['camera'],row['frame_index'])]+=1
    expected={(scene,camera,index) for scene in mapping for camera in ('wrist_cam','top_cam') for index in range(len(meta['stages']))}
    if set(inventory)!=expected or any(n!=1 for n in inventory.values()): raise ValueError('Incomplete or duplicated scene views')
    return meta,rows


def load_arrays(root,row,mode='rgbd',augment=False,rng=None):
    """No pose, ID, ideal depth, or amodal arrays are ever opened here."""
    if mode not in MODES: raise ValueError('Unknown input mode')
    sid=row['sample_id']
    rgb=np.array(Image.open(safe_path(root,f'observations/{sid}/rgb.png')).convert('RGB'),dtype=np.float32)/255
    target=np.array(Image.open(safe_path(root,f'labels/{sid}/mask.png')),dtype=np.int64)
    channels=[rgb.transpose(2,0,1)]
    if mode=='rgbd':
        with np.load(safe_path(root,f'observations/{sid}/depth.npz'),allow_pickle=False) as d:
            if set(d.files)!={'depth_m','valid'}: raise ValueError('Unexpected/privileged observation fields')
            depth=d['depth_m'];valid=d['valid']
        channels += [np.clip(depth,0,2.5)[None]/2.5,valid[None].astype(np.float32)]
    x=np.concatenate(channels,axis=0).astype(np.float32)
    if augment:
        rng=np.random if rng is None else rng
        gain=rng.uniform(.8,1.2); offset=rng.uniform(-.08,.08)
        x[:3]=np.clip(x[:3]*gain+offset,0,1)
        if rng.random()<.5: x=x[:,:,::-1];target=target[:,::-1]
    return np.ascontiguousarray(x),np.ascontiguousarray(target)


def preflight(root,hash_check=True):
    meta,rows=read_manifest(root)
    artifact='full_dataset'
    descriptor=Path(root)/'training-export.json'
    if descriptor.exists():
        export=json.loads(descriptor.read_text())
        if export.get('artifact_kind')!='training_view_of_audited_full_dataset':raise ValueError('Unknown training export kind')
        if export.get('parent_metadata_sha256')!=sha(Path(root)/'dataset.json') or export.get('parent_manifest_sha256')!=sha(Path(root)/'manifest.jsonl'):
            raise ValueError('Training export parent hashes disagree')
        artifact=export['artifact_kind']
    pixels=np.zeros(3,np.int64); split_counts=Counter(); camera_counts=Counter(); rgb_splits={}
    for row in rows:
        sid=row['sample_id']; split_counts[row['split']]+=1;camera_counts[row['camera']]+=1
        # Hash all supervised/input files and annotations. The independent geometric
        # audit belongs to verify_dataset.py; training never loads clean geometry.
        for relative in (f'observations/{sid}/rgb.png',f'observations/{sid}/depth.npz',f'labels/{sid}/mask.png',f'labels/{sid}/annotation.json'):
            path=safe_path(root,relative)
            if relative not in row['files']: raise ValueError('Manifest file missing')
            if hash_check and sha(path)!=row['files'][relative]: raise ValueError(f'Checksum mismatch: {relative}')
        x,y=load_arrays(root,row)
        if x.shape!=(5,meta['height'],meta['width']) or y.shape!=(meta['height'],meta['width']): raise ValueError('Input/target dimensions disagree')
        if not np.isfinite(x).all() or set(np.unique(y))-{0,1,2}: raise ValueError('Invalid inputs/target classes')
        with np.load(safe_path(root,f'observations/{sid}/depth.npz')) as d:
            depth=d['depth_m'];valid=d['valid']
            if depth.dtype!=np.float32 or valid.dtype!=np.uint8 or set(np.unique(valid))-{0,1}:
                raise ValueError('Invalid depth types/validity')
            if np.any(depth[valid==0]!=0) or np.any(depth[valid==1]<=0): raise ValueError('Invalid metric depth')
        if row['split']=='train': pixels+=np.bincount(y.ravel(),minlength=3)
        rgb_hash=sha(safe_path(root,f'observations/{sid}/rgb.png'))
        if rgb_hash in rgb_splits and rgb_splits[rgb_hash]!=row['split']: raise ValueError('Identical RGB across scene splits')
        rgb_splits[rgb_hash]=row['split']
    if np.any(pixels==0): raise ValueError('Training split lacks a class')
    return {'passed':True,'artifact_kind':artifact,'samples':len(rows),'scene_counts':{k:len(v) for k,v in meta['splits'].items()},
            'sample_counts':dict(split_counts),'camera_counts':dict(camera_counts),'train_class_pixels':pixels.tolist(),
            'dataset_metadata_sha256':sha(Path(root)/'dataset.json'),'manifest_sha256':sha(Path(root)/'manifest.jsonl'),
            'inputs':['RGB','observed optical-z depth / 2.5 m','depth validity'],
            'supervision':'visible mask only; simulator poses/IDs/ideal depth/amodal masks are excluded from model inputs',
            'limitation':'training preflight complements the separate geometric dataset audit; it is not that full audit'}
