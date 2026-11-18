import json
from types import SimpleNamespace
import cv2
import numpy as np
import pytest

from cl_pickplace.dataset_capture import DatasetCapture
from cl_pickplace.dataset_generate import write_example
from cl_pickplace.dataset_scene import scene_recipe,build_scene
from cl_pickplace.dataset_verify import check_example,verify_dataset,digest


@pytest.fixture
def example(tmp_path,cfg):
    recipe=scene_recipe(cfg,2,43,'none'); s=build_scene(recipe)
    s.reset(np.random.default_rng(recipe['scene_seed']))
    capture=DatasetCapture(s)
    ep=SimpleNamespace(fsm=SimpleNamespace(state=SimpleNamespace(name='HOME')),result=None)
    try: row=write_example(tmp_path,'scene_00002','train',0,'top_cam',capture,ep,recipe,'phase_delay')
    finally: capture.close()
    return tmp_path,row


def refresh_hashes(root,row):
    for name in row['files']: row['files'][name]=digest(root/name)


def test_valid_example_passes_independent_checks(example):
    root,row=example
    _,res,checked,_=check_example(root,row)
    assert checked>0 and res[1] and res[2]


def test_capsule_pose_shift_detected_even_when_consistency_and_hashes_updated(example):
    root,row=example; path=root/'labels'/row['sample_id']/'annotation.json'
    a=json.loads(path.read_text())
    world=np.array(a['world_from_object']);world[0,3]+=.01
    a['world_from_object']=world.tolist(); a['geometries'][0]['world_from_geom']=world.tolist()
    a['camera_from_object']=(np.linalg.inv(a['world_from_camera'])@world).tolist()
    path.write_text(json.dumps(a));refresh_hashes(root,row)
    with pytest.raises(AssertionError,match='surface mismatch'): check_example(root,row)


def test_depth_shift_detected_even_when_hash_updated(example):
    root,row=example; path=root/'labels'/row['sample_id']/'geometry.npz'
    with np.load(path) as d: arrays={k:d[k] for k in d.files}
    arrays['ideal_depth_m']+=.01
    np.savez_compressed(path,**arrays);refresh_hashes(root,row)
    with pytest.raises(AssertionError,match='surface mismatch'): check_example(root,row)


def test_flipped_mask_detected_even_when_hash_updated(example):
    root,row=example; path=root/'labels'/row['sample_id']/'mask.png'
    mask=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    assert cv2.imwrite(str(path),np.fliplr(mask));refresh_hashes(root,row)
    with pytest.raises(AssertionError,match='mask disagrees'): check_example(root,row)


def test_file_corruption_detected(example):
    root,row=example; path=root/'observations'/row['sample_id']/'rgb.png'
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(AssertionError,match='checksum mismatch'): check_example(root,row)


def test_observed_depth_bias_detected_with_fresh_checksum(example,cfg):
    root,row=example; path=root/'observations'/row['sample_id']/'depth.npz'
    with np.load(path) as d: depth=d['depth_m'];valid=d['valid']
    depth[valid==1]+=.02
    np.savez_compressed(path,depth_m=depth,valid=valid);refresh_hashes(root,row)
    noise=scene_recipe(cfg,2,43,'none')['noise']
    with pytest.raises(AssertionError,match='unexpected bias'): check_example(root,row,noise=noise)


def test_split_leakage_is_rejected(tmp_path):
    meta={'generation_complete':True,'expected_samples':0,'samples':0,'scene_count':3,'dataset_seed':43,
          'splits':{'train':[0,1],'validation':[1],'test':[2]},'stages':[]}
    (tmp_path/'dataset.json').write_text(json.dumps(meta))
    (tmp_path/'manifest.jsonl').write_text('')
    result=verify_dataset(tmp_path,replay_count=0)
    assert not result['passed'] and 'scene split overlap/count mismatch' in result['errors']
