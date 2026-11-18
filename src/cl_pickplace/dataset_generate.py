"""Dataset generation with scene-level splits, privileged labels and replay state."""
import copy
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from .config import load_config
from .dataset_capture import DatasetCapture
from .dataset_scene import build_scene,scene_recipe,split_scenes
from .episode import Episode
from .provenance import run_metadata

STAGES=[('HOME',0),('APPROACH_OBJECT',.2),('DESCEND_OBJECT',.15),('GRASP',.2),
        ('LIFT',.15),('TRANSPORT',.2),('DESCEND_SLOT',.5),('RELEASE',.15),('RETREAT',.15),('DONE',.2)]


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def write_example(out,scene_id,split,index,camera,capture,ep,recipe,reason):
    seed=np.random.SeedSequence([recipe['scene_seed'],index,0 if camera=='wrist_cam' else 1,772])
    rgb,observed,valid,mask,clean,seg,amodal,annotation=capture.capture(camera,np.random.default_rng(seed),recipe['noise'])
    sid=f'{scene_id}_f{index:02d}_{camera}'
    observation=out/'observations'/sid
    labels=out/'labels'/sid
    observation.mkdir(parents=True)
    labels.mkdir(parents=True)
    assert cv2.imwrite(str(observation/'rgb.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
    np.savez_compressed(observation/'depth.npz',depth_m=observed,valid=valid)
    assert cv2.imwrite(str(labels/'mask.png'),mask)
    np.savez_compressed(labels/'geometry.npz',ideal_depth_m=clean,raw_segmentation=seg,amodal_masks=amodal)
    annotation.update({'sample_id':sid,'scene_id':scene_id,'split':split,'frame_index':index,
                       'scene_seed':recipe['scene_seed'],'actual_state':ep.fsm.state.name,
                       'requested_stage':STAGES[index][0],'capture_reason':reason,
                       'privileged_label_warning':'Only observation RGB/noisy depth are model inputs; poses/state/masks/ideal depth are supervision or QA.',
                       'task_result_at_capture':ep.result})
    (labels/'annotation.json').write_text(json.dumps(annotation,indent=2,allow_nan=False))
    files=[observation/'rgb.png',observation/'depth.npz',labels/'mask.png',labels/'geometry.npz',labels/'annotation.json']
    return {'sample_id':sid,'scene_id':scene_id,'split':split,'camera':camera,'frame_index':index,
            'files':{p.relative_to(out).as_posix():sha(p) for p in files}}


def generate_dataset(output,scenes=25,seed=43,width=320,height=240,base_cfg=None):
    out=Path(output)
    if out.exists(): raise FileExistsError(f'Refusing to overwrite dataset: {out}')
    splits=split_scenes(scenes,seed+801)
    cfg=copy.deepcopy(base_cfg or load_config())
    out.mkdir(parents=True)
    (out/'scenes').mkdir()
    metadata={'schema_version':'rgbd_capsule_fixture_v1','dataset_seed':seed,'scene_count':scenes,
              'expected_samples':scenes*len(STAGES)*2,'width':width,'height':height,
              'classes':{'0':'background','1':'capsule','2':'fixture'},'splits':splits,
              'stages':STAGES,'source':'oracle-controlled MuJoCo rollouts; simulator IDs generate labels',
              'limitations':['static visual props outside manipulation corridor have collisions disabled',
                             'single capsule/fixture families; varied dimensions, not unseen categories',
                             'simple Gaussian/dropout depth noise; RGB and extrinsics currently ideal',
                             'pilot scene splits are development data, not the final paper test set'],
              'provenance':run_metadata(cfg,'ground_truth'),'scene_files':{},'generation_complete':False}
    metadata['provenance']['sensor_assumptions']='Oracle controls dataset collection only; model inputs are rendered RGB and noisy optical-z depth. Poses/IDs/clean depth are privileged labels.'
    (out/'dataset.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    rows=[]
    t0=time.perf_counter()
    for split,indices in splits.items():
        for rank,index in enumerate(indices):
            scene_id=f'scene_{index:05d}'
            recipe=scene_recipe(cfg,index,seed,'all' if rank%2 else 'none')
            (out/'scenes'/f'{scene_id}.json').write_text(json.dumps(recipe,indent=2,allow_nan=False))
            metadata['scene_files'][f'scenes/{scene_id}.json']=sha(out/'scenes'/f'{scene_id}.json')
            sim=build_scene(recipe)
            ep=Episode(recipe['config'],sim=sim,perception_mode='ground_truth',mode='closed_loop',perturb=recipe['perturb'])
            capture=None
            try:
                ep.reset(recipe['scene_seed'])
                capture=DatasetCapture(sim,width,height,recipe['appearance']['shadow'])
                completed=set()
                entries={ep.fsm.state.name:ep.t}
                def save(i,reason):
                    for camera in ('wrist_cam','top_cam'):
                        rows.append(write_example(out,scene_id,split,i,camera,capture,ep,recipe,reason))
                    completed.add(i)
                save(0,'phase_delay')
                while len(completed)<len(STAGES):
                    current=ep.fsm.state.name
                    entries.setdefault(current,ep.t)
                    for i,(stage,delay) in enumerate(STAGES):
                        if i in completed or stage not in entries: continue
                        if current==stage and ep.t-entries[stage]>=delay-1e-9:
                            save(i,'phase_delay')
                        elif current!=stage:
                            save(i,'phase_exit_before_delay')
                    if ep.done:
                        for i in range(len(STAGES)):
                            if i not in completed: save(i,'terminal_fallback')
                        break
                    ep.step()
                with open(out/'manifest.jsonl','a',encoding='utf-8') as f:
                    for row in rows[-len(STAGES)*2:]: f.write(json.dumps(row)+'\n')
            finally:
                if capture is not None: capture.close()
                ep.perception.close()
            print(f'{scene_id} {split}: {len(rows)}/{metadata["expected_samples"]} examples',flush=True)
    metadata.update(generation_complete=True,samples=len(rows),generation_seconds=time.perf_counter()-t0)
    (out/'dataset.json').write_text(json.dumps(metadata,indent=2,allow_nan=False))
    return out
