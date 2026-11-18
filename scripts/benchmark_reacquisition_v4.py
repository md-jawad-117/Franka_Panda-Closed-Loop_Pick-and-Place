"""Matched v3/v4 recovery development comparison; no reserved dataset access."""
from pathlib import Path
import argparse,copy,hashlib,json,sys,time,os
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cl_pickplace.config import load_config
from cl_pickplace.episode import Episode
from benchmark_moving_destination import evaluation_trace


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def recovery_evaluation_trace(ep,original):
    base=evaluation_trace(ep,original)
    fingers={mujoco.mj_name2id(ep.sim.model,mujoco.mjtObj.mjOBJ_BODY,name) for name in ('left_finger','right_finger')}
    def log(obs):
        base(obs)
        contact=False
        for pair in ep.sim.data.contact:
            a,b=int(pair.geom1),int(pair.geom2)
            if ep.sim.capsule_geom not in (a,b):continue
            other=b if a==ep.sim.capsule_geom else a
            if int(ep.sim.model.geom_bodyid[other]) in fingers and pair.dist<=0:contact=True
        ep.log.visual[-1]['evaluation_only']['finger_capsule_contact']=contact
    return log


def configuration(method,condition):
    cfg=load_config(ROOT/('config/learned_recovery_v3.yaml' if method=='v3' else 'config/learned_reacquisition_v4.yaml'))
    if method=='v4_push':cfg['task']['visual_acquisition']['rim_extraction']['enabled']=True
    if method=='quality_only':cfg['task'].pop('visual_acquisition')
    if method=='search_no_rim':
        cfg['task']['visual_acquisition']['rim_aware_grasp']=False
        cfg['task']['visual_acquisition'].pop('rim_extraction',None)
    move=cfg['perturbations']['scheduled']['slot_move']
    move.update(shift_m_range=[.03,.03],yaw_change_deg_range=[-15.,15.],move_duration_s=.5,time_s_range=[5.5,5.5])
    if condition=='phase':
        cfg['perturbations']['schedule_mode']='state'
        move.update(states=['DESCEND_SLOT'],trigger_delay_by_state={'DESCEND_SLOT':[0.,0.]},trigger_height_by_state={'DESCEND_SLOT':.04})
    if condition=='release_phase':
        cfg['perturbations']['schedule_mode']='state'
        move.update(states=['RELEASE'],trigger_delay_by_state={'RELEASE':[.1,.1]},trigger_height_by_state={})
    return cfg


def fingerprint():
    paths=list((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'scripts/benchmark_moving_destination.py',
        ROOT/'config/learned_recovery_v3.yaml',ROOT/'config/learned_reacquisition_v4.yaml',ROOT/'config/reacquisition_protocol_v4.json',
        ROOT/'training/candidate_specialists_v3.json',ROOT/'training/learned_pose_worker.py',ROOT/'training/surface_pose_worker.py',
        ROOT/'training/pose_protocol_v1.json',ROOT/'training/fixture_surface_protocol_v2.json',
        ROOT/'training/pose_evaluation/geometry.py',ROOT/'training/pose_evaluation/fixture_surface_v2.py',
        ROOT/'training/reacquisition_pose_worker.py',ROOT/'training/pose_evaluation/capsule_surface_v4.py',ROOT/'training/capsule_surface_protocol_v4.json']
    return {p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(paths))}


def summarize(rows):
    groups={}
    for r in rows:groups.setdefault(r['condition']+'/'+r['method'],[]).append(r)
    return {key:{'n':len(rs),'successes':sum(r['result']['success'] for r in rs),
                 'timeouts':sum(r['result']['timeout'] for r in rs),'aborts':sum(r['result']['final_state']=='ABORT' for r in rs),
                 'retries':sum(r['placement_retries'] for r in rs),
                 'acquisition_confirmed':sum(a['outcome']=='confirmed' for r in rs for a in r['acquisition_history']),
                 'acquisition_inconclusive':sum(a['outcome']=='inconclusive' for r in rs for a in r['acquisition_history']),
                 'done_physical_failure':sum(r['result']['final_state']=='DONE' and not r['result']['success'] for r in rs),
                 'mean_successful_completion_s':float(np.mean([r['result']['time_to_complete_s'] for r in rs if r['result']['success']])) if any(r['result']['success'] for r in rs) else None}
            for key,rs in groups.items()}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--development',action='store_true');p.add_argument('--condition',choices=['nominal','late','phase','release_phase'],required=True)
    p.add_argument('--methods',nargs='+',choices=['v3','quality_only','search_no_rim','v4','v4_push'])
    p.add_argument('--seeds',nargs='+',type=int);a=p.parse_args()
    protocol=json.loads((ROOT/'config/reacquisition_protocol_v4.json').read_text())
    if a.seeds and not a.development:raise ValueError('Custom seeds require --development')
    seeds=a.seeds or ([3,400,403,404,405] if a.development else protocol['seeds'])
    methods=a.methods or (['v4'] if a.development else protocol['methods'][a.condition])
    if not a.development:assert methods==protocol['methods'][a.condition]
    manifest={'protocol':protocol,'development':a.development,'condition':a.condition,'methods':methods,'seeds':seeds,'source_sha256':fingerprint(),
              'thread_environment':{name:os.environ.get(name) for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
    a.out.mkdir(parents=True,exist_ok=True);mp=a.out/'manifest.json'
    if mp.exists():assert json.loads(mp.read_text())==manifest,'Changed source or protocol: use a new run'
    else:
        mp.write_text(json.dumps(manifest,indent=2))
        for rel in manifest['source_sha256']:
            dst=a.out/'frozen_source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/rel).read_bytes())
    (a.out/'episodes').mkdir(exist_ok=True);rows=[]
    for method in methods:
        cfg=configuration(method,a.condition);ep=None
        try:
            for seed in seeds:
                path=a.out/'episodes'/f'{a.condition}__{method}__{seed}.json'
                if path.exists():rows.append(json.loads(path.read_text()));continue
                if ep is None:
                    ep=Episode(cfg,mode='closed_loop',perception_mode='learned',perturb='none' if a.condition=='nominal' else 'slot_move')
                    ep._log=recovery_evaluation_trace(ep,ep._log)
                start=time.perf_counter();ep.reset(seed)
                while not ep.done:ep.step()
                row={'condition':a.condition,'method':method,'seed':seed,'result':ep.result,'config':copy.deepcopy(cfg),
                     'initial_poses':ep.poses,'plan':ep.perturber.manifest(),'delivered':ep.perturber.delivery(),
                     'transitions':ep.fsm.stats.transitions,'visual_trace':ep.log.visual,'capture_trace':ep.perception._learned.history,
                     'learned_perception':ep.perception.learned_summary(),'verification_history':ep.fsm.verification_history,
                     'acquisition_history':ep.fsm.acquisition_history,'placement_retries':ep.fsm.placement_retries,
                     'recovery_grasp_attempts':ep.fsm.recovery_grasp_attempts,'release_vetoes':ep.fsm.release_vetoes,
                     'extraction_history':ep.fsm.extraction_history,'extraction_attempts':ep.fsm.extraction_attempts,
                     'wall_seconds':time.perf_counter()-start}
                path.write_text(json.dumps(row,allow_nan=False));rows.append(row)
                print(json.dumps({'condition':a.condition,'method':method,'seed':seed,'result':ep.result,
                                  'acquisitions':ep.fsm.acquisition_history,'grasp_attempts':ep.fsm.recovery_grasp_attempts}),flush=True)
                (a.out/'summary.json').write_text(json.dumps(summarize(rows),indent=2))
        finally:
            if ep is not None:ep.perception.close()
    (a.out/'complete.json').write_text(json.dumps({'episodes':len(rows)}))


if __name__=='__main__':main()
