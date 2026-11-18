"""Frozen v23 vs v24 occlusion-gated centering comparison with full restorable episodes."""
from pathlib import Path
import argparse,copy,json,os,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cl_pickplace.camera_framing_wide import widen_wrist,design
from cl_pickplace.episode import Episode
from cl_pickplace.settled_inspection import InspectionFSM
from cl_pickplace.centering_feedback import CenteringFSM
from cl_pickplace.reobserving_centering import ReobservingCenterFSM
from cl_pickplace.bounded_blind_centering import BoundedBlindCenterFSM
from cl_pickplace.confirmed_blind_centering import ConfirmedBlindCenterFSM
from cl_pickplace.occlusion_gated_centering import OcclusionGatedCenterFSM
from recovery_diagnostics_v5 import contact_diagnostics
from benchmark_reacquisition_v4 import configuration,recovery_evaluation_trace
from benchmark_confirmed_v23 import configurations as confirmed_configurations
from fresh_grasp_validation_v9 import frozen_sources,STATE_FIELDS
from evaluate_grasp_planner_v7 import sha

PROTOCOL=ROOT/'config/occlusion_gated_protocol_v24.json'


def configurations(protocol,condition):
    parent=json.loads((ROOT/protocol['parent_control_protocol']).read_text())
    baseline=confirmed_configurations(parent,condition)['confirmed_v23'];candidate=copy.deepcopy(baseline)
    candidate['task']['visual_centering'].update(protocol['runtime_additions'])
    return {'confirmed_v23':baseline,'occlusion_v24':candidate}


def initialize(out,cohort):
    if out.exists():raise FileExistsError(out)
    protocol=json.loads(PROTOCOL.read_text());sources=frozen_sources()
    extras=[PROTOCOL,Path(__file__),ROOT/protocol['parent_control_protocol'],ROOT/'config/blind_centering_protocol_v22.json',ROOT/'config/reobserving_control_protocol_v21.json',ROOT/'scripts/benchmark_confirmed_v23.py',ROOT/'config/hand_envelope_v24.npz',ROOT/'scripts/extract_hand_envelope_v24.py',ROOT/'config/centering_control_protocol_v20.json',ROOT/'scripts/benchmark_blind_v22.py',ROOT/'scripts/benchmark_reobserving_v21.py',ROOT/'scripts/assay_recovery_visibility_v22.py',ROOT/'config/centering_assay_protocol_v19.json',
        ROOT/'config/support_control_protocol_v17.json',ROOT/'config/inspection_protocol_v14.json',ROOT/'training/support_plane_protocol_v17.json',
        ROOT/'config/framing_wide_protocol_v13.json',ROOT/'config/retention_protocol_v10r1.json',
        ROOT/'scripts/benchmark_centering_v20.py',ROOT/'scripts/benchmark_support_v17.py',ROOT/'scripts/benchmark_inspection_v14.py',
        ROOT/'training/support_pose_worker_v17.py',ROOT/'training/support_top_worker_v17.py',ROOT/'training/pose_evaluation/observed_support_v17.py']
    sources.update({p.relative_to(ROOT).as_posix():sha(p) for p in extras})
    prior=json.loads((ROOT/'training/results/retention-v10r1/development/manifest.json').read_text())
    for rel,digest in prior['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    spec={**protocol[cohort],'methods':protocol['methods']};out.mkdir(parents=True)
    manifest={'cohort':cohort,'protocol':protocol,'planned':spec,'source_sha256':sources,
        'configurations':{c:configurations(protocol,c) for c in spec['conditions']},
        'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    for rel in sources:
        dest=out/'frozen_source'/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/rel).read_bytes())
    print(json.dumps({'initialized':str(out),'planned':spec,'camera':manifest['configurations']['nominal']['confirmed_v23']['sim']['wrist_cam']}),flush=True)


def check(out):
    manifest=json.loads((out/'manifest.json').read_text())
    assert manifest['protocol']==json.loads(PROTOCOL.read_text())
    for rel,digest in manifest['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    return manifest


def load_case(directory):
    pointer=json.loads((directory/'complete.json').read_text());attempt=directory/pointer['attempt']
    for name,digest in pointer['files_sha256'].items():assert sha(attempt/name)==digest,(directory,name)
    return json.loads((attempt/'episode.json').read_text()),attempt


def summarize(rows,spec):
    groups={}
    for condition in spec['conditions']:
        for method in spec['methods']:
            group=[r for r in rows if r['condition']==condition and r['method']==method]
            groups[condition+'/'+method]={'n':len(group),'placement_successes':sum(r['result']['success'] for r in group),
                'aborts':sum(r['result']['final_state']=='ABORT' for r in group),'timeouts':sum(r['result']['timeout'] for r in group),
                'probe_confirmations':sum(e['event']=='probe_confirmed' for r in group for e in r['retention_history']),
                'retention_retries':sum(r['retention_retries'] for r in group),
                'successful_placement_retry_episodes':sum(r['result']['success'] and r['placement_retries']>0 for r in group),
                'disturbance_started_cases':sum(bool(r['delivered']) for r in group),
                'centering_attempts':sum(r.get('centering_attempts',0) for r in group),
                'successful_centering_episodes':sum(r['result']['success'] and r.get('centering_attempts',0)>0 for r in group),
                'contact_free_successes':sum(r['result']['success'] and not any(v['evaluation_only']['environment_contact'] for v in r['visual_trace']) for r in group)}
    return groups


def run(out):
    manifest=check(out);spec=manifest['planned'];rows=[]
    for condition in spec['conditions']:
        for method in spec['methods']:
            cfg=manifest['configurations'][condition][method];ep=None
            try:
                for seed in spec['seeds']:
                    case=f'{condition}__{method}__{seed}';directory=out/case
                    if (directory/'complete.json').exists():rows.append(load_case(directory)[0]);continue
                    directory.mkdir(exist_ok=True);attempt=directory/f'attempt_{1+len(list(directory.glob("attempt_*"))):04d}';attempt.mkdir()
                    states=[]
                    if ep is None:
                        ep=Episode(cfg,mode='closed_loop',perception_mode='learned',perturb='none' if condition=='nominal' else 'slot_move')
                        ep.fsm=(OcclusionGatedCenterFSM if method=='occlusion_v24' else ConfirmedBlindCenterFSM)(cfg,ep.sim.home_q,ep.sim.table_z,'closed_loop')
                        old_log=recovery_evaluation_trace(ep,ep._log)
                        def log(obs):
                            old_log(obs);sim=ep.sim
                            ep.log.visual[-1].update(retention_phase=getattr(ep.fsm,'retention_phase','baseline'),
                                ee_rotation=obs.ee_mat.tolist(),capsule_estimated_axis=None if obs.est.capsule.axis is None else obs.est.capsule.axis.tolist())
                            ep.log.visual[-1]['evaluation_only']['capsule_true_axis']=sim.data.geom_xmat[sim.capsule_geom].reshape(3,3)[:,2].tolist()
                            groups=contact_diagnostics(sim);env=groups['finger_fixture']+groups['hand_fixture']+groups['finger_table']
                            ep.log.visual[-1]['centering_phase']=getattr(ep.fsm,'centering_phase',None)
                            ep.log.visual[-1]['evaluation_only'].update(environment_contact=bool(env),
                                environment_normal_force_sum_N=sum(c['normal_force_N'] for c in env),
                                maximum_environment_penetration_m=max((c['penetration_m'] for c in env),default=0.))
                            states.append({'t':np.array(sim.time),**{k:np.array(getattr(sim.data,k),copy=True) for k in STATE_FIELDS}})
                        ep._log=log
                    start=time.perf_counter()
                    try:
                        ep.reset(seed)
                        while not ep.done:ep.step()
                        row={'case':case,'condition':condition,'method':method,'seed':seed,'result':ep.result,'config':copy.deepcopy(cfg),
                            'initial_poses':ep.poses,'plan':ep.perturber.manifest(),'delivered':ep.perturber.delivery(),
                            'transitions':ep.fsm.stats.transitions,'visual_trace':ep.log.visual,'capture_trace':ep.perception._learned.history,
                            'learned_perception':ep.perception.learned_summary(),'verification_history':ep.fsm.verification_history,
                            'acquisition_history':ep.fsm.acquisition_history,'placement_retries':ep.fsm.placement_retries,
                            'retention_history':getattr(ep.fsm,'retention_history',[]),'retention_retries':getattr(ep.fsm,'retention_retries',0),'inspection_history':getattr(ep.fsm,'inspection_history',[]),'centering_history':getattr(ep.fsm,'centering_history',[]),'centering_attempts':getattr(ep.fsm,'centering_attempts',0),'occlusion_gated':getattr(ep.fsm,'occlusion_gated',[]),
                            'wall_seconds':time.perf_counter()-start,'manifest_sha256':sha(out/'manifest.json')}
                        (attempt/'episode.json').write_text(json.dumps(row,allow_nan=False))
                        np.savez_compressed(attempt/'control_states.npz',**{k:np.stack([s[k] for s in states]) for k in states[0]})
                        (directory/'complete.json').write_text(json.dumps({'attempt':attempt.name,'files_sha256':{p.name:sha(p) for p in attempt.iterdir() if p.is_file()}},indent=2))
                        rows.append(row);print(json.dumps({'case':case,'result':ep.result,
                            'events':[e for e in row['retention_history'] if e['event']!='visual_sample'],'wall_seconds':row['wall_seconds']}),flush=True)
                    except BaseException as exc:
                        (attempt/'technical_failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc),'t':ep.t}));raise
            finally:
                if ep is not None:ep.perception.close()
    for condition in spec['conditions']:
        for seed in spec['seeds']:
            pair=[r for r in rows if r['condition']==condition and r['seed']==seed]
            assert len(pair)==len(spec['methods'])
            assert all(r['initial_poses']==pair[0]['initial_poses'] and r['plan']==pair[0]['plan'] for r in pair)
    result={'episodes':len(rows),'groups':summarize(rows,spec),'matched_initial_poses_and_planned_events':True}
    (out/'complete.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['init','run'])
    p.add_argument('--cohort',choices=['development','fresh_smoke'],default='development');p.add_argument('--out',required=True,type=Path)
    a=p.parse_args();out=a.out.resolve()
    if a.command=='init':initialize(out,a.cohort)
    else:run(out)


if __name__=='__main__':main()
