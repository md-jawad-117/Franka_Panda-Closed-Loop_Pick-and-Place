"""Frozen new-seed complete-episode collection and prospective checkpoint assay."""
from pathlib import Path
import argparse,copy,json,os,sys,time
from unittest.mock import patch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cl_pickplace.episode import Episode
from cl_pickplace.sim import Sim
from cl_pickplace.learned_perception import WorkerClient
from cl_pickplace.kinematics import quat_to_mat
from cl_pickplace.grasp_planner import validate_observation
from cl_pickplace.grasp_capture_planner import base_observation
from cl_pickplace.grasp_collision import extract_robot_cad
from benchmark_reacquisition_v4 import configuration,fingerprint,recovery_evaluation_trace
from recovery_diagnostics_v5 import FIELDS,frame_diagnostic
from evaluate_grasp_planner_v7 import sha,robot_templates,physical_trial
from evaluate_grasp_capture_v8 import calibration,planning_world,select
from evaluate_capture_ablation_v8 import ablation_protocol

PROTOCOL=ROOT/'config/fresh_grasp_protocol_v9.json'
STATE_FIELDS=('qpos','qvel','ctrl','act','qacc_warmstart','mocap_pos','mocap_quat')


def eligibility(frame,condition):
    return frame['state_before']=='DESCEND_OBJECT' and (condition=='nominal' or frame['placement_retries']>0)


def observation(frame,visual):
    return {'t':frame['t'],'capsule_t':visual['capsule_measurement_t'],'fixture_t':visual['slot_measurement_t'],
            'capsule_visible':frame['capsule_visible'],'fixture_visible':visual['slot_visible'],
            'motion_pending':visual['slot_observation'].get('motion_pending',False),
            'capsule_position_m':frame['capsule_estimate_m'],'capsule_axis_world':frame['capsule_estimated_axis'],
            'fixture_position_m':visual['slot_estimate'],'fixture_yaw_rad':visual['slot_estimated_yaw_rad'],
            'ee_position_m':frame['ee_position_m'],'ee_rotation':frame['ee_rotation'],'finger_width_m':frame['finger_width_m'],
            'recorded_target_m':frame['command_position_m'],'recorded_rotation':None if frame['command_quaternion'] is None else quat_to_mat(np.array(frame['command_quaternion'])).tolist()}


def usability(frame,obs,parent):
    if frame['state_after']!='DESCEND_OBJECT':return 'controller_left_first_descent'
    if obs['recorded_target_m'] is None or obs['recorded_rotation'] is None:return 'missing_recorded_visual_target'
    if obs['capsule_axis_world'] is None:return 'missing_capsule_axis'
    reason=validate_observation(base_observation(obs),parent)
    return reason


def frozen_sources():
    paths=fingerprint()
    extra=[PROTOCOL,Path(__file__),ROOT/'scripts/recovery_diagnostics_v5.py',ROOT/'scripts/evaluate_grasp_planner_v7.py',
           ROOT/'scripts/evaluate_grasp_capture_v8.py',ROOT/'scripts/evaluate_capture_ablation_v8.py',ROOT/'scripts/probe_grasp_feasibility_v5.py',
           ROOT/'config/grasp_candidate_v8.json',ROOT/'config/grasp_capture_protocol_v8.json',ROOT/'config/grasp_capture_ablation_v8.json',
           ROOT/'config/grasp_planner_protocol_v7.json',ROOT/'config/grasp_feasibility_protocol_v5.json']
    extra+=list((ROOT/'assets').rglob('*.xml'))+list((ROOT/'third_party/mujoco_menagerie/franka_emika_panda').rglob('*'))
    extra+=list((ROOT/'training/results/kaggle_specialization_v3_seed17').glob('*/best.pt'))
    paths.update({p.relative_to(ROOT).as_posix():sha(p) for p in extra if p.is_file()})
    return paths


def initialize(out):
    if out.exists():raise FileExistsError(out)
    protocol=json.loads(PROTOCOL.read_text());assert len(protocol['conditions'])*len(protocol['seeds'])==24
    # Existing v4 bytes must still match their original frozen comparison.
    old=json.loads((ROOT/'training/results/reacquisition-v4/comparison_frozen/nominal/manifest.json').read_text())
    for rel,digest in old['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    sources=frozen_sources();out.mkdir(parents=True)
    manifest={'protocol':protocol,'source_sha256':sources,'original_v4_sha256':old['source_sha256'],
              'candidate':json.loads((ROOT/'config/grasp_candidate_v8.json').read_text()),
              'condition_configurations':{c:configuration('v4',c) for c in protocol['conditions']},
              'thread_environment':{key:os.environ.get(key) for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    for rel in sources:
        dst=out/'frozen_source'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/rel).read_bytes())
    print(json.dumps({'initialized':str(out),'cases':24,'source_files':len(sources)}),flush=True)


def check(out):
    m=json.loads((out/'manifest.json').read_text());assert m['protocol']==json.loads(PROTOCOL.read_text())
    for rel,digest in m['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    return m


def collect(out,condition,seeds=None):
    manifest=check(out);protocol=manifest['protocol'];assert condition in protocol['conditions']
    selected=seeds or protocol['seeds'];assert set(selected)<=set(protocol['seeds'])
    cfg=manifest['condition_configurations'][condition];ep=None;context={}
    class Recorder:
        def __init__(self,settings,on_wait=None):
            self.inner=WorkerClient(settings,on_wait);self.metadata=self.inner.metadata
        def predict(self,arrays):
            assert set(arrays)==FIELDS
            response=self.inner.predict(arrays)
            context.update(arrays={key:value.copy() for key,value in arrays.items()},response=copy.deepcopy(response),t=ep.t)
            return response
        def close(self):self.inner.close()
    try:
        with patch('cl_pickplace.learned_perception.WorkerClient',Recorder):
            for seed in selected:
                case=f'{condition}__v4__{seed}';case_dir=out/condition/case;case_dir.mkdir(parents=True,exist_ok=True)
                if (case_dir/'complete.json').exists():continue
                completed=sorted(case_dir.glob('attempt_*/attempt_complete.json'))
                if completed:
                    (case_dir/'complete.json').write_text(json.dumps({'attempt':completed[0].parent.name,'attempt_complete_sha256':sha(completed[0])},indent=2));continue
                attempts=sorted(case_dir.glob('attempt_*'));attempt=case_dir/f'attempt_{len(attempts)+1:04d}';attempt.mkdir()
                (attempt/'manifest.json').write_text(json.dumps({'case':case,'condition':condition,'seed':seed,'cohort_manifest_sha256':sha(out/'manifest.json'),'technical_attempt':len(attempts)+1},indent=2))
                if ep is None:ep=Episode(cfg,mode='closed_loop',perception_mode='learned',perturb='none' if condition=='nominal' else 'slot_move')
                frames=[];states=[];chosen=None;previous='HOME';context.clear()
                parent_log=recovery_evaluation_trace(ep,Episode._log.__get__(ep,Episode))
                def log(obs):
                    nonlocal previous,chosen
                    parent_log(obs);frame=frame_diagnostic(ep,obs,previous);previous=ep.fsm.state.name
                    frames.append(frame);s=ep.sim;states.append({'t':np.array(s.time),**{k:np.array(getattr(s.data,k),copy=True) for k in STATE_FIELDS}})
                    if chosen is None and eligibility(frame,condition):
                        visual=copy.deepcopy(ep.log.visual[-1]);chosen={'index':len(frames)-1,'frame':frame,'visual':visual,'observation':observation(frame,visual)}
                        if context:
                            np.savez_compressed(attempt/'checkpoint_sensor_packet.npz',**context['arrays'])
                            (attempt/'checkpoint_capture.json').write_text(json.dumps({'t':context['t'],'response':context['response'],'allowlist':sorted(context['arrays'])},allow_nan=False))
                ep._log=log;start=time.perf_counter()
                try:
                    ep.reset(seed)
                    while not ep.done:ep.step()
                    row={'condition':condition,'method':'v4','seed':seed,'config':cfg,'result':ep.result,'initial_poses':ep.poses,
                         'plan':ep.perturber.manifest(),'delivered':ep.perturber.delivery(),'transitions':ep.fsm.stats.transitions,
                         'visual_trace':ep.log.visual,'capture_trace':ep.perception._learned.history,'learned_perception':ep.perception.learned_summary(),
                         'placement_retries':ep.fsm.placement_retries,'acquisition_history':ep.fsm.acquisition_history,
                         'verification_history':ep.fsm.verification_history,'wall_seconds':time.perf_counter()-start}
                    (attempt/'episode.json').write_text(json.dumps(row,allow_nan=False))
                    (attempt/'frames.jsonl').write_text('\n'.join(json.dumps(f,allow_nan=False) for f in frames)+'\n')
                    np.savez_compressed(attempt/'control_states.npz',**{key:np.stack([state[key] for state in states]) for key in states[0]})
                    status='no_opportunity' if chosen is None else 'checkpoint_recorded'
                    if chosen is not None:(attempt/'checkpoint.json').write_text(json.dumps(chosen,indent=2,allow_nan=False))
                    complete={'case':case,'checkpoint_status':status,'control_frames':len(frames),'baseline_result':ep.result,
                              'files_sha256':{p.name:sha(p) for p in attempt.iterdir() if p.is_file()}}
                    (attempt/'attempt_complete.json').write_text(json.dumps(complete,indent=2,allow_nan=False))
                    (case_dir/'complete.json').write_text(json.dumps({'attempt':attempt.name,'attempt_complete_sha256':sha(attempt/'attempt_complete.json')},indent=2))
                    print(json.dumps({'case':case,'baseline_success':ep.result['success'],'checkpoint_status':status,'wall_seconds':row['wall_seconds']}),flush=True)
                except BaseException as exc:
                    (attempt/'technical_failure.json').write_text(json.dumps({'type':type(exc).__name__,'message':str(exc),'control_frames':len(frames)}));raise
    finally:
        if ep is not None:ep.perception.close()


def outcome(result):
    if result['collision_free_retained_lift']:return 'contact_free_retained_lift'
    if result['retained_lift']:return 'retained_with_environment_contact'
    return 'failed_lift'


def evaluate(out):
    manifest=check(out);protocol=manifest['protocol'];parent=json.loads((ROOT/'config/grasp_planner_protocol_v7.json').read_text())
    method_protocol=ablation_protocol();initial=json.loads((ROOT/'config/grasp_capture_protocol_v8.json').read_text())
    modes={name:(initial['modes']|method_protocol['modes'])[name] for name in protocol['methods']}
    retention=json.loads((ROOT/'config/grasp_feasibility_protocol_v5.json').read_text());results=[]
    for condition in protocol['conditions']:
        for seed in protocol['seeds']:
            case=f'{condition}__v4__{seed}';case_dir=out/condition/case;pointer=json.loads((case_dir/'complete.json').read_text())
            attempt=case_dir/pointer['attempt'];assert sha(attempt/'attempt_complete.json')==pointer['attempt_complete_sha256']
            completed=json.loads((attempt/'attempt_complete.json').read_text())
            for name,digest in completed['files_sha256'].items():assert sha(attempt/name)==digest,name
            evaluation=case_dir/'assay'
            if (evaluation/'summary.json').exists():results.append(json.loads((evaluation/'summary.json').read_text()));continue
            if evaluation.exists():raise RuntimeError('Partial assay: preserve it and explicitly audit before resuming '+case)
            evaluation.mkdir();episode=json.loads((attempt/'episode.json').read_text());checkpoint=None
            if (attempt/'checkpoint.json').exists():checkpoint=json.loads((attempt/'checkpoint.json').read_text())
            status='no_opportunity' if checkpoint is None else usability(checkpoint['frame'],checkpoint['observation'],parent)
            rows=[];reference=None;decisions={}
            if status is None:
                obs=checkpoint['observation'];cfg=episode['config'];sim=Sim(cfg);cad=extract_robot_cad(sim.model,calibration(sim));templates=robot_templates(sim)
                with np.load(attempt/'control_states.npz',allow_pickle=False) as states:
                    index=checkpoint['index'];arm=states['qpos'][index,sim.arm_qadr].copy()
                    for name,mode in modes.items():
                        world=planning_world(cad,cfg,obs,parent,mode);decision=select(sim,arm,obs,parent,method_protocol,mode,world,templates);decisions[name]=decision
                        (evaluation/(name+'_decision.json')).write_text(json.dumps(decision,indent=2,allow_nan=False))
                    for name,decision in decisions.items():
                        result=None
                        if decision['selected']:
                            result=physical_trial(sim,cfg,states,index,base_observation(obs),decision['selected'],parent,retention)
                            (evaluation/(name+'_physical.json')).write_text(json.dumps(result,allow_nan=False))
                        rows.append({'method':name,'score':'abstain' if result is None else outcome(result),'selected':decision['selected'],
                                     'physical':None if result is None else {k:v for k,v in result.items() if k!='trace'}})
                    action={'name':'recorded_visual_reference','position_m':obs['recorded_target_m'],'rotation':obs['recorded_rotation'],'approach_joint_width_m':.08}
                    ref=physical_trial(sim,cfg,states,index,base_observation(obs),action,parent,retention)
                    (evaluation/'recorded_reference_physical.json').write_text(json.dumps(ref,allow_nan=False));reference={'score':outcome(ref),'physical':{k:v for k,v in ref.items() if k!='trace'}}
            else:
                rows=[{'method':name,'score':'no_opportunity' if checkpoint is None else 'unassessable','selected':None,'physical':None} for name in modes]
            summary={'case':case,'condition':condition,'seed':seed,'baseline_result':episode['result'],'checkpoint_status':'usable' if status is None else status,
                     'rows':rows,'reference':reference,'source_attempt':str(attempt.relative_to(out)),
                     'episode_sha256':sha(attempt/'episode.json'),'cohort_manifest_sha256':sha(out/'manifest.json')}
            (evaluation/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False));results.append(summary)
            print(json.dumps({'case':case,'checkpoint':summary['checkpoint_status'],'scores':{r['method']:r['score'] for r in rows}}),flush=True)
    (out/'complete.json').write_text(json.dumps({'cases':len(results),'protocol_sha256':sha(PROTOCOL),'results':results},indent=2,allow_nan=False))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['init','collect','evaluate']);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--condition');p.add_argument('--seeds',nargs='+',type=int);args=p.parse_args();out=args.out.resolve()
    if args.command=='init':initialize(out)
    elif args.command=='collect':collect(out,args.condition,args.seeds)
    else:evaluate(out)


if __name__=='__main__':main()
