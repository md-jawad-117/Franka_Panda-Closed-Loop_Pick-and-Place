"""Matched, resumable learned-control comparison; evaluation truth is log-only.

Run with the simulation Python from the project root. No reserved dataset is read.
"""
from pathlib import Path
import argparse,copy,hashlib,json,sys,time
from collections import defaultdict
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from cl_pickplace.config import load_config
from cl_pickplace.episode import Episode
from cl_pickplace.kinematics import wrap_half_pi
from cl_pickplace.provenance import run_metadata


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configuration(protocol,controller,condition):
    cfg=load_config(ROOT/'config/learned_active.yaml')
    cfg['task']['placement_policy']='active_observe_v1' if controller=='active_observe_v1' else 'reactive_v1'
    cfg['task']['transport_wait_s']=protocol['fixed_wait_s'] if controller=='wait_reactive_v1' else 0.
    cfg['task']['episode_timeout_s']=protocol['timeout_s']
    cfg['perturbations']['schedule_mode']='time'
    move=cfg['perturbations']['scheduled']['slot_move']
    at=protocol['conditions'][condition]
    if at is not None:
        move.update(time_s_range=[at,at],shift_m_range=[protocol['shift_m']]*2,
                    yaw_change_deg_range=protocol['yaw_change_deg_range'],move_duration_s=protocol['move_duration_s'])
    return cfg


def evaluation_trace(ep,original_log):
    """Called AFTER policy/control; never returned to either controller or worker."""
    def log(obs):
        original_log(obs)
        slot=obs.est.slot;sim=ep.sim
        ep.log.visual[-1]['evaluation_only']={
            'slot_true_position_m':sim.slot_pos().tolist(),'slot_true_yaw_rad':sim.slot_yaw(),
            'slot_pose_error_mm':1000*float(np.linalg.norm(slot.pos-sim.slot_pos())),
            'slot_yaw_error_deg':float(abs(np.degrees(wrap_half_pi(slot.yaw-sim.slot_yaw())))),
            'capsule_true_position_m':sim.capsule_pos().tolist(),
            'gripper_command':ep.cmd.gripper}
        ep.log.visual[-1]['active_observation_retries']=ep.fsm.observation_attempts
        ep.log.visual[-1]['slot_estimated_yaw_rad']=slot.yaw
        target=(ep.fsm.frozen.slot if ep.mode=='open_loop' and ep.fsm.frozen is not None else slot)
        used=(target.pos,target.yaw,target.measurement_t)
        if ep.fsm.active_placement and ep.fsm.placement_target is not None:used=ep.fsm.placement_target
        ep.log.visual[-1]['placement_target']={'position_m':used[0].tolist(),'yaw_rad':used[1],'measurement_t':used[2]}
        ep.log.visual[-1]['slot_geometry_quality']=slot.quality
        ep.log.visual[-1]['observation_view_index']=ep.fsm.view_index
    return log


def diagnostics(trace,transitions,delivered,open_ctrl):
    released=[r for r in trace if r['state']=='RELEASE' and r['evaluation_only']['gripper_command']==open_ctrl]
    release=released[0] if released else None
    decisions=[r for r in trace if r['state']=='RELEASE']
    decision=decisions[0] if decisions else None
    out={'release_t_s':None,'release_slot_age_s':None,'release_slot_error_mm':None,
         'release_slot_yaw_error_deg':None,'release_guard':None,'reacquisition_after_motion_s':None}
    if release:
        # RELEASE opens on the next tick. Score the pose that actually committed
        # the previous descent, rather than a possibly new post-decision image.
        truth=release['evaluation_only']
        target=decision['placement_target']
        out.update(release_t_s=release['t'],release_slot_age_s=release['t']-target['measurement_t'],
                   release_slot_error_mm=1000*float(np.linalg.norm(np.array(target['position_m'])-truth['slot_true_position_m'])),
                   release_slot_yaw_error_deg=float(abs(np.degrees(wrap_half_pi(target['yaw_rad']-truth['slot_true_yaw_rad'])))),
                   release_guard=decision['guard'])
    if delivered:
        end=max(e['actual_end_s'] for e in delivered)
        correct=[r for r in trace if r['t']>=end and r['slot_visible']
                 and r['slot_measurement_t'] is not None and abs(r['t']-r['slot_measurement_t'])<1e-7
                 and r['evaluation_only']['slot_pose_error_mm']<=5
                 and r['evaluation_only']['slot_yaw_error_deg']<=10]
        if correct:out['reacquisition_after_motion_s']=correct[0]['t']-end
    out['descent_retreats']=sum(a=='DESCEND_SLOT' and b=='TRANSPORT' for _,a,b,_ in transitions)
    out['observation_guard_s']=.02*sum('observe' in r['guard'] or 'reobserve' in r['guard'] for r in trace)
    out['active_observation_retries']=max((r['active_observation_retries'] for r in trace),default=0)
    return out


def source_fingerprint():
    paths=list((ROOT/'src').rglob('*.py'))+[Path(__file__),ROOT/'config/learned_active.yaml']
    return {str(p.relative_to(ROOT)).replace('\\','/'):sha(p) for p in sorted(paths)}


def summarize(records):
    groups=defaultdict(list)
    for record in records:groups[(record['condition'],record['controller'])].append(record)
    output={}
    for (condition,controller),rows in groups.items():
        good=[r for r in rows if r['result']['success']]
        def avg(key):
            vals=[r['diagnostics'][key] for r in rows if r['diagnostics'][key] is not None]
            return float(np.mean(vals)) if vals else None
        output[condition+'/'+controller]={
            'n':len(rows),'successes':len(good),'success_rate':len(good)/len(rows),
            'timeouts':sum(r['result']['timeout'] for r in rows),
            'mean_completion_s_successful':float(np.mean([r['result']['time_to_complete_s'] for r in good])) if good else None,
            'mean_final_center_error_mm_all':1000*float(np.mean([r['result']['final_place_error_m'] for r in rows])),
            'mean_release_slot_error_mm':avg('release_slot_error_mm'),'mean_release_slot_age_s':avg('release_slot_age_s'),
            'mean_reacquisition_after_motion_s_among_reacquired':avg('reacquisition_after_motion_s'),
            'reacquired_count':sum(r['diagnostics']['reacquisition_after_motion_s'] is not None for r in rows),
            'descent_retreats':sum(r['diagnostics']['descent_retreats'] for r in rows),
            'mean_observation_guard_s':avg('observation_guard_s')}
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--cohort',choices=['development','comparison'],default='development')
    parser.add_argument('--controllers',nargs='+')
    parser.add_argument('--conditions',nargs='+')
    parser.add_argument('--seeds',nargs='+',type=int,help='diagnostic subset; recorded explicitly')
    args=parser.parse_args()
    protocol_path=ROOT/'config/moving_destination_protocol_v1.json'
    protocol=json.loads(protocol_path.read_text());source=source_fingerprint()
    controllers=args.controllers or protocol['controllers'];conditions=args.conditions or list(protocol['conditions'])
    assert set(controllers)<=set(protocol['controllers']) and set(conditions)<=set(protocol['conditions'])
    seeds=args.seeds or protocol[args.cohort+'_seeds']
    args.out.mkdir(parents=True,exist_ok=True)
    manifest={'protocol':protocol,'protocol_sha256':sha(protocol_path),'source_sha256':source,
              'cohort':args.cohort,'seeds':seeds,'controllers':controllers,'conditions':conditions}
    manifest_path=args.out/'manifest.json'
    if manifest_path.exists():
        assert json.loads(manifest_path.read_text())==manifest,'Cannot resume with different source/protocol/cohort'
    else:manifest_path.write_text(json.dumps(manifest,indent=2))
    (args.out/'episodes').mkdir(exist_ok=True)
    records=[]
    for condition in conditions:
        for controller in controllers:
            cfg=configuration(protocol,controller,condition)
            mode='open_loop' if controller=='open_loop' else 'closed_loop'
            ep=None
            try:
                for seed in seeds:
                    path=args.out/'episodes'/f'{condition}__{controller}__{seed}.json'
                    if path.exists():records.append(json.loads(path.read_text()));continue
                    if ep is None:
                        ep=Episode(cfg,mode=mode,perception_mode='learned',perturb='none' if condition=='nominal' else 'slot_move')
                        ep._log=evaluation_trace(ep,ep._log)
                    started=time.perf_counter();ep.reset(seed)
                    while not ep.done:ep.step()
                    record={'seed':seed,'controller':controller,'condition':condition,'mode':mode,
                            'initial_poses':ep.poses,'plan_id':ep.perturber.plan_id(),'plan':ep.perturber.manifest(),
                            'delivered':ep.perturber.delivery(),'result':ep.result,'transitions':ep.fsm.stats.transitions,
                            'visual_trace':ep.log.visual,'learned_capture_trace':ep.perception._learned.history,
                            'learned_perception':ep.perception.learned_summary(),
                            'wall_seconds':time.perf_counter()-started}
                    record['diagnostics']=diagnostics(record['visual_trace'],record['transitions'],record['delivered'],cfg['gripper']['open_ctrl'])
                    path.write_text(json.dumps(record,allow_nan=False));records.append(record)
                    print(json.dumps({k:record[k] for k in ['condition','controller','seed','result','diagnostics','wall_seconds']}),flush=True)
                    (args.out/'summary.json').write_text(json.dumps(summarize(records),indent=2))
            finally:
                if ep is not None:ep.perception.close()
    (args.out/'provenance.json').write_text(json.dumps(run_metadata(configuration(protocol,controllers[0],conditions[0]),'learned'),indent=2))
    print('Saved',args.out,flush=True)


if __name__=='__main__':main()
