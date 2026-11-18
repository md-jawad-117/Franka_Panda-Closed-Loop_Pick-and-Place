"""Physical counterfactual grasp assay; planning uses recorded visual estimates only."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cl_pickplace.sim import Sim
from cl_pickplace.controller import TaskSpaceController
from cl_pickplace.kinematics import orientation_error
from recovery_diagnostics_v5 import contact_diagnostics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def action_candidates(observation,protocol):
    """Explicit visual-only interface; truth stays in initialization/evaluation."""
    cap=np.asarray(observation['capsule_position_m']);axis=np.asarray(observation['capsule_axis_world'])[:2]
    axis=axis/np.linalg.norm(axis);q=np.asarray(observation['gripper_quaternion'])
    actions=[{'name':'recorded_target','position_m':observation['recorded_target_m'],'quaternion':q.tolist()}]
    for height in protocol['candidates']['center_height_offsets_m']:
        for axial in protocol['candidates']['axial_offsets_m']:
            position=cap+np.r_[axial*axis,height]
            actions.append({'name':f'h{height*1000:02.0f}_a{axial*1000:+03.0f}',
                            'position_m':position.tolist(),'quaternion':q.tolist()})
    return actions


def select_checkpoint(frames,nominal=False):
    closures=[i for i,f in enumerate(frames) if f['state_before']=='GRASP' and f['gripper_command']==0
              and (nominal or f['placement_retries']>0)]
    if not closures:raise ValueError('No actual closure in selected source')
    close=closures[0]
    entries=[i for i in range(close) if frames[i]['state_after']=='DESCEND_OBJECT' and frames[i]['state_before']!='DESCEND_OBJECT']
    if not entries:raise ValueError('No preceding descent')
    return next(i for i in range(entries[-1]+1,close) if frames[i]['state_before']=='DESCEND_OBJECT' and frames[i]['state_after']=='DESCEND_OBJECT')


def restore_checkpoint(sim,states,index):
    mujoco.mj_resetData(sim.model,sim.data)
    for name in ('qpos','qvel','ctrl','act','qacc_warmstart','mocap_pos','mocap_quat'):
        getattr(sim.data,name)[:]=states[name][index]
    sim.data.time=float(states['t'][index]);mujoco.mj_forward(sim.model,sim.data)


def trial(sim,cfg,states,index,action,observed_fixture,protocol):
    restore_checkpoint(sim,states,index);controller=TaskSpaceController(sim,cfg)
    controller.q_target=sim.data.ctrl[sim.arm_act].copy()
    grasp=np.asarray(action['position_m']);q=np.asarray(action['quaternion']);start_t=sim.time
    cap0=sim.capsule_pos().copy();lift_z=float(grasp[2]+protocol['lift_above_grasp_m'])
    phases=[('preposition',grasp+[0,0,protocol['approach_height_m']],cfg['gripper']['open_ctrl']),
            ('descend',grasp,cfg['gripper']['open_ctrl']),('close',grasp,cfg['gripper']['closed_ctrl']),
            ('lift',np.r_[grasp[:2],lift_z],cfg['gripper']['closed_ctrl']),
            ('transport',np.r_[np.asarray(observed_fixture)[:2],lift_z],cfg['gripper']['closed_ctrl']),
            ('hold',np.r_[np.asarray(observed_fixture)[:2],lift_z],cfg['gripper']['closed_ctrl'])]
    records=[];reason='completed';phase_results=[]
    for phase,target,gripper in phases:
        phase_start=sim.time;arrived=False;limit=protocol['phase_deadlines_s'][phase]
        while sim.time-phase_start<limit-1e-9:
            controller.step(target,q);sim.set_gripper(gripper);sim.step(sim.n_substeps)
            # No perception/controller uses this explicit evaluation forward.
            mujoco.mj_forward(sim.model,sim.data)
            contacts=contact_diagnostics(sim);position_error=float(np.linalg.norm(target-sim.ee_pos()))
            rotation_error=float(np.degrees(np.linalg.norm(orientation_error(q,sim.ee_mat()))))
            cap=sim.capsule_pos();finger_contacts=contacts['finger_capsule']
            bodies=set()
            for c in sim.data.contact:
                if c.dist<=0 and sim.capsule_geom in (c.geom1,c.geom2):
                    other=int(c.geom2 if c.geom1==sim.capsule_geom else c.geom1);bodies.add(int(sim.model.geom_bodyid[other]))
            bilateral=all(sim.link_bodies[n] in bodies for n in ('left_finger','right_finger'))
            penetrating=contacts['finger_fixture']+contacts['hand_fixture']
            maximum_penetration=max((c['penetration_m'] for c in penetrating),default=0.)
            rise=float(cap[2]-cap0[2]);distance=float(np.linalg.norm(cap-sim.ee_pos()))
            record={'t':sim.time-start_t,'phase':phase,'capsule_position_m':cap.tolist(),'ee_position_m':sim.ee_pos().tolist(),
                    'rise_m':rise,'capsule_tcp_distance_m':distance,'finger_width_m':sim.finger_width(),
                    'bilateral_contact':bilateral,'finger_fixture_contact':bool(contacts['finger_fixture']),
                    'hand_fixture_contact':bool(contacts['hand_fixture']),'maximum_fixture_penetration_m':maximum_penetration,
                    'position_error_m':position_error,'rotation_error_deg':rotation_error,
                    'finger_capsule_normal_force_N':sum(c['normal_force_N'] for c in finger_contacts),
                    'finger_fixture_normal_force_N':sum(c['normal_force_N'] for c in contacts['finger_fixture'])}
            records.append(record)
            if maximum_penetration>protocol['maximum_finger_fixture_penetration_m']:
                reason='collision_penetration_limit';break
            arrived=position_error<protocol['position_tolerance_m'] and rotation_error<protocol['rotation_tolerance_deg']
            if phase not in ('close','hold') and arrived:break
        phase_results.append({'phase':phase,'arrived':arrived,'duration_s':sim.time-phase_start})
        if reason!='completed':break
        if phase not in ('close','hold') and not arrived:
            reason=phase+'_tracking_timeout';break
    window=[r for r in records if r['t']>=records[-1]['t']-protocol['retention_window_s']-1e-9]
    retention=(reason=='completed' and len(window)>=2 and window[-1]['t']-window[0]['t']>=protocol['retention_window_s']-1e-8
               and all(r['rise_m']>=protocol['minimum_capsule_rise_m'] and r['capsule_tcp_distance_m']<=protocol['maximum_capsule_tcp_distance_m']
                       and r['bilateral_contact'] and cfg['gripper']['min_grasp_width']<=r['finger_width_m']<=cfg['gripper']['max_grasp_width'] for r in window))
    fixture_ticks=sum(r['finger_fixture_contact'] or r['hand_fixture_contact'] for r in records)
    return {'action':action,'termination':reason,'phase_results':phase_results,'max_rise_mm':1000*max(r['rise_m'] for r in records),
            'retained_lift':bool(retention),'fixture_contact_ticks':fixture_ticks,
            'collision_free_retained_lift':bool(retention and fixture_ticks==0),
            'max_fixture_penetration_mm':1000*max(r['maximum_fixture_penetration_m'] for r in records),'trace':records}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();protocol_path=ROOT/'config/grasp_feasibility_protocol_v5.json';protocol=json.loads(protocol_path.read_text())
    assert a.case in protocol['cases'];source=ROOT/'training/results/recovery-diagnostics-v5'/a.case
    done=json.loads((source/'complete.json').read_text());assert all(done['exact_original_replay'].values())
    original=json.loads((ROOT/json.loads((source/'manifest.json').read_text())['original_episode_path']).read_text())
    frames=[json.loads(line) for line in (source/'evaluation_only/frames.jsonl').read_text().splitlines()]
    index=select_checkpoint(frames,nominal=a.case.startswith('nominal'));frame=frames[index]
    assert frame['capsule_visible'] and frame['capsule_age_s']<=original['config']['perception']['learned']['max_age_s']+1e-8
    observation={'capsule_position_m':frame['capsule_estimate_m'],'capsule_axis_world':frame['capsule_estimated_axis'],
                 'recorded_target_m':frame['command_position_m'],'gripper_quaternion':frame['command_quaternion'],
                 'fixture_position_m':original['visual_trace'][index]['slot_estimate']}
    if a.out.exists():raise FileExistsError(a.out)
    a.out.mkdir(parents=True)
    manifest={'case':a.case,'checkpoint_index':index,'source_time_s':frame['t'],'planning_observation':observation,'protocol':protocol,
              'source_sha256':{str(path.relative_to(ROOT)):sha(path) for path in (source/'complete.json',source/'evaluation_only/control_states.npz',source/'evaluation_only/frames.jsonl',protocol_path,Path(__file__),ROOT/'scripts/recovery_diagnostics_v5.py')},
              'scope':'Selected checkpoint mechanics only. Frozen visual targets; not a deployable stale-vision policy or final placement performance.'}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    sim=Sim(original['config']);rows=[]
    with np.load(source/'evaluation_only/control_states.npz',allow_pickle=False) as states:
        for action in action_candidates(observation,protocol):
            result=trial(sim,original['config'],states,index,action,observation['fixture_position_m'],protocol)
            (a.out/(action['name']+'.json')).write_text(json.dumps(result,allow_nan=False))
            rows.append({k:v for k,v in result.items() if k!='trace'})
            print(json.dumps({'case':a.case,**rows[-1]}),flush=True)
    (a.out/'summary.json').write_text(json.dumps({'case':a.case,'trials':rows},indent=2))


if __name__=='__main__':main()
