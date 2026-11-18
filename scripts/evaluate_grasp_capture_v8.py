"""Matched collision/capture modes on selected states, with visual-only selection."""
from pathlib import Path
import argparse,json,sys,time
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cl_pickplace.sim import Sim
from cl_pickplace.kinematics import quat_to_mat
from cl_pickplace.grasp_collision import extract_robot_cad,GraspCollisionWorld
from cl_pickplace.grasp_capture_planner import plan,base_observation,capture_check
from evaluate_grasp_planner_v7 import load_case,sha,robot_templates,kinematic_check,physical_trial
from probe_grasp_feasibility_v5 import trial as original_trial


def calibration(sim):
    return {'arm_qadr':sim.arm_qadr,'finger_qadr':sim.finger_qadr,'home_q':sim.home_q,
            'ee_site':sim.ee_site,'hand_body':sim.link_bodies['hand'],
            'left_body':sim.link_bodies['left_finger'],'right_body':sim.link_bodies['right_finger']}


def prepare(case):
    source,old,original,obs=load_case(case)
    obs={**obs,'recorded_target_m':old['planning_observation']['recorded_target_m'],
         'recorded_rotation':quat_to_mat(np.array(old['planning_observation']['gripper_quaternion'])).tolist()}
    parent=json.loads((ROOT/'config/grasp_planner_protocol_v7.json').read_text())
    protocol=json.loads((ROOT/'config/grasp_capture_protocol_v8.json').read_text())
    return source,old,original,obs,parent,protocol


def planning_world(cad,cfg,obs,parent,mode):
    s=cfg['scene']['slot'];radius=np.linalg.norm([s['cavity'][0]/2+s['wall_thickness'],s['cavity'][1]/2+s['wall_thickness'],s['wall_height']])
    inflation=mode['fixture_position_bound_m']+2*radius*np.sin(np.radians(mode['fixture_yaw_bound_deg'])/2)
    return GraspCollisionWorld(cad,s,parent['table_z_m'],parent['capsule_prior'],
                               {'position_m':obs['fixture_position_m'],'yaw_rad':obs['fixture_yaw_rad']},inflation)


def select(sim,arm,obs,parent,protocol,mode,world,templates):
    start=time.perf_counter();decision=plan(obs,templates,sim.cfg['scene']['slot'],world,parent,protocol,mode)
    chosen=None;checks=[]
    for action in decision['ranked']:
        result=kinematic_check(sim,arm,base_observation(obs),action,parent);checks.append({'action':action['name'],**result})
        if result['feasible']:chosen=action;break
    decision.update(selected=chosen,kinematic_checks=checks,status='execute_assay' if chosen else 'abstain',planning_wall_seconds=time.perf_counter()-start)
    if chosen:decision['reason']='geometry_and_waypoint_ik_admitted'
    elif decision['ranked']:decision['reason']='ranked_kinematic_candidates_exhausted'
    return decision


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',required=True);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();source,old,original,obs,parent,protocol=prepare(args.case);assert args.case in protocol['cases']
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True);cfg=original['config'];sim=Sim(cfg);cad=extract_robot_cad(sim.model,calibration(sim));templates=robot_templates(sim)
    files=['config/grasp_capture_protocol_v8.json','config/grasp_planner_protocol_v7.json','src/cl_pickplace/grasp_collision.py',
           'src/cl_pickplace/grasp_capture_planner.py','src/cl_pickplace/grasp_planner.py','scripts/evaluate_grasp_capture_v8.py',
           'scripts/evaluate_grasp_planner_v7.py','scripts/probe_grasp_feasibility_v5.py']
    files+=[str((source/name).relative_to(ROOT)) for name in ('complete.json','evaluation_only/control_states.npz','evaluation_only/frames.jsonl')]
    manifest={'case':args.case,'observation':obs,'checkpoint_index':old['checkpoint_index'],'protocol':protocol,'parent_protocol':parent,
              'robot_cad_sha256':__import__('hashlib').sha256(json.dumps(cad,sort_keys=True).encode()).hexdigest(),
              'source_sha256':{rel:sha(ROOT/rel) for rel in files},'frozen_v4_sha256':json.loads((source/'manifest.json').read_text())['frozen_v4_source_sha256'],
              'boundary':'No true target pose/contact participates in selection; the separate CAD world contains only explicit learned target poses and fixed geometry.'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False));decisions={};rows=[]
    with np.load(source/'evaluation_only/control_states.npz',allow_pickle=False) as states:
        arm=states['qpos'][old['checkpoint_index'],sim.arm_qadr].copy()
        # Save every mode's choice BEFORE any physical action is evaluated.
        for name,mode in protocol['modes'].items():
            world=planning_world(cad,cfg,obs,parent,mode)
            decision=select(sim,arm,obs,parent,protocol,mode,world,templates);decisions[name]=decision
            (args.out/(name+'_decision.json')).write_text(json.dumps(decision,indent=2,allow_nan=False))
            print(json.dumps({'case':args.case,'mode':name,'status':decision['status'],'selected':None if decision['selected'] is None else decision['selected']['name'],
                              'screened':decision['screened_count'],'wall_s':decision['planning_wall_seconds']}),flush=True)
        for name,decision in decisions.items():
            physical=None;capture=None
            if decision['selected']:
                world=planning_world(cad,cfg,obs,parent,protocol['modes']['convex_point_capture'])
                capture=capture_check(world,obs,decision['selected'],protocol,False)
                physical=physical_trial(sim,cfg,states,old['checkpoint_index'],base_observation(obs),decision['selected'],parent,old['protocol'])
                (args.out/(name+'_physical.json')).write_text(json.dumps(physical,allow_nan=False))
            rows.append({'mode':name,'status':decision['status'],'selected':decision['selected'],
                         'capture_diagnostic':capture,'physical':None if physical is None else {k:v for k,v in physical.items() if k!='trace'}})
            print(json.dumps({'case':args.case,'mode':name,'physical':None if physical is None else {k:physical[k] for k in ('retained_lift','collision_free_retained_lift','max_rise_mm','fixture_contact_ticks','finger_table_contact_ticks')}}),flush=True)
        reference={'name':'recorded_visual_reference','position_m':obs['recorded_target_m'],'rotation':obs['recorded_rotation'],'approach_joint_width_m':.08}
        common=physical_trial(sim,cfg,states,old['checkpoint_index'],base_observation(obs),reference,parent,old['protocol'])
        (args.out/'recorded_reference_common_physical.json').write_text(json.dumps(common,allow_nan=False))
        original_action={'name':'recorded_target','position_m':obs['recorded_target_m'],'quaternion':old['planning_observation']['gripper_quaternion']}
        control=original_trial(sim,cfg,states,old['checkpoint_index'],original_action,obs['fixture_position_m'],old['protocol'])
        previous=json.loads((ROOT/'training/results/recovery-diagnostics-v5/feasibility'/args.case/'recorded_target.json').read_text());assert control==previous
    (args.out/'summary.json').write_text(json.dumps({'case':args.case,'rows':rows,'original_reference_exact_replay':True,
        'common_reference':{k:v for k,v in common.items() if k!='trace'},'original_reference':{k:v for k,v in control.items() if k!='trace'}},indent=2,allow_nan=False))


if __name__=='__main__':main()
