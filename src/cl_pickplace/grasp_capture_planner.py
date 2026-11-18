"""Opt-in v8 geometry/capture experiment; the frozen v7 code remains unchanged."""
import numpy as np
from .grasp_planner import (FIELDS,validate_observation,candidates,fixture_boxes,
                           clearance_at,interpolate_rotation,waypoints,rotation)

EXTRA={'recorded_target_m','recorded_rotation'}


def base_observation(obs):return {key:obs[key] for key in FIELDS}


def registered_candidates(obs,parent):
    base=base_observation(obs);actions=candidates(base,parent)
    cap=np.asarray(obs['capsule_position_m']);p=np.asarray(obs['recorded_target_m']);r=np.asarray(obs['recorded_rotation'])
    offset=float(np.linalg.norm(p[:2]-cap[:2]));height=float(p[2]-cap[2]);opening=.08
    turn=float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(obs['ee_rotation']).T@r)-1)/2,-1,1))))
    score=abs(height-.002)/.004+offset/.008+.1*turn/180+.15*(opening-.024)/.016
    actions.append({'name':'recorded_visual_reference','position_m':p.tolist(),'rotation':r.tolist(),
                    'approach_joint_width_m':opening,'height_offset_m':height,'axial_offset_m':offset,
                    'yaw_offset_deg':0,'tilt':'recorded','symmetry_deg':None,'rank_cost':score})
    return sorted(actions,key=lambda action:(action['rank_cost'],action['name']))


def capsule_scenarios(obs,protocol,perturb):
    center=np.asarray(obs['capsule_position_m']);axis=np.asarray(obs['capsule_axis_world']);axis=axis/np.linalg.norm(axis)
    poses=[('center',center,axis)]
    if perturb:
        for i,world_axis in enumerate(np.eye(3)):
            for sign in (-1,1):
                poses.append((f'position_{i}_{sign}',center+world_axis*sign*protocol['capsule_position_perturbation_m'],axis))
                poses.append((f'axis_{i}_{sign}',center,rotation(world_axis,np.radians(sign*protocol['capsule_axis_perturbation_deg']))@axis))
    return poses


def capture_check(world,obs,action,protocol,perturb=False):
    results=[]
    for name,center,axis in capsule_scenarios(obs,protocol,perturb):
        world.set_capsule(center,axis)
        result=world.capture(action['position_m'],np.asarray(action['rotation']),action['approach_joint_width_m'],protocol['capture'])
        results.append({'scenario':name,**result})
        if not result['geometrically_capturable']:break
    return {'passed':all(row['geometrically_capturable'] for row in results),'tested_scenarios':len(results),'results':results}


def screen(obs,action,query,parent,minimum):
    lowest=float('inf')
    def check(p,r,w):
        nonlocal lowest
        gap=float(query(p,r,w))
        if not np.isfinite(gap):raise ValueError('Nonfinite clearance; stop selection')
        lowest=min(lowest,gap)
        return gap>=minimum
    r=np.asarray(action['rotation']);p=np.asarray(action['position_m']);opening=action['approach_joint_width_m']
    for width in np.linspace(0,opening,1+int(np.ceil(opening/parent['width_step_m']))):
        if not check(p,r,width):return {'feasible':False,'reason':'grasp_or_closure_clearance','minimum_reported_gap_m':lowest}
    p=np.asarray(obs['ee_position_m']);r=np.asarray(obs['ee_rotation']);width=obs['finger_width_m']
    for phase,target,rot,w in waypoints(base_observation(obs),action,parent):
        angle=np.degrees(np.arccos(np.clip((np.trace(r.T@rot)-1)/2,-1,1)))
        count=max(1,int(np.ceil(np.linalg.norm(target-p)/parent['translation_step_m'])),int(np.ceil(angle/parent['rotation_step_deg'])),int(np.ceil(abs(w-width)/parent['width_step_m'])))
        for fraction in np.linspace(0,1,count+1):
            if not check(p+(target-p)*fraction,interpolate_rotation(r,rot,fraction),width+(w-width)*fraction):
                return {'feasible':False,'reason':phase+'_clearance','minimum_reported_gap_m':lowest}
        p,r,width=target,rot,w
    return {'feasible':True,'reason':'sampled_path_clear','minimum_reported_gap_m':lowest}


def plan(obs,templates,fixture_cad,world,parent,protocol,mode):
    if set(obs)!=FIELDS|EXTRA:
        return {'status':'abstain','reason':'invalid_sensor_interface','ranked':[],'rejections':{},'candidate_count':0}
    reason=validate_observation(base_observation(obs),parent)
    if reason:return {'status':'abstain','reason':reason,'ranked':[],'rejections':{},'candidate_count':0}
    if (not np.isfinite(obs['recorded_target_m']).all() or not np.isfinite(obs['recorded_rotation']).all()):
        return {'status':'abstain','reason':'nonfinite_recorded_visual_target','ranked':[],'rejections':{},'candidate_count':0}
    if mode['representation']=='obb':
        obstacles=fixture_boxes(np.asarray(obs['fixture_position_m']),obs['fixture_yaw_rad'],fixture_cad)
        query=lambda p,r,w:clearance_at(templates,(p,r),w,obstacles,parent['table_z_m'],0.)
    else:query=world.environment_clearance
    ranked=[];rejections={};actions=registered_candidates(obs,parent)
    assert len(actions)==protocol['candidate_count']
    for action in actions:
        result=screen(obs,action,query,parent,protocol['minimum_clearance_m'])
        if not result['feasible']:
            rejections[result['reason']]=rejections.get(result['reason'],0)+1;continue
        cap=None
        if mode['capture']:
            cap=capture_check(world,obs,action,protocol,mode['capsule_perturbations'])
            if not cap['passed']:
                rejections['static_capture_interception']=rejections.get('static_capture_interception',0)+1;continue
        ranked.append({**action,'clearance_v8':result,'capture_v8':cap})
        if len(ranked)>=parent['maximum_kinematic_candidates']:break
    return {'status':'candidates' if ranked else 'abstain','reason':'requires_kinematic_check' if ranked else 'no_admissible_candidate',
            'candidate_count':len(actions),'screened_count':len(ranked)+sum(rejections.values()),'ranked':ranked,'rejections':rejections}
