"""Experimental CAD/learned-pose grasp screening, independent of simulator state.

Boxes conservatively enclose collision meshes. Clearance is checked on sampled
paths, with declared uncertainty margins; it is not a continuous collision proof.
No simulator, contacts, labels or target truth are accepted by this module.
"""
from __future__ import annotations

import itertools
import numpy as np


FIELDS = {'t', 'capsule_t', 'fixture_t', 'capsule_visible', 'fixture_visible',
          'motion_pending', 'capsule_position_m', 'capsule_axis_world',
          'fixture_position_m', 'fixture_yaw_rad', 'ee_position_m',
          'ee_rotation', 'finger_width_m'}


def rotation(axis, angle):
    axis = np.asarray(axis, float);length=np.linalg.norm(axis)
    if length<1e-12:
        if abs(angle)<1e-12:return np.eye(3)
        raise ValueError('Undefined rotation axis')
    axis = axis / length
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle)*skew + (1-np.cos(angle))*(skew@skew)


def interpolate_rotation(a, b, fraction):
    if np.allclose(a,b,atol=1e-10,rtol=0):return a.copy()
    delta = a.T @ b
    skew_axis=np.array([delta[2,1]-delta[1,2],delta[0,2]-delta[2,0],delta[1,0]-delta[0,1]])/2
    theta = np.arctan2(np.linalg.norm(skew_axis),np.clip((np.trace(delta)-1)/2,-1,1))
    if theta < 1e-8:
        return a.copy()
    # Eigenvector also handles rotations arbitrarily close to pi.
    if abs(theta-np.pi) < 1e-5:
        values, vectors = np.linalg.eigh((delta+delta.T)/2)
        axis = vectors[:, np.argmax(values)]
    else:
        axis = skew_axis/np.linalg.norm(skew_axis)
    return a @ rotation(axis, theta*fraction)


def box_gaps(centers, axes, half_sizes, obstacle_centers, obstacle_axes, obstacle_half_sizes):
    """All-pairs OBB SAT gaps. Positive means separated, not Euclidean distance."""
    a = np.asarray(axes); b = np.asarray(obstacle_axes)
    na, nb = len(a), len(b)
    aa = np.broadcast_to(a[:, None, :, :], (na, nb, 3, 3)).swapaxes(-1,-2)
    bb = np.broadcast_to(b[None, :, :, :], (na, nb, 3, 3)).swapaxes(-1,-2)
    cross = np.cross(aa[:,:,:,None,:], bb[:,:,None,:,:]).reshape(na,nb,9,3)
    directions = np.concatenate((aa, bb, cross), axis=2)
    norm = np.linalg.norm(directions,axis=-1)
    directions = directions/np.maximum(norm[...,None],1e-12)
    delta = np.asarray(obstacle_centers)[None,:,:]-np.asarray(centers)[:,None,:]
    center_distance = np.abs(np.einsum('abkd,abd->abk',directions,delta))
    rad_a = np.einsum('abkj,aj->abk',np.abs(np.einsum('abkd,adj->abkj',directions,a)),half_sizes)
    rad_b = np.einsum('abkj,bj->abk',np.abs(np.einsum('abkd,bdj->abkj',directions,b)),obstacle_half_sizes)
    gap = np.where(norm>1e-8,center_distance-rad_a-rad_b,-np.inf)
    return gap.max(axis=2)


def fixture_boxes(position, yaw, cad, inflation=0.):
    cx, cy = np.asarray(cad['cavity'])/2
    t, h, bt = (cad[k] for k in ('wall_thickness','wall_height','bottom_thickness'))
    # Observed position is the cavity FLOOR center, not fixture body origin.
    local = np.array([[0,0,-bt/2],[cx+t/2,0,h/2],[-cx-t/2,0,h/2],
                      [0,cy+t/2,h/2],[0,-cy-t/2,h/2]])
    half = np.array([[cx+t,cy+t,bt/2],[t/2,cy+t,h/2],[t/2,cy+t,h/2],
                     [cx,t/2,h/2],[cx,t/2,h/2]])
    rot = rotation([0,0,1],yaw)
    return local@rot.T+position, np.repeat(rot[None,:,:],5,axis=0), half+inflation


def gripper_boxes(templates, position, rot, width):
    local = np.asarray(templates['centers']) + np.asarray(templates['width_directions'])*width/2
    centers = local@rot.T+position
    axes = np.einsum('ij,njk->nik',rot,np.asarray(templates['axes']))
    return centers,axes,np.asarray(templates['half_sizes'])


def validate_observation(obs, protocol):
    if set(obs) != FIELDS:
        return 'invalid_sensor_interface'
    if not obs['capsule_visible'] or not obs['fixture_visible'] or obs['motion_pending']:
        return 'unconfirmed_or_missing_observation'
    for key in ('capsule_t','fixture_t'):
        if obs[key] is None or not 0 <= obs['t']-obs[key] <= protocol['maximum_observation_age_s']+1e-8:
            return 'stale_observation'
    for key in ('capsule_position_m','capsule_axis_world','fixture_position_m','ee_position_m','ee_rotation'):
        if not np.isfinite(np.asarray(obs[key],float)).all():
            return 'nonfinite_observation'
    if np.linalg.norm(np.asarray(obs['capsule_axis_world'])[:2]) < 1e-6:
        return 'unobservable_horizontal_axis'
    if not np.isfinite(obs['fixture_yaw_rad']) or not np.isfinite(obs['finger_width_m']):
        return 'nonfinite_observation'
    r=np.asarray(obs['ee_rotation'])
    if r.shape != (3,3) or not np.allclose(r.T@r,np.eye(3),atol=1e-5) or np.linalg.det(r)<0:
        return 'invalid_proprioceptive_rotation'
    return None


def candidates(obs, protocol):
    cap=np.asarray(obs['capsule_position_m']);a=np.asarray(obs['capsule_axis_world'])[:2]
    a=a/np.linalg.norm(a);yaw=np.arctan2(a[1],a[0]);rows=[]
    tilt_options=[('vertical',0,0),('roll_negative',-1,0),('roll_positive',1,0),
                  ('pitch_negative',0,-1),('pitch_positive',0,1)]
    grid=protocol['candidates']
    for symmetry,yaw_offset,tilt,width,height,offset in itertools.product(
            (0,180),grid['yaw_offsets_deg'],tilt_options,grid['joint_widths_m'],
            grid['center_height_offsets_m'],grid['axial_offsets_m']):
        angle=yaw+np.radians(symmetry+yaw_offset)
        x=np.array([np.cos(angle),np.sin(angle),0.]);z=np.array([0.,0.,-1.]);y=np.cross(z,x)
        r=np.column_stack((x,y,z))
        r=r@rotation([1,0,0],np.radians(tilt[1]*grid['tilt_deg']))@rotation([0,1,0],np.radians(tilt[2]*grid['tilt_deg']))
        p=cap+np.r_[offset*a,height]
        delta=obs['ee_rotation'];turn=float(np.degrees(np.arccos(np.clip((np.trace(np.asarray(delta).T@r)-1)/2,-1,1))))
        score=(abs(height-.002)/.004+abs(offset)/.008+abs(yaw_offset)/30
               +(0 if tilt[0]=='vertical' else .7)+.1*turn/180+.15*(width-.024)/.016)
        rows.append({'name':f's{symmetry}_y{yaw_offset:+03d}_{tilt[0]}_w{width*1000:.0f}_h{height*1000:.0f}_a{offset*1000:+.0f}',
                     'position_m':p.tolist(),'rotation':r.tolist(),'approach_joint_width_m':width,
                     'height_offset_m':height,'axial_offset_m':offset,'yaw_offset_deg':yaw_offset,
                     'tilt':tilt[0],'symmetry_deg':symmetry,'rank_cost':score})
    return sorted(rows,key=lambda row:(row['rank_cost'],row['name']))


def waypoints(obs, action, protocol):
    r=np.asarray(action['rotation']);p=np.asarray(action['position_m']);width=action['approach_joint_width_m']
    initial=np.asarray(obs['ee_position_m']);initial_r=np.asarray(obs['ee_rotation'])
    above=p-r[:,2]*protocol['approach_height_m'];lift=p-r[:,2]*protocol['lift_above_grasp_m']
    high=initial.copy();high[2]=max(initial[2],above[2],protocol['retreat_height_m'])
    transport=np.r_[np.asarray(obs['fixture_position_m'])[:2],lift[2]]
    return [('retreat',high,initial_r,obs['finger_width_m']),('preshape',high,initial_r,width),
            ('orient',high,r,width),('preposition',above,r,width),('descend',p,r,width),
            ('close',p,r,0.),('lift',lift,r,0.),('transport',transport,r,0.),('hold',transport,r,0.)]


def clearance_at(templates, pose, width, obstacles, table_z, margin):
    p,r=pose;centers,axes,half=gripper_boxes(templates,p,r,width)
    # Tracking and finite sampling inflation is a design bound, not calibrated
    # learned uncertainty. Apply it to the gripper's local box dimensions.
    half=half+margin
    fixture_gap=float(box_gaps(centers,axes,half,*obstacles).min())
    bottoms=centers[:,2]-np.einsum('nj,nj->n',np.abs(axes[:,2,:]),half)
    if not np.isfinite(fixture_gap) or not np.isfinite(bottoms).all():
        raise ValueError('Nonfinite collision geometry; no action may be selected')
    return min(fixture_gap,float(bottoms.min()-table_z))


def screen_action(obs, action, templates, cad, protocol, uncertainty):
    radius=np.linalg.norm([cad['cavity'][0]/2+cad['wall_thickness'],
                           cad['cavity'][1]/2+cad['wall_thickness'],cad['wall_height']])
    inflation=uncertainty['fixture_position_bound_m']+2*radius*np.sin(np.radians(uncertainty['fixture_yaw_bound_deg'])/2)
    obstacles=fixture_boxes(np.asarray(obs['fixture_position_m']),obs['fixture_yaw_rad'],cad,inflation)
    margin=protocol['gripper_inflation_m']
    # First reject colliding end grasps cheaply, including finger closure.
    minimum=float('inf')
    for width in np.linspace(0,action['approach_joint_width_m'],1+int(np.ceil(action['approach_joint_width_m']/protocol['width_step_m']))):
        gap=clearance_at(templates,(action['position_m'],np.asarray(action['rotation'])),width,obstacles,protocol['table_z_m'],margin)
        minimum=min(minimum,gap)
        if gap<=0:return {'feasible':False,'reason':'grasp_or_closure_clearance','minimum_sat_gap_m':minimum}
    p=np.asarray(obs['ee_position_m']);r=np.asarray(obs['ee_rotation']);w=obs['finger_width_m']
    for phase,target,rot,width in waypoints(obs,action,protocol):
        angle=np.arccos(np.clip((np.trace(r.T@rot)-1)/2,-1,1))
        count=max(1,int(np.ceil(np.linalg.norm(target-p)/protocol['translation_step_m'])),
                  int(np.ceil(np.degrees(angle)/protocol['rotation_step_deg'])),int(np.ceil(abs(width-w)/protocol['width_step_m'])))
        for frac in np.linspace(0,1,count+1):
            gap=clearance_at(templates,(p+(target-p)*frac,interpolate_rotation(r,rot,frac)),w+(width-w)*frac,
                             obstacles,protocol['table_z_m'],margin)
            minimum=min(minimum,gap)
            if gap<=0:return {'feasible':False,'reason':phase+'_clearance','minimum_sat_gap_m':minimum}
        p,r,w=target,rot,width
    return {'feasible':True,'reason':'sampled_gripper_path_clear','minimum_sat_gap_m':minimum}


def plan(obs, templates, cad, protocol, uncertainty):
    reason=validate_observation(obs,protocol)
    if reason:return {'status':'abstain','reason':reason,'candidate_count':0,'ranked':[],'rejections':{}}
    all_actions=candidates(obs,protocol);ranked=[];rejections={}
    for action in all_actions:
        r=np.asarray(action['rotation']);local=r.T@(np.asarray(obs['capsule_position_m'])-action['position_m'])
        axis=r.T@np.asarray(obs['capsule_axis_world']);prior=protocol['capsule_prior']
        required_width=2*(prior['radius_m']+prior['half_length_m']*abs(axis[1])+abs(local[1]))+protocol['capture_slack_m']
        if (required_width>action['approach_joint_width_m'] or
                abs(local[0])>prior['half_length_m']+1e-9 or
                abs(local[2]-protocol['pad_center_z_from_tcp_m'])>prior['radius_m']+.0085-protocol['capture_slack_m']):
            rejections['nominal_capture_geometry']=rejections.get('nominal_capture_geometry',0)+1
            continue
        screen=screen_action(obs,action,templates,cad,protocol,uncertainty)
        if screen['feasible']:
            ranked.append({**action,'clearance':screen})
            if len(ranked)>=protocol['maximum_kinematic_candidates']:break
        else:rejections[screen['reason']]=rejections.get(screen['reason'],0)+1
    return {'status':'candidates' if ranked else 'abstain','reason':'requires_kinematic_check' if ranked else 'no_clear_gripper_path',
            'candidate_count':len(all_actions),'screened_count':len(ranked)+sum(rejections.values()),'ranked':ranked,'rejections':rejections}
