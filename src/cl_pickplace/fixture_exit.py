"""One physical extraction stroke planned from learned poses and nominal CAD."""
import numpy as np


def plan_exit(obs,task,settings):
    cap,slot=obs.est.capsule,obs.est.slot
    if not (cap.source.startswith('learned:') and slot.source.startswith('learned:')):
        return {'selected':None,'reason':'requires_learned_pair','candidates':[]}
    if not (cap.visible and slot.visible and cap.measurement_t is not None and slot.measurement_t is not None):
        return {'selected':None,'reason':'missing_visual_pair','candidates':[]}
    if max(obs.t-cap.measurement_t,obs.t-slot.measurement_t)>settings['maximum_age_s']:
        return {'selected':None,'reason':'stale_visual_pair','candidates':[]}
    if min(obs.t-cap.measurement_t,obs.t-slot.measurement_t)<0:
        return {'selected':None,'reason':'future_visual_pair','candidates':[]}
    if slot.observation.get('motion_pending',False):return {'selected':None,'reason':'fixture_motion_pending','candidates':[]}
    if not (np.isfinite(cap.pos).all() and np.isfinite(slot.pos).all() and np.isfinite(slot.yaw)):
        return {'selected':None,'reason':'nonfinite_visual_pair','candidates':[]}
    axis=np.array([np.cos(cap.yaw),np.sin(cap.yaw),0.]) if cap.axis is None else np.array(cap.axis,dtype=float,copy=True)
    if not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-9:return {'selected':None,'reason':'invalid_axis','candidates':[]}
    axis/=np.linalg.norm(axis);c,s=np.cos(slot.yaw),np.sin(slot.yaw)
    rotation=np.array([[c,-s],[s,c]]);offset=cap.pos[:2]-slot.pos[:2]
    half=np.asarray(task['placement_recovery']['cavity_m'])/2+settings['wall_thickness_m']
    local=offset@rotation
    if np.any(np.abs(local)>half+settings['near_fixture_slack_m']):return {'selected':None,'reason':'outside_extraction_region','candidates':[]}
    limits=np.asarray(task['visual_acquisition']['robot_search_bounds_xy_m']);plans=[]
    for face in range(2):
        for sign in (-1,1):
            direction=rotation[:,face]*sign
            extent=settings['capsule_half_segment_m']*abs(axis[:2]@direction)+settings['capsule_radius_m']
            distance=max(settings['minimum_stroke_m'],float(half[face]+extent+settings['pickup_clearance_m']-offset@direction))
            contact_offset=extent+settings['tool_forward_extent_m']
            start=cap.pos[:2]-direction*max(settings['approach_distance_m'],contact_offset+settings['approach_clearance_m'])
            capsule_goal=cap.pos[:2]+distance*direction
            finish=capsule_goal-contact_offset*direction
            feasible=(distance<=settings['maximum_stroke_m'] and all(np.all(p>=limits[:,0]+settings['workspace_margin_m']) and np.all(p<=limits[:,1]-settings['workspace_margin_m']) for p in (start,finish,capsule_goal)))
            height=max(slot.pos[2]+task['visual_acquisition']['nominal_wall_height_m']+
                task['visual_acquisition']['finger_tip_below_tcp_m']+task['visual_acquisition']['rim_clearance_m'],
                cap.pos[2]+settings['capsule_radius_m']+task['visual_acquisition']['finger_tip_below_tcp_m']-settings['surface_overlap_m'])
            plans.append({'face':face,'sign':sign,'feasible':bool(feasible),'stroke_m':distance,
                'plan':{'start':np.r_[start,height].tolist(),'finish':np.r_[finish,height].tolist(),
                'clear_z':max(float(obs.ee_pos[2]),settings['safe_height_m']),
                'fixture_yaw':float(slot.yaw),'fixture_floor_z':float(slot.pos[2]),
                'yaw':float(np.arctan2(direction[1],direction[0])),'capsule_position_m':cap.pos.tolist(),
                'fixture_position_m':slot.pos.tolist(),'observation_t':min(cap.measurement_t,slot.measurement_t)}})
    eligible=sorted((p for p in plans if p['feasible']),key=lambda p:(p['stroke_m'],p['face'],p['sign']))
    return {'selected':None if not eligible else eligible[0]['plan'],'reason':'shortest_feasible_exit_face' if eligible else 'no_workspace_exit','candidates':plans}
