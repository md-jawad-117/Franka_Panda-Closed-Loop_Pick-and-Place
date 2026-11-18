"""Short inward push planned from a learned capsule/fixture pair and robot CAD."""
import numpy as np


def plan_centering(obs,task,settings):
    cap,slot=obs.est.capsule,obs.est.slot
    def reject(reason):return {'selected':None,'reason':reason}
    if not (cap.source.startswith('learned:') and slot.source.startswith('learned:') and cap.visible and slot.visible
            and cap.measurement_t is not None and slot.measurement_t is not None):return reject('missing_learned_pair')
    ages=[obs.t-cap.measurement_t,obs.t-slot.measurement_t]
    if min(ages)<-1e-9 or max(ages)>settings['maximum_age_s']:return reject('stale_or_future_pair')
    if slot.observation.get('motion_pending',False):return reject('fixture_motion_pending')
    if not (np.isfinite(cap.pos).all() and np.isfinite(slot.pos).all() and np.isfinite(slot.yaw)):return reject('nonfinite_pair')
    axis=np.array([np.cos(cap.yaw),np.sin(cap.yaw),0.]) if cap.axis is None else np.array(cap.axis,float,copy=True)
    if not np.isfinite(axis).all() or np.linalg.norm(axis)<1e-9:return reject('invalid_axis')
    axis/=np.linalg.norm(axis)
    if abs(axis[2])>np.sin(np.radians(settings['maximum_capsule_tilt_deg'])):return reject('capsule_not_flat')
    c,s=np.cos(slot.yaw),np.sin(slot.yaw);R=np.array([[c,s],[-s,c]])
    local=R@(cap.pos[:2]-slot.pos[:2]);half=np.asarray(task['placement_recovery']['cavity_m'])/2
    extent=settings['capsule_radius_m']+settings['capsule_half_segment_m']*np.abs(R@axis[:2])
    if np.min(half-extent)<settings['centred_clearance_m']:return reject('orientation_cannot_fit_centred')
    if np.any(np.abs(local)>half+settings['near_fixture_slack_m']):return reject('outside_centering_region')
    if abs(cap.pos[2]-slot.pos[2]-settings['capsule_radius_m'])>settings['height_tolerance_m']:return reject('capsule_not_near_support')
    delta=slot.pos[:2]-cap.pos[:2];distance=float(np.linalg.norm(delta))
    if not settings['minimum_correction_m']<=distance<=settings['maximum_correction_m']:return reject('correction_distance_outside_band')
    direction=delta/distance
    contact=settings['capsule_half_segment_m']*abs(axis[:2]@direction)+settings['capsule_radius_m']+settings['tool_forward_extent_m']
    start=cap.pos[:2]-(contact+settings['approach_clearance_m'])*direction
    finish=slot.pos[:2]-contact*direction
    limits=np.asarray(task['visual_acquisition']['robot_search_bounds_xy_m'])
    if not all(np.all(p>=limits[:,0]+settings['workspace_margin_m']) and np.all(p<=limits[:,1]-settings['workspace_margin_m']) for p in (start,finish)):
        return reject('pusher_waypoint_outside_workspace')
    height=max(slot.pos[2]+task['visual_acquisition']['nominal_wall_height_m']+task['visual_acquisition']['finger_tip_below_tcp_m']+task['visual_acquisition']['rim_clearance_m'],
        cap.pos[2]+settings['capsule_radius_m']+task['visual_acquisition']['finger_tip_below_tcp_m']-settings['surface_overlap_m'])
    return {'selected':{'start':np.r_[start,height].tolist(),'finish':np.r_[finish,height].tolist(),
        'goal_capsule_m':slot.pos.tolist(),'initial_capsule_m':cap.pos.tolist(),'initial_distance_m':distance,
        'fixture_yaw_rad':float(slot.yaw),'clear_z':max(float(obs.ee_pos[2]),float(slot.pos[2]+settings['safe_clearance_m'])),
        'yaw_rad':float(np.arctan2(direction[1],direction[0])),'observation_t':min(cap.measurement_t,slot.measurement_t)},'reason':'short_inward_centre_correction'}
