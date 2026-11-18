"""Evaluation-only diagnostic readers. Never imported by a policy or worker."""
import itertools
import mujoco
import numpy as np
from cl_pickplace.kinematics import gripper_down_quat,orientation_error,wrap_half_pi
from cl_pickplace.dataset_geometry import transform

FIELDS={f'{field}_{camera}' for field in ('rgb','depth','valid','K','T') for camera in ('wrist_cam','top_cam')}


def axis_error_deg(predicted,truth):
    if predicted is None:return None
    a,b=np.asarray(predicted,float),np.asarray(truth,float)
    scale=np.linalg.norm(a)*np.linalg.norm(b)
    if scale<1e-12:return None
    return float(np.degrees(np.arccos(np.clip(abs(a@b)/scale,0,1))))


def contact_diagnostics(sim):
    fingers={sim.link_bodies[n] for n in ('left_finger','right_finger')}
    groups={key:[] for key in ('finger_capsule','finger_fixture','hand_fixture','capsule_fixture','finger_table')}
    for index,contact in enumerate(sim.data.contact):
        if contact.dist>0:continue
        a,b=int(contact.geom1),int(contact.geom2);ba,bb=int(sim.model.geom_bodyid[a]),int(sim.model.geom_bodyid[b])
        pair={a,b};bodies={ba,bb};keys=[]
        if sim.capsule_geom in pair and bodies&fingers:keys.append('finger_capsule')
        if sim.slot_body in bodies and bodies&fingers:keys.append('finger_fixture')
        if sim.slot_body in bodies and sim.link_bodies['hand'] in bodies:keys.append('hand_fixture')
        if sim.capsule_geom in pair and sim.slot_body in bodies:keys.append('capsule_fixture')
        names=[mujoco.mj_id2name(sim.model,mujoco.mjtObj.mjOBJ_GEOM,g) or f'geom_{g}' for g in (a,b)]
        if bodies&fingers and any('table' in name.lower() for name in names):keys.append('finger_table')
        if not keys:continue
        force=np.zeros(6);mujoco.mj_contactForce(sim.model,sim.data,index,force)
        record={'geom_names':names,'penetration_m':max(0.,-float(contact.dist)),
                'normal_force_N':float(force[0]),'position_m':contact.pos.tolist()}
        for key in keys:groups[key].append(record)
    return groups


def lowest_finger_collision_z(sim):
    fingers={sim.link_bodies[n] for n in ('left_finger','right_finger')};heights=[]
    for gid in range(sim.model.ngeom):
        if int(sim.model.geom_bodyid[gid]) not in fingers:continue
        if not (sim.model.geom_contype[gid] or sim.model.geom_conaffinity[gid]):continue
        kind=sim.model.geom_type[gid]
        if kind==mujoco.mjtGeom.mjGEOM_MESH:
            mid=int(sim.model.geom_dataid[gid]);start=int(sim.model.mesh_vertadr[mid]);count=int(sim.model.mesh_vertnum[mid])
            vertices=sim.model.mesh_vert[start:start+count]
        elif kind==mujoco.mjtGeom.mjGEOM_BOX:
            vertices=np.asarray(list(itertools.product((-1,1),repeat=3)))*sim.model.geom_size[gid]
        else:continue
        world=vertices@sim.data.geom_xmat[gid].reshape(3,3).T+sim.data.geom_xpos[gid]
        heights.append(float(world[:,2].min()))
    return min(heights) if heights else None


def frame_diagnostic(ep,obs,state_before):
    """Read physical truth AFTER policy/control, without stepping/forwarding physics."""
    s=ep.sim;cap=obs.est.capsule;settings=ep.cfg['perception']['learned']
    truth=s.data.geom_xpos[s.capsule_geom].copy();axis=s.data.geom_xmat[s.capsule_geom].reshape(3,3)[:,2].copy()
    yaw=float(np.arctan2(axis[1],axis[0]));age=None if cap.measurement_t is None else obs.t-cap.measurement_t
    rotation_error=float(np.degrees(np.linalg.norm(orientation_error(gripper_down_quat(cap.yaw),obs.ee_mat))))
    xy=float(np.linalg.norm(obs.ee_pos[:2]-cap.pos[:2]));clearance=float(obs.ee_pos[2]-cap.pos[2])
    floor=float(s.slot_pos()[2]);lowest=lowest_finger_collision_z(s)
    command_error=None if ep.cmd.p_des is None else float(np.linalg.norm(ep.cmd.p_des-obs.ee_pos))
    return {'t':obs.t,'state_before':state_before,'state_after':ep.fsm.state.name,'guard':ep.fsm.guard_action,
            'placement_retries':ep.fsm.placement_retries,'gripper_command':ep.cmd.gripper,
            'capsule_visible':cap.visible,'capsule_measurement_t':cap.measurement_t,'capsule_age_s':age,
            'fresh_capture':bool(cap.visible and age is not None and abs(age)<1e-7),
            'capsule_estimate_m':cap.pos.tolist(),'capsule_estimated_axis':None if cap.axis is None else cap.axis.tolist(),
            'ee_position_m':obs.ee_pos.tolist(),'ee_rotation':obs.ee_mat.tolist(),
            'command_position_m':None if ep.cmd.p_des is None else ep.cmd.p_des.tolist(),
            'command_quaternion':None if ep.cmd.q_des is None else ep.cmd.q_des.tolist(),
            'command_position_error_m':command_error,'finger_width_m':obs.finger_width,
            'grasp_commit_gates':{'age':age is not None and 0<=age<=settings['grasp_commit_max_age_s'],
                                  'xy':xy<=settings['grasp_commit_xy_m'],
                                  'clearance':0<=clearance<=settings['grasp_commit_clearance_m'],
                                  'rotation':rotation_error<=ep.cfg['task']['rot_tol_deg']},
            'grasp_alignment':{'xy_m':xy,'clearance_m':clearance,'rotation_error_deg':rotation_error},
            'evaluation_only':{'capsule_true_position_m':truth.tolist(),'capsule_true_axis':axis.tolist(),
                'capsule_center_error_mm':1000*float(np.linalg.norm(cap.pos-truth)),
                'capsule_axis_error_deg':axis_error_deg(cap.axis,axis),
                'capsule_yaw_error_deg':abs(float(np.degrees(wrap_half_pi(cap.yaw-yaw)))),
                'capsule_in_hand_frame_m':(obs.ee_mat.T@(truth-obs.ee_pos)).tolist(),
                'capsule_axis_in_hand_frame':(obs.ee_mat.T@axis).tolist(),
                'fixture_floor_z_m':floor,'lowest_finger_collision_z_m':lowest,
                'finger_to_rim_vertical_clearance_m':None if lowest is None else lowest-floor-ep.cfg['scene']['slot']['wall_height'],
                'capsule_velocity':s.data.qvel[s.capsule_dadr:s.capsule_dadr+6].tolist(),
                'contacts':contact_diagnostics(s)}}


def capture_labels_without_forward(sim,capture,arrays):
    """A separate matching renderer; no mj_forward, physics step, or random noise."""
    assert set(arrays)==FIELDS
    labels={};geometry=[]
    for camera in ('wrist_cam','top_cam'):
        capture._scene(camera);rgb=capture.renderer.render().copy()
        if not np.array_equal(rgb,arrays['rgb_'+camera]):raise AssertionError('Diagnostic renderer RGB differs from actual policy input')
        capture.renderer.enable_depth_rendering()
        try:
            capture._scene(camera);labels['ideal_depth_'+camera]=capture.renderer.render().copy()
        finally:capture.renderer.disable_depth_rendering()
        seg=capture._segmentation(camera);mask=np.zeros(seg.shape[:2],np.uint8)
        for cls,ids in capture.entities.items():mask[np.isin(seg[:,:,0],ids)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))]=cls
        labels['semantic_'+camera]=mask;labels['raw_segmentation_'+camera]=seg
        labels['amodal_'+camera]=np.stack([capture.amodal(camera,capture.entities[cls]) for cls in (1,2)],axis=-1).astype(np.uint8)
    for cls,ids in capture.entities.items():
        for gid in ids:
            geometry.append({'geom_id':int(gid),'class_id':cls,'kind':'capsule' if cls==1 else 'box',
                             'world_from_geom':transform(sim.data.geom_xmat[gid].reshape(3,3),sim.data.geom_xpos[gid]).tolist(),
                             'size':sim.model.geom_size[gid].tolist()})
    return labels,geometry
