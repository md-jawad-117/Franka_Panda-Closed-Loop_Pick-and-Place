"""Optional sensor-only view search and observation checks."""
import copy
import numpy as np
from cl_pickplace.config import ROOT,load_config
from cl_pickplace.observation_selection import select_observation
from cl_pickplace.task_fsm import TaskFSM,Obs,S
from cl_pickplace.perception import Estimate,PoseEstimate
from cl_pickplace.kinematics import gripper_down_quat,quat_to_mat


def setup(t=2.):
    cfg=load_config(ROOT/'config/learned_reacquisition_v4.yaml');f=TaskFSM(cfg,np.zeros(7),.4)
    cap=PoseEstimate(np.array([.45,.15,.411]),0,True,0,'learned:wrist_cam',t,np.array([1.,0,0]))
    slot=PoseEstimate(np.array([.45,.15,.403]),0,True,0,'learned:top_cam',t)
    o=Obs(t,np.array([.45,.15,.55]),quat_to_mat(gripper_down_quat(0)),np.zeros(7),.08,Estimate(cap,slot))
    f.frozen=o.est.copy();return f,o,cfg


def views():
    cap={'accepted':True,'position_m':[.45,.15,.411],'axis_world':[1.,0,0],'quality':{},'reasons':[]}
    slot={**cap,'position_m':[.45,.15,.403]}
    return {c:{'capsule':copy.deepcopy(cap),'fixture':copy.deepcopy(slot)} for c in ('wrist_cam','top_cam')}


def test_floor_check_uses_observed_floor_and_allows_table_object():
    _,_,cfg=setup();settings=cfg['perception']['learned']['observation_selection'];v=views()
    v['wrist_cam']['capsule']['position_m'][2]=.396
    result,reason=select_observation(v,'capsule',settings)
    assert result['source']=='learned:top_cam' and reason['rejected_views']['wrist_cam']
    # An object on the adjacent table is below floor-plus-radius by bottom thickness.
    v=views()
    for c in v:v[c]['capsule']['position_m'][2]=.408
    assert select_observation(v,'capsule',settings)[0] is not None
    for c in v:
        for target in v[c]:v[c][target]['position_m'][2]+=1.3
    assert select_observation(v,'capsule',settings)[0] is not None


def test_disagreeing_views_abstain_and_legacy_routing_remains_unchanged():
    _,_,cfg=setup();v=views();v['wrist_cam']['capsule']['position_m'][0]+=.025
    chosen,reason=select_observation(v,'capsule',cfg['perception']['learned']['observation_selection'])
    assert chosen is None and reason['cross_view_disagreement']
    assert select_observation(v,'capsule',None)[0]['source']=='learned:wrist_cam'


def test_two_view_floor_rim_alias_uses_lower_observed_plane():
    _,_,cfg=setup();v=views();v['wrist_cam']['fixture']['position_m'][2]+=.010
    result,_=select_observation(v,'fixture',cfg['perception']['learned']['observation_selection'])
    assert result['source']=='learned:top_cam'
    v['top_cam']['capsule']['accepted']=False
    assert select_observation(v,'capsule',cfg['perception']['learned']['observation_selection'])[0] is not None


def test_lost_object_moves_to_acquisition_instead_of_waiting_forever():
    f,o,_=setup();f.state=S.APPROACH_OBJECT;o.est.capsule.visible=False
    command=f.update(o)
    assert f.state==S.ACQUIRE and f.guard_action=='active_acquire_capsule' and command.gripper==255
    f.acquisition_started=o.t-8.1;f.update(o)
    assert f.state==S.ABORT and f.acquisition_history[-1]['outcome']=='inconclusive'


def test_startup_wait_allows_normal_temporal_confirmation_before_search():
    f,o,_=setup(t=0);f.frozen=None;f.state=S.HOME;o.est.capsule.visible=False
    assert f.update(o).joint_goal is not None and f.state==S.HOME
    o.t=.7;f.update(o);assert f.state==S.ACQUIRE


def test_cached_camera_transform_is_not_combined_with_later_robot_pose():
    f,o,_=setup();camera=np.eye(4);camera[:3,:3]=np.diag([1,-1,-1]);camera[:3,3]=o.ee_pos+[.05,0,.03]
    o.est.capsule.observation={'capture_t':o.t,'wrist_world_from_camera':camera.tolist()}
    first,_=f.observation_waypoint(o,'capsule',0)
    o.t+=.1;o.ee_pos+=np.array([.02,.01,.01])
    second,_=f.observation_waypoint(o,'capsule',0)
    np.testing.assert_allclose(first,second,atol=1e-9)


def test_calibrated_waypoint_centers_target_on_camera_optical_axis():
    f,o,_=setup();camera=np.eye(4);camera[:3,:3]=np.diag([1,-1,-1]);camera[:3,3]=o.ee_pos+[.05,0,.03]
    o.est.capsule.observation={'capture_t':o.t,'wrist_world_from_camera':camera.tolist()}
    position,q=f.observation_waypoint(o,'capsule',0);rotation=quat_to_mat(q)
    origin=position+rotation@f.hand_camera_offset;ray=(rotation@f.hand_camera_rotation)[:,2]
    relative=o.est.capsule.pos-origin
    assert np.linalg.norm(np.cross(relative,ray))<1e-9 and relative@ray>0
    np.testing.assert_allclose(ray,[0,0,-1],atol=1e-9)


def test_acquisition_needs_distinct_fresh_samples_after_arrival():
    f,o,_=setup();f.begin_acquisition(o);o.ee_pos=f.observation_waypoint(o,'capsule',0)[0]
    for i in range(8):o.t=2+i*.02;f.update(o)
    assert f.state==S.ACQUIRE and len(f.acquisition_samples)==1
    for t in [2.2,2.4,2.6]:
        o.t=t;o.est.capsule.measurement_t=t;o.est.slot.measurement_t=t;f.update(o)
        if f.state!=S.ACQUIRE:break
    assert f.state==S.APPROACH_OBJECT and f.acquisition_history[-1]['outcome']=='confirmed'


def test_retry_grasp_raises_finger_pads_above_nominal_rim_only_when_nearby():
    f,o,cfg=setup();f.placement_retries=1
    target=f.grasp_position(o,o.est)
    assert target[2]>=o.est.slot.pos[2]+.010+.0073+.002-1e-9
    f.placement_retries=0
    assert f.grasp_position(o,o.est)[2]==o.est.capsule.pos[2]+cfg['task']['grasp_height_offset']
    f.placement_retries=1;o.est.slot.pos[0]+=.2
    assert f.grasp_position(o,o.est)[2]==o.est.capsule.pos[2]+cfg['task']['grasp_height_offset']


def test_retry_clearance_cannot_drop_when_fixture_becomes_occluded():
    f,o,_=setup();f.placement_retries=1;f.state=S.APPROACH_OBJECT
    initial=f.grasp_position(o,o.est)
    f._go(S.DESCEND_OBJECT,o.t,'above object')
    o.est.slot.visible=False;o.t+=.2
    np.testing.assert_allclose(f.grasp_position(o,o.est),initial)


def test_blocked_fourth_attempt_is_not_counted_as_a_gripper_closure():
    f,o,_=setup();f.state=S.GRASP;f.placement_retries=1;f.recovery_grasp_attempts=3
    f.update(o)
    assert f.state==S.ABORT and f.recovery_grasp_attempts==3


def test_recovery_grasp_limit_and_open_loop_behavior():
    f,o,_=setup();f.state=S.GRASP;f.placement_retries=1;f.recovery_grasp_attempts=4
    assert f.update(o).gripper==255 and f.state==S.ABORT
    f,o,_=setup();f.mode='open_loop';f.state=S.APPROACH_OBJECT;o.est.capsule.visible=False
    f.update(o);assert f.state!=S.ACQUIRE


def test_rim_extraction_is_a_bounded_robot_trajectory_and_requires_clear_direction():
    f,o,_=setup();f.placement_retries=1;o.est.capsule.pos[0]-=.025
    f.acquisition_settings['rim_extraction']['enabled']=True
    assert f.begin_extraction(o) and f.state==S.EXTRACT
    original=o.est.capsule.pos.copy()
    for _ in range(45):
        command=f.extraction_tick(o)
        o.ee_pos=command.p_des.copy();o.ee_mat=quat_to_mat(command.q_des);o.t+=.1
        f.extraction_tick(o)
        if f.state!=S.EXTRACT:break
    assert f.extraction_attempts==1
    np.testing.assert_array_equal(o.est.capsule.pos,original)
    assert any(event['event']=='trajectory_complete' for event in f.extraction_history)
    assert not f.begin_extraction(o) # no repeated unbounded push
    f,o,_=setup()
    assert not f.begin_extraction(o) # capsule and fixture centers coincide


def test_rim_extraction_timeout_is_failure_not_false_completion():
    f,o,_=setup();f.placement_retries=1;o.est.capsule.pos[0]-=.025
    f.acquisition_settings['rim_extraction']['enabled']=True
    assert f.begin_extraction(o)
    o.t+=8.1;command=f.extraction_tick(o)
    assert f.state==S.ABORT and command.gripper==255


def test_real_acquisition_under_dropout_cannot_read_target_truth(monkeypatch):
    from cl_pickplace.episode import Episode
    cfg=load_config(ROOT/'config/learned_reacquisition_v4.yaml')
    ep=Episode(cfg,perception_mode='learned',perturb='none')
    try:
        ep.reset(3);monkeypatch.setattr(ep,'_log',lambda obs:None)
        worker=ep.perception._learned.worker;predict=worker.predict;calls=[]
        def dropout(arrays):
            response=predict(arrays);calls.append(set(arrays))
            if len(calls)<=5:
                for view in response['views'].values():view['capsule']['accepted']=False;view['capsule']['reasons']=['injected_visual_dropout']
            return response
        monkeypatch.setattr(worker,'predict',dropout)
        def forbidden(*args):raise AssertionError('Privileged target state entered acquisition')
        for name in ('capsule_pos','capsule_yaw','slot_pos','slot_yaw'):monkeypatch.setattr(ep.sim,name,forbidden)
        for _ in range(500):ep.control_tick();ep.physics_period()
        assert any(t[2]=='ACQUIRE' for t in ep.fsm.stats.transitions)
        assert ep.fsm.hand_camera_rotation is not None
        assert calls and all(fields=={f'{field}_{camera}' for field in ('rgb','depth','valid','K','T') for camera in ('wrist_cam','top_cam')} for fields in calls)
        assert worker.metadata['capsule_estimator']=='capsule_raised_surface_fallback_v4'
    finally:ep.perception.close()
