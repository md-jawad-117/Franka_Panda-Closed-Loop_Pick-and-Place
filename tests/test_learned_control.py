"""Learned observation guards and sensor-only integration contracts."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest

from cl_pickplace.perception import Estimate,PoseEstimate
from cl_pickplace.task_fsm import TaskFSM,Obs,S
from cl_pickplace.kinematics import quat_to_mat,gripper_down_quat
from cl_pickplace.learned_perception import LearnedPerception


def observations(t=1.):
    cap=PoseEstimate(np.array([.45,-.1,.408]),0.,True,0,'learned:top_cam',t)
    slot=PoseEstimate(np.array([.45,.15,.403]),0.,True,0,'learned:top_cam',t)
    return Obs(t,np.array([.45,0,.60]),quat_to_mat(gripper_down_quat(0)),np.zeros(7),.013,Estimate(cap,slot))


def learned_fsm(cfg,mode='closed_loop'):
    fsm=TaskFSM(cfg,np.zeros(7),.4,mode);fsm.require_visual_observations=True
    return fsm


def test_initial_missing_detections_cannot_command_default_coordinates(cfg):
    fsm=learned_fsm(cfg);obs=observations();obs.est.slot.visible=False
    command=fsm.update(obs)
    assert fsm.frozen is None and fsm.state==S.HOME
    assert command.p_des is None
    np.testing.assert_array_equal(command.joint_goal,obs.q_arm)
    assert command.gripper==cfg['gripper']['open_ctrl']


def test_stale_object_during_descent_retracts_instead_of_grasping(cfg):
    fsm=learned_fsm(cfg);obs=observations();fsm.frozen=obs.est.copy();fsm.state=S.DESCEND_OBJECT
    obs.ee_pos=np.array([.48,-.12,.414]);obs.est.capsule.measurement_t=obs.t-.3
    command=fsm.update(obs)
    assert fsm.state==S.APPROACH_OBJECT and fsm.guard_action=='reobserve_object'
    np.testing.assert_allclose(command.p_des[:2],obs.ee_pos[:2])
    assert command.p_des[2]>=.55 and command.gripper==cfg['gripper']['open_ctrl']


def test_missing_destination_blocks_release_and_preserves_grasp(cfg):
    fsm=learned_fsm(cfg);obs=observations();fsm.frozen=obs.est.copy();fsm.state=S.DESCEND_SLOT
    obs.ee_pos=np.array([.45,.15,.425]);obs.est.slot.visible=False
    obs.est.slot.measurement_t=obs.t-.9
    fsm.slot_hist.extend([(obs.t-.2,obs.est.slot.pos[:2]),(obs.t,obs.est.slot.pos[:2])])
    command=fsm.update(obs)
    assert fsm.state==S.TRANSPORT and fsm.guard_action=='reobserve_destination'
    assert command.gripper==cfg['gripper']['closed_ctrl'] and command.p_des[2]>=.55
    assert fsm.t_in_tol is None


def test_final_grasp_commitment_is_bounded_and_does_not_relabel_visibility(cfg):
    fsm=learned_fsm(cfg);obs=observations();fsm.frozen=obs.est.copy();fsm.state=S.DESCEND_OBJECT
    obs.ee_pos=obs.est.capsule.pos+np.array([0,0,.005]);obs.est.capsule.visible=False
    obs.est.capsule.measurement_t=obs.t-.3
    fsm.update(obs)
    assert fsm.guard_action=='bounded_grasp_commit' and not obs.est.capsule.visible
    fsm.state=S.DESCEND_OBJECT;obs.est.capsule.measurement_t=obs.t-.7
    fsm.update(obs)
    assert fsm.guard_action=='reobserve_object' and fsm.state==S.APPROACH_OBJECT


def test_stationary_final_place_commitment_keeps_real_age_and_has_deadline(cfg):
    fsm=learned_fsm(cfg);obs=observations();fsm.frozen=obs.est.copy();fsm.state=S.DESCEND_SLOT
    obs.est.capsule.visible=False;obs.est.capsule.lost_frames=10
    obs.ee_pos=obs.est.slot.pos+np.array([0,0,cfg['task']['place_height_offset']])
    fsm.p_des=obs.ee_pos.copy();fsm.q_des=gripper_down_quat(0)
    obs.est.slot.visible=False;obs.est.slot.measurement_t=obs.t-.3
    fsm.slot_hist.extend([(obs.t-.5,obs.est.slot.pos[:2].copy()),(obs.t-.3,obs.est.slot.pos[:2].copy())])
    fsm.update(obs)
    assert fsm.guard_action=='bounded_place_commit' and fsm.state==S.DESCEND_SLOT
    obs.t+=.22;fsm.update(obs)
    assert fsm.state==S.RELEASE and not obs.est.slot.visible
    assert obs.est.slot.measurement_t==pytest.approx(.7)
    # Beyond the deadline the same geometry must trigger re-observation.
    fsm.state=S.DESCEND_SLOT;obs.t=1.6;fsm.update(obs)
    assert fsm.state==S.TRANSPORT and fsm.guard_action=='reobserve_destination'


def test_missing_destination_known_to_be_moving_cannot_commit(cfg):
    fsm=learned_fsm(cfg);obs=observations();fsm.frozen=obs.est.copy();fsm.state=S.DESCEND_SLOT
    obs.est.slot.visible=False;obs.est.slot.measurement_t=obs.t-.2
    obs.ee_pos=obs.est.slot.pos+np.array([0,0,.025]);fsm.p_des=obs.ee_pos.copy()
    fsm.slot_hist.extend([(obs.t-.4,np.array([.45,.13])),(obs.t-.2,np.array([.45,.15]))])
    fsm.update(obs)
    assert fsm.state==S.TRANSPORT and fsm.guard_action=='reobserve_destination'


def test_grasp_offset_uses_capture_time_robot_pose_only(cfg):
    fsm=learned_fsm(cfg);obs=observations();fsm.state=S.TRANSPORT
    obs.est.capsule.pos=obs.ee_pos+np.array([.008,0,-.002])
    obs.t=1.1;fsm._place_xy_yaw(obs,obs.est)
    assert fsm.in_hand is None
    obs.t=1.;fsm._place_xy_yaw(obs,obs.est)
    assert fsm.in_hand is not None


def test_cached_carried_pose_cannot_trigger_false_drop_when_hand_moves(cfg):
    fsm=learned_fsm(cfg);obs=observations()
    obs.est.capsule.pos=obs.ee_pos+np.array([-.05,0,0]);obs.t=1.18
    assert not fsm.dropped(obs) # use current gripper width between captures
    obs.t=1.
    assert fsm.dropped(obs) # a genuinely current distant capsule is drop evidence


def test_carried_capsule_occlusion_uses_gripper_measurement(cfg):
    fsm=learned_fsm(cfg);obs=observations();fsm.frozen=obs.est.copy();fsm.state=S.TRANSPORT
    obs.est.capsule.visible=False;obs.est.capsule.lost_frames=20
    fsm.update(obs)
    assert fsm.guard_action=='' and fsm.stats.num_regrasps==0


def test_repeated_cached_measurement_does_not_fake_stationarity_window(cfg):
    fsm=learned_fsm(cfg);obs=observations()
    for i in range(20):
        obs.t=1+i*.02
        assert fsm._slot_motion(obs)==float('inf')
    assert len(fsm.slot_hist)==1
    obs.est.slot.measurement_t=1.2;obs.t=1.2
    assert fsm._slot_motion(obs)==0


def test_open_loop_freezes_only_valid_initial_snapshot(cfg):
    fsm=learned_fsm(cfg,'open_loop');obs=observations();obs.est.slot.visible=False
    fsm.update(obs);assert fsm.frozen is None
    obs.est.slot.visible=True;fsm.update(obs);assert fsm.frozen is not None
    fsm.state=S.DESCEND_OBJECT;obs.est.capsule.visible=False
    fsm.update(obs)
    assert fsm.guard_action=='' # defined open-loop baseline continues frozen targets


def test_estimate_copy_detaches_axis_and_quality():
    obs=observations();obs.est.capsule.axis=np.array([1.,0,0]);obs.est.capsule.quality={'x':[1]}
    detached=obs.est.copy();detached.capsule.pos[:]=0;detached.capsule.axis[:]=0;detached.capsule.quality['x'][0]=2
    assert obs.est.capsule.pos[0]==.45 and obs.est.capsule.axis[0]==1 and obs.est.capsule.quality['x']==[1]


class ReplaySensor:
    def __init__(self):self.t=0.;self.closed=False;self.position=np.zeros(3)
    def capture(self,camera):
        return SimpleNamespace(timestamp=self.t,rgb=np.zeros((240,320,3),np.uint8),depth=np.ones((240,320),np.float32),
                               world_position=self.position.copy(),world_rotation=np.eye(3),fovy_deg=60.)
    def close(self):self.closed=True


class FakeWorker:
    metadata={'ready':True}
    def __init__(self):self.position=np.array([.45,-.1,.408]);self.requests=[];self.reject=False
    def predict(self,arrays):
        self.requests.append(set(arrays))
        result={'accepted':not self.reject,'position_m':self.position.tolist(),'axis_world':[1.,0,0],
                'quality':{},'reasons':['test_missing'] if self.reject else []}
        return {'views':{camera:{target:copy.deepcopy(result) for target in ('capsule','fixture')}
                         for camera in ('wrist_cam','top_cam')},'wall_seconds':.1}
    def close(self):pass


def test_confirmation_requires_distinct_captures_and_payload_has_only_sensors(cfg):
    sensor=ReplaySensor();worker=FakeWorker();config=copy.deepcopy(cfg['perception']['learned'])
    config['depth_noise']['sigma_m']=0;config['depth_noise']['dropout']=0
    learned=LearnedPerception(sensor,config,np.random.default_rng(0),worker)
    try:
        assert learned.update(0)=={}
        sensor.t=.1;assert learned.update(.1) is None
        assert len(worker.requests)==1
        sensor.t=.2;result=learned.update(.2)
        assert result['capsule'][2]=='learned:wrist_cam'
        assert worker.requests[0]=={f'{field}_{camera}' for field in ('rgb','depth','valid','K','T') for camera in ('wrist_cam','top_cam')}
        worker.position[0]+=.03;sensor.t=.4
        assert learned.update(.4)=={}
        sensor.t=.6;assert 'capsule' in learned.update(.6)
        worker.reject=True;sensor.t=.8
        assert learned.update(.8)=={}
    finally:learned.close()
    assert sensor.closed


def test_carried_capsule_can_confirm_in_wrist_frame_but_fixture_cannot(cfg):
    sensor=ReplaySensor();worker=FakeWorker();config=copy.deepcopy(cfg['perception']['learned'])
    learned=LearnedPerception(sensor,config,np.random.default_rng(0),worker)
    try:
        learned.update(0);sensor.t=.2;assert 'capsule' in learned.update(.2)
        sensor.t=.4;sensor.position[0]=.03;worker.position[0]+=.03
        measurements=learned.update(.4)
        assert 'capsule' in measurements and 'fixture' not in measurements
    finally:learned.close()


def test_real_learned_controller_path_does_not_read_target_pose_getters(cfg,monkeypatch):
    from cl_pickplace.episode import Episode
    ep=Episode(cfg,perception_mode='learned')
    try:
        ep.reset(3);monkeypatch.setattr(ep,'_log',lambda obs:None)
        import threading
        capture_lock=threading.Lock();refreshes=[]
        def refresh():
            assert not capture_lock.locked()
            refreshes.append(ep.t)
        ep.perception.set_live_hooks(lambda:capture_lock,refresh)
        def forbidden(*args):raise AssertionError('Learned policy accessed privileged target pose')
        for name in ('capsule_pos','capsule_yaw','slot_pos','slot_yaw'):monkeypatch.setattr(ep.sim,name,forbidden)
        ep.sim.data.time+=.2;ep.control_tick()
        assert ep.perception._learned.sample_index==2
        assert ep.est.capsule.visible and ep.est.slot.visible
        assert ep.est.capsule.source.startswith('learned:')
        assert not hasattr(ep.perception._learned,'sim')
        assert refreshes # actual neural I/O wait refreshed with capture lock released
        client=ep.perception._learned.worker;client.process.kill();client.process.wait()
        ep.sim.data.time+=.2
        with pytest.raises(RuntimeError,match='without oracle fallback'):ep.control_tick()
    finally:ep.perception.close()
