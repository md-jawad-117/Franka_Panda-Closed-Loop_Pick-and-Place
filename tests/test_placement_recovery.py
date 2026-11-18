"""Release and recovery invariants, including unknown evidence and retry limits."""
import copy
import numpy as np
import pytest

from cl_pickplace.config import ROOT,load_config
from cl_pickplace.perception import Estimate,PoseEstimate
from cl_pickplace.task_fsm import TaskFSM,Obs,S
from cl_pickplace.kinematics import gripper_down_quat,quat_to_mat
from test_learned_control import ReplaySensor,FakeWorker
from cl_pickplace.learned_perception import LearnedPerception


def setup(t=3.,state=S.DESCEND_SLOT):
    cfg=load_config(ROOT/'config/learned_recovery_v3.yaml')
    fsm=TaskFSM(cfg,np.zeros(7),.4)
    cap=PoseEstimate(np.array([.45,.15,.411]),0,True,0,'learned:top_cam',t, np.array([1.,0,0]))
    slot=PoseEstimate(np.array([.45,.15,.403]),0,True,0,'learned:top_cam',t)
    obs=Obs(t,np.array([.45,.15,.424]),quat_to_mat(gripper_down_quat(0)),np.zeros(7),.013,Estimate(cap,slot))
    fsm.frozen=obs.est.copy();fsm.state=state;fsm.p_des=obs.ee_pos.copy()
    fsm.slot_hist.extend([(t-.2,slot.pos[:2].copy()),(t,slot.pos[:2].copy())])
    return fsm,obs,cfg


@pytest.mark.parametrize('state',[S.DESCEND_SLOT,S.RELEASE])
def test_pending_innovation_prevents_descent_and_actual_opening(state):
    fsm,obs,cfg=setup(state=state)
    obs.est.slot.observation={'motion_pending':True,'capture_t':obs.t}
    obs.est.slot.visible=False;obs.est.slot.measurement_t-=.2
    command=fsm.update(obs)
    assert fsm.state==S.TRANSPORT and command.gripper==cfg['gripper']['closed_ctrl']
    assert command.p_des[2]>=.55 and not fsm.slot_hist and fsm.release_vetoes==1
    obs.t+=.02;fsm.update(obs)
    assert fsm.release_vetoes==1 # count events, not cached control ticks


def test_open_gripper_is_not_reclosed_when_motion_appears_after_release():
    fsm,obs,cfg=setup(state=S.RELEASE);fsm.release_opened_t=2.9;fsm.t_enter=2.9
    obs.est.slot.observation={'motion_pending':True}
    assert fsm.update(obs).gripper==cfg['gripper']['open_ctrl']


@pytest.mark.parametrize('kind',['position','yaw','age'])
def test_last_moment_change_or_expiry_vetoes_opening(kind):
    fsm,obs,cfg=setup(state=S.RELEASE)
    fsm.release_reference=(obs.est.slot.pos.copy(),obs.est.slot.yaw)
    if kind=='position':obs.est.slot.pos[0]+=.005
    if kind=='yaw':obs.est.slot.yaw+=np.radians(6)
    if kind=='age':obs.est.slot.measurement_t-=.9
    command=fsm.update(obs)
    assert fsm.state==S.TRANSPORT and command.gripper==cfg['gripper']['closed_ctrl']


def test_motion_warning_survives_rejected_capture_until_confirmed():
    cfg=load_config(ROOT/'config/learned_recovery_v3.yaml')
    sensor=ReplaySensor();worker=FakeWorker()
    adapter=LearnedPerception(sensor,cfg['perception']['learned'],np.random.default_rng(1),worker)
    try:
        adapter.update(0);sensor.t=.2;adapter.update(.2)
        worker.position[0]+=.03;sensor.t=.4;assert 'fixture' not in adapter.update(.4)
        assert adapter.latest['fixture']['motion_pending']
        worker.reject=True;sensor.t=.6;adapter.update(.6)
        assert adapter.latest['fixture']['motion_pending']
        worker.reject=False;sensor.t=.8;adapter.update(.8)
        assert adapter.latest['fixture']['motion_pending']
        sensor.t=1.;adapter.update(1.)
        assert not adapter.latest['fixture']['motion_pending']
    finally:adapter.close()


def verify_samples(fsm,obs,start=3.,outside=False):
    fsm.state=S.VERIFY;fsm.verification_started=start;fsm.verification_anchor=obs.ee_pos.copy()
    for t in np.arange(start,start+.601,.2):
        obs.t=float(t);obs.est.slot.measurement_t=obs.t;obs.est.capsule.measurement_t=obs.t
        if outside:obs.est.capsule.pos[1]=.20
        fsm.update(obs)


def test_verification_requires_full_footprint_and_abstains_on_boundary():
    fsm,obs,_=setup();fsm.verification_started=3.
    assert fsm.inspect_placement(obs)[0]=='inside'
    obs.est.capsule.yaw=np.pi/2;obs.est.capsule.axis=np.array([0.,1,0])
    assert fsm.inspect_placement(obs)[0]=='outside' # center alone is insufficient
    obs.est.capsule.axis=np.array([1.,0,0]);obs.est.capsule.yaw=0
    obs.est.capsule.pos[1]+=.0095
    assert fsm.inspect_placement(obs)[0]=='ambiguous'


def test_cached_pose_cannot_fake_multiple_verification_samples():
    fsm,obs,_=setup(state=S.VERIFY);fsm.verification_started=obs.t
    for i in range(10):obs.t=3.+i*.02;fsm.update(obs)
    assert len(fsm.verification_samples)==1 and fsm.state==S.VERIFY


def test_unknown_or_skewed_or_preinspection_poses_cannot_pass():
    fsm,obs,_=setup();fsm.verification_started=3.
    obs.est.capsule.visible=False;assert fsm.inspect_placement(obs)[0]=='unknown'
    obs.est.capsule.visible=True;obs.est.capsule.measurement_t-=.1
    assert fsm.inspect_placement(obs)[0]=='unknown'
    obs.est.slot.measurement_t-=.1
    assert fsm.inspect_placement(obs)[0]=='unknown'


def test_confirmed_inside_waits_for_monitor_horizon_then_done():
    fsm,obs,_=setup();verify_samples(fsm,obs)
    assert fsm.state==S.VERIFY
    for t in np.arange(3.8,5.201,.2):
        obs.t=float(t);obs.est.slot.measurement_t=obs.t;obs.est.capsule.measurement_t=obs.t;fsm.update(obs)
    assert fsm.state==S.DONE
    obs.est.capsule.visible=False;obs.t+=.02;fsm.update(obs)
    assert fsm.state==S.VERIFY # observation loss revokes completion while episode still active


def test_outside_retries_from_current_estimate_and_limit_is_bounded():
    fsm,obs,_=setup();verify_samples(fsm,obs,outside=True)
    assert fsm.state==S.APPROACH_OBJECT and fsm.placement_retries==1
    fsm.placement_retries=2;verify_samples(fsm,obs,start=4.,outside=True)
    assert fsm.state==S.ABORT and fsm.verification_status=='retry_limit'


def test_missing_verification_views_abort_without_false_done():
    fsm,obs,_=setup(state=S.VERIFY)
    fsm.verification_started=3.;obs.t=11.;obs.est.capsule.visible=False
    command=fsm.update(obs)
    assert fsm.state==S.ABORT and command.gripper==255


def test_open_loop_ignores_optional_recovery_to_preserve_baseline():
    fsm,obs,_=setup(state=S.RELEASE);fsm.mode='open_loop'
    obs.est.slot.observation={'motion_pending':True}
    assert fsm.update(obs).gripper==255 and fsm.release_vetoes==0


def test_recovery_physical_path_cannot_read_target_truth(monkeypatch):
    from cl_pickplace.episode import Episode
    cfg=load_config(ROOT/'config/learned_recovery_v3.yaml')
    ep=Episode(cfg,perception_mode='learned',perturb='none')
    try:
        ep.reset(3);monkeypatch.setattr(ep,'_log',lambda obs:None)
        def forbidden(*args):raise AssertionError('Privileged target data entered recovery policy')
        for name in ('capsule_pos','capsule_yaw','slot_pos','slot_yaw'):monkeypatch.setattr(ep.sim,name,forbidden)
        # No evaluator/termination calls: policy runs through release and inspection.
        for _ in range(650):ep.control_tick();ep.physics_period()
        assert ep.fsm.verification_history
        assert any(b=='VERIFY' for _,_,b,_ in ep.fsm.stats.transitions)
    finally:ep.perception.close()
