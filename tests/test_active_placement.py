"""Active placement: distinct visual evidence, motion response and age bounds."""
import copy
import numpy as np
import pytest

from cl_pickplace.config import load_config,ROOT
from cl_pickplace.task_fsm import TaskFSM,S
from test_learned_control import observations
from cl_pickplace.kinematics import gripper_down_quat


def setup():
    cfg=load_config(ROOT/'config/learned_active.yaml')
    fsm=TaskFSM(cfg,np.zeros(7),.4,'closed_loop');obs=observations()
    obs.est.slot.quality={'footprint_extent_m':[.066,.041]}
    obs.est.capsule.visible=False;obs.est.capsule.lost_frames=10
    fsm.frozen=obs.est.copy();fsm.state=S.TRANSPORT
    fsm.p_des=obs.ee_pos.copy()
    return fsm,obs


def at_view(fsm,obs):
    command=fsm.update(obs);obs.ee_pos=command.p_des.copy()
    for t in [1.0,1.2,1.4,1.6]:
        obs.t=t;obs.est.slot.measurement_t=t;fsm.update(obs)


def test_active_view_keeps_grasp_and_advances_from_occlusion():
    fsm,obs=setup();command=fsm.update(obs)
    assert command.p_des[0]==pytest.approx(obs.est.slot.pos[0])
    assert command.p_des[2]>obs.est.slot.pos[2]+.08
    assert command.gripper==fsm.g['closed_ctrl']
    assert fsm.state==S.TRANSPORT
    obs.est.slot.quality={};obs.t+=3.1;command=fsm.update(obs)
    assert fsm.view_index==1 and command.p_des[0]<obs.est.slot.pos[0]-.03


def test_cached_or_prearrival_frames_cannot_confirm_view():
    fsm,obs=setup();command=fsm.update(obs);obs.ee_pos=command.p_des.copy()
    for t in np.arange(1.,2.,.02):
        obs.t=float(t);fsm.update(obs)
    assert fsm.state==S.TRANSPORT
    assert len(fsm.observation_samples)==1


def test_distinct_stable_view_enables_diagonal_descent():
    fsm,obs=setup();at_view(fsm,obs)
    assert fsm.state==S.DESCEND_SLOT
    command=fsm.update(obs)
    np.testing.assert_allclose(command.p_des[:2],obs.est.slot.pos[:2])
    assert fsm.guard_action=='active_place_visible'


def test_motion_or_expired_view_aborts_before_opening():
    for reason in ['motion','age']:
        fsm,obs=setup();at_view(fsm,obs)
        if reason=='motion':
            obs.t+=.2;obs.est.slot.measurement_t=obs.t;obs.est.slot.pos[0]+=.01
        else:
            obs.t+=.82;obs.est.slot.visible=False
        command=fsm.update(obs)
        assert fsm.state==S.TRANSPORT and fsm.observation_attempts==1
        assert command.gripper==fsm.g['closed_ctrl']
        assert fsm.guard_action=='active_observe_destination'


def test_final_commit_retains_true_timestamp_and_deadline():
    fsm,obs=setup();at_view(fsm,obs)
    obs.ee_pos=obs.est.slot.pos+np.array([0,0,fsm.p['place_height_offset']])
    obs.est.slot.visible=False;obs.t+=.2
    fsm.update(obs);obs.t+=.22;fsm.update(obs)
    assert fsm.state==S.RELEASE and fsm.guard_action=='active_place_commit'
    assert obs.est.slot.measurement_t==pytest.approx(1.6)


def test_active_profile_does_not_change_open_loop_definition():
    fsm,obs=setup();fsm.mode='open_loop'
    assert not fsm.active_placement


def test_fixed_wait_does_not_use_new_destination_or_release():
    fsm,obs=setup();fsm.p=copy.deepcopy(fsm.p)
    fsm.p['placement_policy']='reactive_v1';fsm.p['transport_wait_s']=1.
    fsm.t_enter=obs.t
    original=fsm.p_des.copy();command=fsm.update(obs)
    np.testing.assert_array_equal(command.p_des,original)
    assert fsm.state==S.TRANSPORT and fsm.guard_action=='fixed_transport_wait'


def test_partial_but_stable_box_cannot_authorize_placement():
    fsm,obs=setup();obs.est.slot.quality={'footprint_extent_m':[.052,.041]}
    at_view(fsm,obs)
    assert fsm.state==S.TRANSPORT and fsm.qualified_slot is None


def test_active_physical_control_uses_no_privileged_target_getters(monkeypatch):
    from cl_pickplace.episode import Episode
    cfg=load_config(ROOT/'config/learned_active.yaml')
    ep=Episode(cfg,perception_mode='learned',perturb='none')
    try:
        ep.reset(3)
        monkeypatch.setattr(ep,'_log',lambda obs:None)
        def forbidden(*args):raise AssertionError('Privileged target state entered active control')
        for name in ['capsule_pos','capsule_yaw','slot_pos','slot_yaw']:
            monkeypatch.setattr(ep.sim,name,forbidden)
        for _ in range(350):
            ep.control_tick();ep.physics_period()
        assert any(b=='TRANSPORT' for _,_,b,_ in ep.fsm.stats.transitions)
        assert ep.perception._learned.sample_index>=35
    finally:ep.perception.close()
