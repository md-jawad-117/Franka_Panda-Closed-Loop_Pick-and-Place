from pathlib import Path
import json,numpy as np
from cl_pickplace.reobserving_centering import ReobservingCenterFSM
from cl_pickplace.task_fsm import S
from test_centering_feedback import setup,observed

ROOT=Path(__file__).resolve().parents[1]


def revised():
    fsm=setup();cfg={'task':fsm.p,'gripper':fsm.g,'control':fsm.c,'perception':fsm.pc}
    cfg['task']['visual_centering'].update(json.loads((ROOT/'config/reobserving_control_protocol_v21.json').read_text())['runtime_additions'])
    result=ReobservingCenterFSM(cfg,np.zeros(7),.4,'closed_loop');result.state=S.VERIFY
    return result


def at_view(fsm,t,visible=True):
    obs=observed(t);obs.ee_pos=np.r_[fsm.centering_plan['start'][:2],fsm.centering_plan['clear_z']]
    from cl_pickplace.kinematics import quat_to_mat,gripper_down_quat,wrap_half_pi
    obs.ee_mat=quat_to_mat(gripper_down_quat(wrap_half_pi(fsm.centering_plan['yaw_rad'])))
    obs.est.capsule.visible=visible
    return obs


def test_missing_capsule_does_not_block_safe_high_waypoint_but_blocks_lowering():
    fsm=revised();fsm._try_centering(observed(0),'test');fsm._centering_phase(observed(0),'approach')
    obs=at_view(fsm,.8,False);obs.est.capsule.measurement_t=0
    fsm.centering_tick(obs)
    assert fsm.centering_phase=='observe' and fsm.centering_active
    fsm.centering_tick(at_view(fsm,1.,False));assert fsm.centering_phase=='observe'


def test_two_new_stable_pairs_revalidate_before_descent_and_cached_pair_cannot():
    fsm=revised();fsm._try_centering(observed(0),'test');fsm._centering_phase(observed(0),'observe')
    fsm.centering_tick(at_view(fsm,.2));assert fsm.centering_phase=='observe'
    old=at_view(fsm,.22);old.est.capsule.measurement_t=.2;old.est.slot.measurement_t=.2
    fsm.centering_tick(old);assert fsm.centering_phase=='observe'
    fsm.centering_tick(at_view(fsm,.4));assert fsm.centering_phase=='lower'
    assert [r['event'] for r in fsm.centering_history].count('revalidated')==1


def test_safe_waypoint_observation_timeout_retracts_without_touching_or_done():
    fsm=revised();fsm._try_centering(observed(0),'test');fsm._centering_phase(observed(0),'observe')
    fsm.centering_tick(at_view(fsm,2.1,False))
    assert fsm.state!=S.DONE and fsm.centering_phase=='retreat'


def test_separated_observations_do_not_count_as_continuous_confirmation():
    fsm=revised();fsm._try_centering(observed(0),'test');fsm._centering_phase(observed(0),'observe')
    fsm.centering_tick(at_view(fsm,.2));fsm.centering_tick(at_view(fsm,.6))
    assert fsm.centering_phase=='observe'
    fsm.centering_tick(at_view(fsm,.8));assert fsm.centering_phase=='lower'
