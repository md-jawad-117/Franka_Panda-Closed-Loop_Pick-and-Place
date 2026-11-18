import numpy as np
from collections import deque
from cl_pickplace.retention_feedback_v10r1 import RetentionFSM
from cl_pickplace.task_fsm import S
from test_retention_feedback import setup,obs,start_carry


def revised():
    cfg,_=setup();return RetentionFSM(cfg,np.zeros(7),.4,'closed_loop')


def test_first_slip_warning_holds_immediately_and_second_distinct_warning_aborts():
    fsm=revised();start_carry(fsm)
    cmd=fsm._carry_guard(obs(1.6,z=.52,cap_z=.508))
    assert cmd is not None and cmd.gripper==fsm.g['closed_ctrl'] and fsm.guard_action=='retention_slip_warning_hold'
    fsm._carry_guard(obs(1.8,z=.52,cap_z=.508));assert fsm.state==S.ABORT


def test_preopening_tick_cannot_open_on_first_slip_warning():
    fsm=revised();start_carry(fsm);fsm.state=S.RELEASE;fsm.release_opened_t=None
    cmd=fsm.update(obs(1.6,z=.52,cap_z=.508))
    assert cmd.gripper==fsm.g['closed_ctrl'] and fsm.release_opened_t is None
    assert fsm.retention_phase=='carrying' and fsm.t_enter==1.6


def test_visibility_search_retracts_without_yawing_near_placement_surface():
    fsm=revised();start_carry(fsm);fsm.state=S.DESCEND_SLOT
    near=obs(2.1,z=.43,cap_z=.428,visible=False);cmd=fsm._carry_guard(near)
    assert cmd.p_des[2]>=.55-1e-9 and fsm.guard_action=='retention_observe_retract'
    assert fsm.state!=S.ABORT


def test_excessive_width_is_uncertain_not_empty_and_does_not_trigger_blind_release():
    fsm=revised();start_carry(fsm)
    fsm._carry_guard(obs(1.6,width=.07,visible=False));cmd=fsm._carry_guard(obs(1.7,width=.07,visible=False))
    assert fsm.state==S.ABORT and fsm.retention_retries==0 and cmd.gripper==fsm.g['closed_ctrl']


def test_actual_descent_to_release_transition_retains_guard_until_opening():
    fsm=revised();start_carry(fsm);fsm.state=S.DESCEND_SLOT
    current=obs(1.6,z=.424,cap_z=.422)
    current.ee_pos[1]=.15;current.est.capsule.pos[1]=.15
    fsm.slot_hist=deque([(1.4,np.array([.5,.15]))]);fsm.t_in_tol=1.
    cmd=fsm.update(current)
    assert fsm.state==S.RELEASE and fsm.retention_phase=='carrying'
    assert fsm.retention_reference is not None and cmd.gripper==fsm.g['closed_ctrl']
    slipped=obs(1.8,z=.424,cap_z=.414)
    slipped.ee_pos[1]=.15;slipped.est.capsule.pos[1]=.15
    cmd=fsm.update(slipped)
    assert cmd.gripper==fsm.g['closed_ctrl'] and fsm.release_opened_t is None
