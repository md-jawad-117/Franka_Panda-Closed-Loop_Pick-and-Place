"""Behavioral tests for evidence, outage handling, bounded retries and isolation."""
import ast,copy,json
from pathlib import Path
import numpy as np
from cl_pickplace.config import load_config
from cl_pickplace.perception import PoseEstimate,Estimate
from cl_pickplace.task_fsm import TaskFSM,Obs,S
from cl_pickplace.retention_feedback import RetentionFSM
from cl_pickplace.kinematics import quat_to_mat,gripper_down_quat

ROOT=Path(__file__).resolve().parents[1]


def setup():
    cfg=load_config(ROOT/'config/learned_reacquisition_v4.yaml')
    cfg['task']['retention_feedback']=json.loads((ROOT/'config/retention_protocol_v10.json').read_text())['settings']
    fsm=RetentionFSM(cfg,np.zeros(7),.4,'closed_loop')
    return cfg,fsm


def obs(t,z=.45,cap_z=.448,width=.015,visible=True,stamp=None):
    cap=PoseEstimate(np.array([.5,0,cap_z]),visible=visible,source='learned:test',measurement_t=t if stamp is None else stamp)
    slot=PoseEstimate(np.array([.5,.15,.403]),visible=True,source='learned:test',measurement_t=t)
    return Obs(t,np.array([.5,0,z]),quat_to_mat(gripper_down_quat(0)),np.zeros(7),width,Estimate(cap,slot))


def start_probe(fsm):
    fsm.state=S.LIFT;fsm.p_des=np.array([.5,0,.425]);fsm.hold_pos=fsm.p_des.copy()
    fsm.retention_grasp_origin=fsm.p_des.copy()
    fsm.retention_last_ground_sample=(.8,np.array([.5,0,.423]))
    fsm._begin_probe(obs(1,z=.425,cap_z=.423,visible=False))


def start_carry(fsm):
    start_probe(fsm)
    fsm._probe_tick(obs(1.2));fsm._probe_tick(obs(1.4))
    assert fsm.retention_phase=='carrying'


def test_two_distinct_visual_samples_required_and_held_sample_never_counts_twice():
    _,fsm=setup();start_probe(fsm)
    fsm._probe_tick(obs(1.2));assert fsm.retention_phase=='probe'
    fsm._probe_tick(obs(1.22,stamp=1.2));assert len(fsm.retention_samples)==1
    fsm._probe_tick(obs(1.4));assert fsm.retention_phase=='carrying'


def test_capsule_near_hand_without_measured_rise_is_not_a_confirmed_lift():
    _,fsm=setup();start_probe(fsm)
    for t in (1.2,1.4,1.6):fsm._probe_tick(obs(t,cap_z=.423))
    assert not any(e['event']=='probe_confirmed' for e in fsm.retention_history)


def test_missing_visual_evidence_aborts_closed_and_does_not_retry_blindly():
    _,fsm=setup();start_probe(fsm)
    cmd=fsm._probe_tick(obs(4.1,visible=False))
    assert fsm.state==S.ABORT and cmd.gripper==fsm.g['closed_ctrl'] and fsm.retention_retries==0


def test_carried_reference_is_in_robot_frame_and_detects_slip_after_confirmation():
    _,fsm=setup();start_carry(fsm)
    same=obs(1.6,z=.52,cap_z=.518);assert fsm._carry_guard(same) is None
    slipped=obs(1.8,z=.52,cap_z=.508);assert fsm._carry_guard(slipped) is None
    fsm._carry_guard(obs(2,z=.52,cap_z=.508));assert fsm.state==S.ABORT


def test_carrying_visual_outage_pauses_before_timeout_and_resumes_on_agreement():
    _,fsm=setup();start_carry(fsm)
    assert fsm._carry_guard(obs(2.1,visible=False)) is not None
    assert fsm.retention_phase=='observe_hold' and fsm.state!=S.ABORT
    assert fsm._carry_guard(obs(2.2)) is None and fsm.retention_phase=='carrying'


def test_width_collapse_debounces_and_reacquires_with_finite_retry_budget():
    _,fsm=setup();start_carry(fsm)
    assert fsm._carry_guard(obs(1.6,width=0,visible=False)) is None
    fsm._carry_guard(obs(1.7,width=0,visible=False))
    assert fsm.retention_retries==1 and fsm.retention_phase=='retract_empty'
    fsm.retention_retries=2;fsm._retry_retention(obs(1.9),'empty')
    assert fsm.state==S.ABORT and fsm.retention_retries==2


def test_stale_preclose_pose_cannot_validate_a_new_grasp():
    _,fsm=setup();start_probe(fsm);fsm.retention_probe_cap=(-2,np.array([.5,0,.423]))
    fsm._probe_tick(obs(1.2));fsm._probe_tick(obs(1.4))
    assert fsm.retention_phase=='probe'


def test_disabled_profile_matches_baseline_and_policy_has_no_simulation_imports():
    cfg,fsm=setup();cfg['task'].pop('retention_feedback')
    original=TaskFSM(cfg,np.zeros(7),.4,'closed_loop')
    a,b=fsm.update(obs(0)),original.update(obs(0))
    np.testing.assert_equal(a.p_des,b.p_des);np.testing.assert_equal(a.joint_goal,b.joint_goal)
    assert a.gripper==b.gripper and fsm.stats.transitions==original.stats.transitions
    tree=ast.parse((ROOT/'src/cl_pickplace/retention_feedback.py').read_text())
    imported=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert imported==['collections','task_fsm','kinematics']
