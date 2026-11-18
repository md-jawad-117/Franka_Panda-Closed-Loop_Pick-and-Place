from pathlib import Path
import json,numpy as np
from cl_pickplace.config import load_config
from cl_pickplace.settled_inspection import InspectionFSM
from cl_pickplace.task_fsm import S,Obs
from cl_pickplace.perception import PoseEstimate,Estimate
from cl_pickplace.kinematics import quat_to_mat,gripper_down_quat

ROOT=Path(__file__).resolve().parents[1]


def setup():
    cfg=load_config(ROOT/'config/learned_reacquisition_v4.yaml')
    cfg['task']['settled_inspection']=json.loads((ROOT/'config/inspection_protocol_v14.json').read_text())['inspection_settings']
    fsm=InspectionFSM(cfg,np.zeros(7),.4,'closed_loop');fsm.state=S.VERIFY
    return fsm


def sample(t,visible=True,x=.5,stamp=None):
    stamp=t if stamp is None else stamp
    cap=PoseEstimate(np.array([x,.15,.411]),visible=visible,source='learned:test',measurement_t=stamp,axis=np.array([1.,0,0]),observation={'capture_t':t})
    slot=PoseEstimate(np.array([.5,.15,.403]),visible=visible,source='learned:test',measurement_t=stamp,observation={'capture_t':t})
    return Obs(t,np.array([.5,.15,.55]),quat_to_mat(gripper_down_quat(0)),np.zeros(7),.08,Estimate(cap,slot))


def confirm(fsm):
    for t in (0,.2,.4,.6):fsm.verification_tick(sample(t))
    assert fsm.inspection_confirmed_t==.6


def test_no_evidence_while_robot_moves_and_no_cached_pair_is_a_new_sample():
    fsm=setup();fsm.verification_tick(sample(0));fsm.inspection_pose[0][0]+=.03
    for t in (.2,.4,.6):fsm.verification_tick(sample(t))
    assert not fsm.inspection_samples and fsm.inspection_confirmed_t is None
    fsm=setup();fsm.verification_tick(sample(0));fsm.verification_tick(sample(.2))
    fsm.verification_tick(sample(.4,stamp=.2));assert not fsm.inspection_samples


def test_three_distinct_settled_captures_confirm_without_relaxing_containment():
    fsm=setup();confirm(fsm)
    assert fsm.verification_status=='inside' and len(fsm.inspection_samples)==3
    fsm=setup()
    for t in (0,.2,.4,.6):fsm.verification_tick(sample(t,x=.535))
    assert fsm.placement_retries==1 and fsm.inspection_confirmed_t is None


def test_short_missing_observation_is_unknown_and_confirmation_expires():
    fsm=setup();confirm(fsm);fsm.recovery_settings['monitor_s']=.5
    fsm.verification_tick(sample(.8,visible=False));assert fsm.state==S.DONE
    assert fsm.verification_status=='unknown' and fsm.inspection_confirmed_t==.6
    fsm.verification_tick(sample(1.3,visible=False));assert fsm.state==S.VERIFY


def test_confirmed_view_is_held_and_new_contradictory_evidence_triggers_retry():
    fsm=setup();confirm(fsm);pose=fsm.inspection_pose[0].copy()
    for t in (.8,1.,1.2):fsm.verification_tick(sample(t,x=.535))
    assert fsm.placement_retries==1 and fsm.state==S.APPROACH_OBJECT
    assert np.isfinite(pose).all()
