from pathlib import Path
import json,numpy as np
from cl_pickplace.centering_feedback import CenteringFSM
from cl_pickplace.task_fsm import S
from test_settled_inspection import sample
from cl_pickplace.config import load_config

ROOT=Path(__file__).resolve().parents[1]


def setup():
    cfg=load_config(ROOT/'config/learned_reacquisition_v4.yaml')
    cfg['task']['settled_inspection']=json.loads((ROOT/'config/inspection_protocol_v14.json').read_text())['inspection_settings']
    protocol=json.loads((ROOT/'config/centering_control_protocol_v20.json').read_text())
    prior=json.loads((ROOT/'config/centering_assay_protocol_v19.json').read_text())
    cfg['task']['visual_centering']={**protocol['runtime_settings'],'plan_settings':prior['settings'],'motion':prior['motion']}
    fsm=CenteringFSM(cfg,np.zeros(7),.4,'closed_loop');fsm.state=S.VERIFY
    return fsm


def observed(t,x=.51):
    obs=sample(t,x=x);obs.est.slot.quality['estimator']='observed_support_fixture_v17'
    return obs


def test_centering_is_a_distinct_physical_action_not_a_regrasp_or_success_flag():
    fsm=setup();cmd=fsm._inspection_retry(observed(0))
    assert fsm.state==S.EXTRACT and fsm.centering_active and fsm.centering_attempts==1
    assert fsm.placement_retries==0 and fsm.stats.num_regrasps==0 and cmd.gripper==fsm.g['closed_ctrl']


def test_lost_destination_stops_push_and_retracts_before_reinspection():
    fsm=setup();fsm._try_centering(observed(0),'test');fsm._centering_phase(observed(0),'push')
    obs=observed(.2);obs.est.slot.observation['motion_pending']=True
    cmd=fsm.centering_tick(obs)
    assert fsm.centering_phase=='retreat' and cmd.p_des[2]>=.55 and fsm.state!=S.DONE


def test_visual_goal_stops_motion_and_only_returns_to_verification():
    fsm=setup();fsm._try_centering(observed(0),'test');fsm._centering_phase(observed(0),'push')
    obs=observed(.2,x=.5);fsm.centering_tick(obs)
    assert not any(b=='DONE' for _,_,b,_ in fsm.stats.transitions)
    assert fsm.centering_phase=='retreat' or fsm.state==S.VERIFY


def test_stable_uncertain_clearance_can_recenter_but_centre_noise_cannot():
    fsm=setup()
    for t in (0,.2,.4,.6,.8):fsm.verification_tick(observed(t,x=.506))
    assert fsm.centering_attempts==1 and fsm.centering_active
    fsm=setup()
    for t in (0,.2,.4,.6,.8):fsm.verification_tick(observed(t,x=.5))
    assert fsm.centering_attempts==0


def test_missing_support_evidence_or_exhausted_budget_cannot_launch_push():
    fsm=setup();obs=observed(0);obs.est.slot.quality.clear()
    assert fsm._try_centering(obs,'test') is None
    obs=observed(0);fsm.centering_attempts=2
    assert fsm._try_centering(obs,'test') is None
