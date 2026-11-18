from pathlib import Path
import json,numpy as np
from cl_pickplace.bounded_blind_centering import BoundedBlindCenterFSM
from cl_pickplace.task_fsm import S
from test_centering_feedback import setup,observed
from test_reobserving_centering import at_view

ROOT=Path(__file__).resolve().parents[1]


def revised():
    fsm=setup();cfg={'task':fsm.p,'gripper':fsm.g,'control':fsm.c,'perception':fsm.pc}
    for name in ('reobserving_control_protocol_v21','blind_centering_protocol_v22'):
        cfg['task']['visual_centering'].update(json.loads((ROOT/f'config/{name}.json').read_text())['runtime_additions'])
    result=BoundedBlindCenterFSM(cfg,np.zeros(7),.4,'closed_loop');result.state=S.VERIFY
    return result


def blind(fsm,t,slot_visible=True):
    obs=at_view(fsm,t,False);obs.est.capsule.measurement_t=0
    if not slot_visible:obs.est.slot.visible=False;obs.est.slot.measurement_t=0
    return obs


def planned():
    fsm=revised();fsm._try_centering(observed(0),'test');return fsm


def test_plan_still_requires_fresh_learned_pair():
    fsm=revised();obs=observed(0);obs.est.capsule.visible=False
    assert fsm._try_centering(obs,'test') is None and not fsm.centering_active


def test_observe_timeout_within_budget_proceeds_to_lower_not_done():
    fsm=planned();fsm._centering_phase(observed(0),'observe')
    fsm.centering_tick(blind(fsm,1.05))
    assert fsm.centering_phase=='lower' and fsm.state!=S.DONE
    assert any(r['event']=='blind_proceed' for r in fsm.centering_history)


def test_observe_timeout_beyond_budget_still_retracts():
    fsm=planned();fsm._centering_phase(observed(8.5),'observe')
    fsm.centering_tick(blind(fsm,9.6))
    assert fsm.centering_phase=='retreat'


def test_missing_evidence_during_lower_and_push_tolerated_only_within_budget():
    fsm=planned();fsm._centering_phase(observed(1),'lower')
    fsm.centering_tick(blind(fsm,1.2,slot_visible=False));assert fsm.centering_phase=='lower'
    fsm._centering_phase(observed(2),'push')
    fsm.centering_tick(blind(fsm,2.2,slot_visible=False));assert fsm.centering_phase=='push'
    fsm.centering_tick(blind(fsm,8.2,slot_visible=False));assert fsm.centering_phase=='retreat'
    assert fsm.centering_failure=='blind budget exhausted'


def test_fresh_contradicting_destination_evidence_stops_blind_stroke():
    fsm=planned();fsm._centering_phase(observed(1),'push')
    obs=blind(fsm,1.2);obs.est.slot.pos=obs.est.slot.pos+np.r_[.01,0,0]
    fsm.centering_tick(obs);assert fsm.centering_failure=='destination moved'
    fsm=planned();fsm._centering_phase(observed(1),'lower')
    obs=blind(fsm,1.2);obs.est.slot.observation['motion_pending']=True
    fsm.centering_tick(obs);assert fsm.centering_failure=='destination motion pending'


def test_fresh_capsule_displacement_before_contact_stops_lowering():
    fsm=planned();fsm._centering_phase(observed(1),'lower')
    obs=at_view(fsm,1.2);obs.est.capsule.pos=obs.est.capsule.pos+np.r_[.006,0,0]
    fsm.centering_tick(obs);assert fsm.centering_failure=='capsule moved before contact'


def test_fresh_pair_renews_evidence_clock():
    fsm=planned();fsm._centering_phase(observed(6),'push')
    fsm.centering_tick(at_view(fsm,6.0))
    assert fsm.centering_evidence_t==6.0
    fsm.centering_tick(blind(fsm,9.0,slot_visible=False))
    assert fsm.centering_failure!='blind budget exhausted'
