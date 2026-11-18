from pathlib import Path
import json,numpy as np
from cl_pickplace.confirmed_blind_centering import ConfirmedBlindCenterFSM
from cl_pickplace.task_fsm import S
from test_centering_feedback import setup,observed
from test_reobserving_centering import at_view
from test_bounded_blind_centering import blind

ROOT=Path(__file__).resolve().parents[1]


def planned():
    fsm=setup();cfg={'task':fsm.p,'gripper':fsm.g,'control':fsm.c,'perception':fsm.pc}
    for name in ('reobserving_control_protocol_v21','blind_centering_protocol_v22','confirmed_centering_protocol_v23'):
        cfg['task']['visual_centering'].update(json.loads((ROOT/f'config/{name}.json').read_text())['runtime_additions'])
    result=ConfirmedBlindCenterFSM(cfg,np.zeros(7),.4,'closed_loop');result.state=S.VERIFY
    result._try_centering(observed(0),'test');return result


def displaced(fsm,t,dx):
    obs=at_view(fsm,t);obs.est.capsule.pos=obs.est.capsule.pos+np.r_[dx,0,0];return obs


def test_single_outlier_before_contact_does_not_stop_lowering():
    fsm=planned();fsm._centering_phase(observed(1),'lower')
    fsm.centering_tick(displaced(fsm,1.2,.007));assert fsm.centering_phase=='lower'
    repeat=displaced(fsm,1.22,.007);repeat.est.capsule.measurement_t=1.2;repeat.est.slot.measurement_t=1.2
    fsm.centering_tick(repeat);assert fsm.centering_phase=='lower'  # same capture seen on a later control tick
    fsm.centering_tick(at_view(fsm,1.4));assert fsm.centering_phase=='lower' and fsm.capsule_alarms==[]


def test_two_consistent_displaced_detections_stop_lowering():
    fsm=planned();fsm._centering_phase(observed(1),'lower')
    fsm.centering_tick(displaced(fsm,1.2,.007));fsm.centering_tick(displaced(fsm,1.4,.008))
    assert fsm.centering_failure=='capsule moved before contact' and fsm.centering_phase=='retreat'


def test_inconsistent_displaced_detections_do_not_confirm():
    fsm=planned();fsm._centering_phase(observed(1),'lower')
    fsm.centering_tick(displaced(fsm,1.2,.007));fsm.centering_tick(displaced(fsm,1.4,-.012))
    assert fsm.centering_phase=='lower'


def test_corridor_guard_needs_confirmation_too():
    fsm=planned();fsm._centering_phase(observed(1),'push')
    fsm.centering_tick(displaced(fsm,1.2,.03));assert fsm.centering_phase=='push'
    fsm.centering_tick(displaced(fsm,1.4,.031))
    assert fsm.centering_failure=='capsule deviated from push corridor'


def test_fixture_guard_and_blind_budget_unchanged():
    fsm=planned();fsm._centering_phase(observed(1),'push')
    obs=blind(fsm,1.2);obs.est.slot.pos=obs.est.slot.pos+np.r_[.01,0,0]
    fsm.centering_tick(obs);assert fsm.centering_failure=='destination moved'
    fsm=planned();fsm._centering_phase(observed(2),'push')
    fsm.centering_tick(blind(fsm,8.2,slot_visible=False));assert fsm.centering_failure=='blind budget exhausted'
