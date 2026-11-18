from pathlib import Path
import json,numpy as np
from cl_pickplace.occlusion_gated_centering import OcclusionGatedCenterFSM
from cl_pickplace.hand_occlusion import line_of_sight_clearance,capsule_samples
from cl_pickplace.kinematics import quat_to_mat,gripper_down_quat
from cl_pickplace.task_fsm import S
from test_centering_feedback import setup,observed
from test_reobserving_centering import at_view

ROOT=Path(__file__).resolve().parents[1]


def planned():
    fsm=setup();cfg={'task':fsm.p,'gripper':fsm.g,'control':fsm.c,'perception':fsm.pc}
    for name in ('reobserving_control_protocol_v21','blind_centering_protocol_v22','confirmed_centering_protocol_v23','occlusion_gated_protocol_v24'):
        cfg['task']['visual_centering'].update(json.loads((ROOT/f'config/{name}.json').read_text())['runtime_additions'])
    result=OcclusionGatedCenterFSM(cfg,np.zeros(7),.4,'closed_loop');result.state=S.VERIFY
    result._try_centering(observed(0),'test');return result


def hand_between(fsm,t,dx,source='learned:top_cam'):
    """Displaced capsule detection with the TCP placed on the camera sight line."""
    obs=at_view(fsm,t);cap=obs.est.capsule;cap.pos=cap.pos+np.r_[dx,0,0];cap.source=source
    camera=np.asarray(fsm.centering_settings['top_cam_position_m'])
    obs.ee_pos=cap.pos+0.3*(camera-cap.pos);obs.ee_mat=quat_to_mat(gripper_down_quat(0.))
    return obs


def test_sight_line_clearance_geometry():
    points=np.array([[0.,0.,.5]])
    assert line_of_sight_clearance([0,0,1],[[0,0,0]],points)<1e-9
    assert abs(line_of_sight_clearance([0,0,1],[[0,0,0]],np.array([[.05,0,.5]]))-.05)<1e-9
    assert line_of_sight_clearance([0,0,1],[[0,0,0]],np.array([[0,0,-.1]]))>.09  # behind target ignored
    assert capsule_samples([0,0,0],[2,0,0],.015).shape==(5,3)


def test_occluded_displaced_detections_cannot_stop_the_stroke():
    fsm=planned();fsm._centering_phase(observed(1),'lower')
    fsm.centering_tick(hand_between(fsm,1.2,.02));fsm.centering_tick(hand_between(fsm,1.4,.02))
    assert fsm.centering_phase=='lower' and len(fsm.occlusion_gated)==2


def test_clear_view_displaced_detections_still_stop_the_stroke():
    fsm=planned();fsm._centering_phase(observed(1),'lower')
    for t in (1.2,1.4):
        obs=at_view(fsm,t);obs.est.capsule.pos=obs.est.capsule.pos+np.r_[.007,0,0];obs.est.capsule.source='learned:top_cam'
        obs.ee_pos=obs.ee_pos+np.r_[0,.3,0]  # hand well clear of the top-camera sight line
        fsm.centering_tick(obs)
    assert fsm.centering_failure=='capsule moved before contact' and fsm.occlusion_gated==[]


def test_gate_only_applies_during_lower_and_push():
    fsm=planned();fsm._centering_phase(observed(1),'observe')
    obs=hand_between(fsm,1.2,0.)
    assert fsm._fresh(obs.est.capsule,obs,1.)
    fsm._centering_phase(observed(1),'push')
    assert not fsm._fresh(obs.est.capsule,obs,1.)


def test_gated_evidence_still_limited_by_blind_budget():
    fsm=planned();fsm._centering_phase(observed(2),'push')
    fsm.centering_tick(hand_between(fsm,8.3,0.))
    assert fsm.centering_failure=='blind budget exhausted'
