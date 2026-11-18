import copy
import numpy as np
import pytest

from cl_pickplace.episode import Episode
from cl_pickplace.perturbations import KINDS


def episode(cfg, mode='closed_loop', perturb='all', seed=3):
    ep = Episode(cfg, mode=mode, perception_mode='ground_truth', perturb=perturb)
    ep.reset(seed)
    return ep


def test_paired_timed_plans_and_delivery_independent_of_policy(cfg):
    a, b = episode(cfg), episode(cfg, 'open_loop')
    assert a.perturber.manifest() == b.perturber.manifest()
    for ep in (a,b):
        while ep.t < 5.2:
            ep.step()
    assert a.perturber.delivery() == b.perturber.delivery()
    assert len(a.perturber.events) == 3
    for event, planned in zip(sorted(a.perturber.events,key=lambda e:e.plan_index), a.perturber.schedule):
        assert 0 <= event.t_start-planned.start_s < a.sim.model.opt.timestep+1e-9


@pytest.mark.parametrize('kind', KINDS)
def test_single_and_combined_runs_share_kind_parameters(cfg,kind):
    combined, single = episode(cfg), episode(cfg, perturb=kind)
    plan = next(p for p in combined.perturber.schedule if p.kind == kind)
    assert plan.description() == single.perturber.schedule[0].description()


def test_object_force_does_not_rewrite_position_or_velocity(cfg):
    ep = episode(cfg, perturb='none')
    q, v = ep.sim.data.qpos.copy(), ep.sim.data.qvel.copy()
    ep.perturber.nudge_object(ep.t,.1,.08)
    ep.perturber.apply(ep.t)
    np.testing.assert_array_equal(ep.sim.data.qpos,q)
    np.testing.assert_array_equal(ep.sim.data.qvel,v)
    assert np.linalg.norm(ep.sim.data.xfrc_applied[ep.sim.capsule_body,:3]) == pytest.approx(.1)
    ep.sim.step(10)
    assert not np.array_equal(ep.sim.data.qvel,v)
    ep.perturber.apply(ep.t+.1)
    assert np.linalg.norm(ep.sim.data.xfrc_applied[ep.sim.capsule_body,:3]) == 0


def test_overlapping_pushes_aggregate_and_clean_up(cfg):
    ep = episode(cfg,perturb='none')
    a=ep.perturber.push_arm(0,10,.2,link='link4')
    b=ep.perturber.push_arm(0,20,.2,link='link4')
    ep.perturber.apply(0)
    np.testing.assert_allclose(ep.sim.data.xfrc_applied[a.params['body'],:3],
                               np.array(a.params['force'])+b.params['force'])
    ep.perturber.apply(.3)
    assert not ep.sim.data.xfrc_applied.any()


def test_undelivered_phase_events_are_not_silently_finished(cfg):
    c=copy.deepcopy(cfg)
    c['perturbations']['schedule_mode']='state'
    ep=episode(c)
    assert not ep.perturber.finished(100,.4)
    assert len(ep.perturber.schedule)==3


def test_episode_timeout_also_applies_after_done(cfg):
    from cl_pickplace.task_fsm import S
    c=copy.deepcopy(cfg)
    c['task']['episode_timeout_s']=.1
    ep=episode(c)
    ep.fsm.state=S.DONE
    while not ep.done:
        ep.step()
    assert ep.result['timeout']
    assert not ep.result['success']


def test_repeated_fixture_move_has_no_jump_at_start(cfg):
    ep=episode(cfg,perturb='none')
    ep.perturber.set_schedule([('slot_move','TRANSPORT',0),('slot_move','DESCEND_SLOT',0)])
    ep.perturber._fire(0,0,'TRANSPORT')
    ep.perturber.apply(1)
    before=ep.sim.data.mocap_pos[ep.sim.slot_mocap].copy()
    ep.perturber._fire(1,1,'DESCEND_SLOT')
    ep.perturber.apply(1)
    np.testing.assert_array_equal(ep.sim.data.mocap_pos[ep.sim.slot_mocap],before)


def test_height_trigger_fires_during_descent_before_release(cfg):
    c=copy.deepcopy(cfg)
    c['perturbations']['scheduled']['slot_move']['trigger_height_by_state']={'DESCEND_SLOT':.01}
    ep=episode(c,perturb='none')
    ep.perturber.set_schedule([('slot_move','DESCEND_SLOT',0)])
    p=ep.perturber
    p.update_schedule(0,'DESCEND_SLOT',.1,True,ee_above_place=.02)
    assert not p.events
    p.update_schedule(.02,'DESCEND_SLOT',.1,True,ee_above_place=.009)
    assert len(p.events)==1
    assert p.events[0].state=='DESCEND_SLOT'


def test_manual_fixture_retarget_preserves_current_pose_and_cancels_old_path(cfg):
    ep=episode(cfg,perturb='none');p=ep.perturber;s=ep.sim
    first=p.move_slot(0,.04,.5,.5);p.apply(.2)
    position=s.data.mocap_pos[s.slot_mocap].copy();quaternion=s.data.mocap_quat[s.slot_mocap].copy()
    second=p.move_slot(.2,.04,-.3,.5)
    assert [e for e in p.active if e.kind=='slot_move']==[second]
    assert first.t_end==.2 and first.params['interrupted_by']=='manual_slot_move'
    p.apply(.2)
    np.testing.assert_allclose(s.data.mocap_pos[s.slot_mocap],position,atol=1e-14)
    np.testing.assert_allclose(s.data.mocap_quat[s.slot_mocap],quaternion,atol=1e-14)
    p.apply(.8)
    np.testing.assert_allclose(s.data.mocap_pos[s.slot_mocap][:2],second.params['to'][:2])
    assert not p.active and p.finished(1.3,.4)


def test_many_manual_moves_have_one_stage_writer_and_preserve_arm_force(cfg):
    ep=episode(cfg,perturb='none');p=ep.perturber
    force=p.push_arm(0,10,1,link='link4')
    for i in range(100):
        p.move_slot(i*.001,.03,.2,.5);p.apply(i*.001)
        assert sum(e.kind=='slot_move' for e in p.active)==1
        assert force in p.active
    assert sum('interrupted_at_s' in e.params for e in p.events)==99


def test_scheduled_fixture_move_replaces_an_existing_manual_stage_command(cfg):
    ep=episode(cfg,perturb='slot_move');p=ep.perturber
    old=p.move_slot(0,.03,.2,1);p.apply(.2)
    before=ep.sim.data.mocap_pos[ep.sim.slot_mocap].copy()
    p._fire(0,.2,'');p.apply(.2)
    assert sum(e.kind=='slot_move' for e in p.active)==1
    assert old.params['interrupted_by']=='scheduled_slot_move'
    np.testing.assert_allclose(ep.sim.data.mocap_pos[ep.sim.slot_mocap],before)
