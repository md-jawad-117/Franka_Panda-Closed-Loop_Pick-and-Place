from dataclasses import replace
import mujoco
import numpy as np
import pytest

from cl_pickplace.evaluation import PlacementEvaluator, PlacementSample, SettlingWindow, capsule_bounds
from cl_pickplace.sim import Sim


def bounds(x=0, y=0, z=.008, yaw=0, slot_yaw=0):
    c, s = np.cos(slot_yaw), np.sin(slot_yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return capsule_bounds([x,y,z], [np.cos(yaw),np.sin(yaw),0], np.zeros(3), R,
                          .008, .015, [.060,.035])


@pytest.mark.parametrize('yaw', [0, np.pi, -np.pi])
def test_capsule_axis_symmetry_and_valid_shallow_fixture(yaw):
    _, _, clearance, gap = bounds(yaw=yaw)
    assert clearance == pytest.approx(.007)
    assert gap == pytest.approx(0)
    # Diameter 16 mm exceeds 10 mm wall height: this is still a valid shallow placement.


@pytest.mark.parametrize('x,y,yaw', [(.010,0,0), (0,.014,0), (0,0,np.pi/2)])
def test_center_inside_is_not_whole_capsule_inside(x,y,yaw):
    assert bounds(x=x,y=y,yaw=yaw)[2] < 0


def test_fixture_rotation_is_respected():
    assert bounds(yaw=np.pi/2,slot_yaw=np.pi/2)[2] > 0
    assert bounds(yaw=0,slot_yaw=np.pi/2)[2] < 0


@pytest.mark.parametrize('height', [-.05, .030])
def test_below_floor_and_floating_object_have_invalid_support_height(height):
    assert abs(bounds(z=height)[3]) > .001


def sample(t, **kwargs):
    s = PlacementSample(t, True, True, True, True, .007, 0, 0,
                        np.array([0.,0.,.008]), np.array([1.,0.,0.]))
    return replace(s, **kwargs)


def window():
    return SettlingWindow(.4, .001, np.radians(2), .031)


def test_stability_requires_elapsed_covered_window():
    w = window()
    for i in range(20):
        w.add(sample(i*.02))
    assert not w.stable()
    w.add(sample(.4))
    assert w.stable()


@pytest.mark.parametrize('bad', ['contained','supported','released','slow'])
def test_one_invalid_tick_breaks_settling(bad):
    w = window()
    for i in range(21):
        w.add(sample(i*.02, **({bad: False} if i == 10 else {})))
    assert not w.stable()


def test_sampling_gap_and_pose_drift_are_not_stability():
    w = window()
    w.add(sample(0))
    w.add(sample(.4))
    assert not w.stable()
    w = window()
    for i in range(21):
        w.add(sample(i*.02, local_pos=np.array([i*.0001,0,.008])))
    assert not w.stable()


@pytest.fixture
def supported_sim(cfg):
    s = Sim(cfg)
    s.reset(poses={'capsule':(.45,-.1,0), 'slot':(.45,.15,0)})
    s.set_capsule_pose(.45,.15,0)
    s.data.qpos[s.capsule_qadr+2] = s.slot_pos()[2]+cfg['scene']['capsule']['radius']
    mujoco.mj_forward(s.model,s.data)
    return s


def test_physics_supported_released_capsule_passes(cfg,supported_sim):
    ev = PlacementEvaluator(supported_sim,cfg)
    for _ in range(25):
        supported_sim.step(supported_sim.n_substeps)
        ev.observe(True)
    result = ev.result(False,'DONE')
    assert result['success'], result
    assert result['minimum_xy_clearance_m'] > 0


def test_closed_gripper_and_timeout_cannot_pass(cfg,supported_sim):
    ev = PlacementEvaluator(supported_sim,cfg)
    for _ in range(25):
        supported_sim.step(supported_sim.n_substeps)
        ev.observe(False)
    assert not ev.result(False,'DONE')['success']
    assert not ev.result(True,'DONE')['success']


def test_floor_contact_is_required_even_at_valid_geometric_height(cfg,supported_sim):
    # No mj_forward/mj_step after clearing contacts: an isolated evaluator negative case.
    supported_sim.data.ncon = 0
    ev = PlacementEvaluator(supported_sim,cfg)
    assert not ev.observe(True).supported
