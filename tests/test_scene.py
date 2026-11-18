import mujoco
import numpy as np
import pytest

from cl_pickplace.sim import Sim, name2id


@pytest.fixture(scope="module")
def sim(cfg):
    return Sim(cfg)


def test_expected_names_exist(sim):
    m, O = sim.model, mujoco.mjtObj
    for n in ["capsule", "slot", "table", "hand", "link0", "left_finger", "right_finger"]:
        assert name2id(m, O.mjOBJ_BODY, n) >= 0
    for n in ["ee_site", "capsule_site", "slot_site"]:
        assert name2id(m, O.mjOBJ_SITE, n) >= 0
    for n in [f"actuator{i}" for i in range(1, 9)]:
        assert name2id(m, O.mjOBJ_ACTUATOR, n) >= 0
    for n in ["wrist_cam", "front_cam", "top_cam"]:
        assert name2id(m, O.mjOBJ_CAMERA, n) >= 0


def test_missing_name_lists_alternatives(sim):
    with pytest.raises(KeyError, match="Available"):
        name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, "no_such_body")


def test_gripper_range_and_slot_is_mocap(sim):
    lo, hi = sim.model.actuator_ctrlrange[sim.grip_act]
    assert (lo, hi) == (0, 255)
    assert sim.slot_mocap >= 0


def test_capsule_rests_on_table(sim, cfg):
    sim.reset(np.random.default_rng(0))
    sim.step(1000)                       # 2 s
    r = cfg["scene"]["capsule"]["radius"]
    z = sim.capsule_pos()[2]
    assert abs(z - (cfg["scene"]["table_top_z"] + r)) < 2e-3
    assert sim.capsule_speed() < 1e-3


def test_capsule_fits_in_slot(cfg):
    c, s = cfg["scene"]["capsule"], cfg["scene"]["slot"]
    assert 2 * (c["half_length"] + c["radius"]) < s["cavity"][0]
    assert 2 * c["radius"] < s["cavity"][1]


def test_capsule_contact_enables_declared_rolling_friction(sim):
    assert sim.model.geom_condim[sim.capsule_geom] == 6


def test_slot_collides_with_capsule(sim, cfg):
    """Drop the capsule into the slot: it must stop on the slot floor, not fall through."""
    sim.reset(poses={"capsule": (0.45, -0.1, 0.0), "slot": (0.45, 0.15, 0.0)})
    a = sim.capsule_qadr
    sim.data.qpos[a:a + 3] = [0.45, 0.15, 0.45]
    sim.step(1000)
    floor = sim.slot_pos()[2]
    assert sim.capsule_pos()[2] == pytest.approx(floor + cfg["scene"]["capsule"]["radius"], abs=2e-3)
