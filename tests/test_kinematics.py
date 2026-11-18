import numpy as np
import pytest

from cl_pickplace.controller import TaskSpaceController
from cl_pickplace.kinematics import (closest_symmetric_yaw, dls_step, gripper_down_quat,
                                     orientation_error, quat_to_mat, wrap_half_pi)
from cl_pickplace.sim import Sim


def test_gripper_down_quat_points_down():
    for yaw in np.linspace(-np.pi, np.pi, 7):
        R = quat_to_mat(gripper_down_quat(yaw))
        assert np.allclose(R[:, 2], [0, 0, -1], atol=1e-9)
        assert np.allclose(R[:2, 0], [np.cos(yaw), np.sin(yaw)], atol=1e-9)


def test_orientation_error_zero_and_sign():
    q = gripper_down_quat(0.3)
    assert np.allclose(orientation_error(q, quat_to_mat(q)), 0, atol=1e-9)
    e = orientation_error(gripper_down_quat(0.4), quat_to_mat(gripper_down_quat(0.3)))
    assert np.allclose(e, [0, 0, 0.1], atol=1e-6)   # world-frame rotation about +z


def test_symmetric_yaw():
    assert wrap_half_pi(np.pi) == pytest.approx(0.0, abs=1e-9)
    assert wrap_half_pi(np.radians(100)) == pytest.approx(np.radians(-80))
    assert closest_symmetric_yaw(np.radians(170), 0.0) == pytest.approx(np.radians(-10))


def test_dls_matches_pinv_for_small_lambda():
    rng = np.random.default_rng(0)
    J = rng.normal(size=(6, 7))
    v = rng.normal(size=6)
    assert np.allclose(dls_step(J, v, 1e-6), np.linalg.pinv(J) @ v, atol=1e-6)


def test_ik_reaches_20_random_targets(cfg):
    """M2 acceptance: position error < 3 mm and orientation error < 2 deg."""
    sim = Sim(cfg)
    sim.reset(np.random.default_rng(0))
    ctrl = TaskSpaceController(sim, cfg)
    rng = np.random.default_rng(42)
    worst_p, worst_r = 0.0, 0.0
    for _ in range(20):
        p = np.array([rng.uniform(0.35, 0.6), rng.uniform(-0.25, 0.3), rng.uniform(0.42, 0.62)])
        q = gripper_down_quat(rng.uniform(-np.pi / 2, np.pi / 2))
        for _ in range(200):                    # 4 s per target
            ctrl.step(p, q)
            sim.step(sim.n_substeps)
        worst_p = max(worst_p, float(np.linalg.norm(p - sim.ee_pos())))
        worst_r = max(worst_r, float(np.degrees(np.linalg.norm(orientation_error(q, sim.ee_mat())))))
    assert worst_p < 3e-3, worst_p
    assert worst_r < 2.0, worst_r
