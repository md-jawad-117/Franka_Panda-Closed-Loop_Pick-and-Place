"""FSM transitions under synthetic observations (no physics)."""
import numpy as np
import pytest

from cl_pickplace.kinematics import quat_to_mat
from cl_pickplace.perception import Estimate, PoseEstimate
from cl_pickplace.task_fsm import Obs, S, TaskFSM

HOME = np.zeros(7)
TABLE = 0.4


def make_est(cap=(0.45, -0.1), slot=(0.45, 0.15), cap_yaw=0.0, slot_yaw=0.0):
    return Estimate(PoseEstimate(np.array([cap[0], cap[1], TABLE + 0.008]), cap_yaw, True),
                    PoseEstimate(np.array([slot[0], slot[1], TABLE + 0.003]), slot_yaw, True))


class Driver:
    """Teleports the 'EE' exactly onto the commanded pose each tick."""

    def __init__(self, fsm, est, width=0.013):
        self.fsm, self.est, self.t = fsm, est, 0.0
        self.ee = np.array([0.45, 0.0, 0.62])
        self.R = quat_to_mat(np.array([0, 1, 0, 0.0]))
        self.width = width
        self.carry = False

    def obs(self):
        if self.carry:   # capsule follows the TCP when carried
            self.est.capsule.pos = self.ee.copy()
        return Obs(self.t, self.ee.copy(), self.R.copy(), HOME.copy(), self.width, self.est)

    def tick(self, follow=True):
        cmd = self.fsm.update(self.obs())
        if follow and cmd.p_des is not None:
            self.ee = cmd.p_des.copy()
            self.R = quat_to_mat(cmd.q_des)
        self.t += 0.02
        return cmd

    def run_until(self, state, max_ticks=500, **kw):
        for _ in range(max_ticks):
            if self.fsm.state == state:
                return True
            self.tick(**kw)
        return self.fsm.state == state


@pytest.fixture
def fsm(cfg):
    return TaskFSM(cfg, HOME, TABLE, "closed_loop")


def test_nominal_sequence(fsm):
    d = Driver(fsm, make_est())
    assert d.run_until(S.APPROACH_OBJECT)
    assert d.run_until(S.GRASP)
    assert d.tick().gripper == fsm.g["closed_ctrl"]
    assert d.run_until(S.LIFT)
    d.carry = True
    assert d.run_until(S.TRANSPORT)
    assert d.run_until(S.DESCEND_SLOT)
    assert d.run_until(S.RELEASE)
    d.carry = False
    assert d.run_until(S.DONE)
    names = [b for _, a, b, _ in fsm.stats.transitions]
    assert names == ["APPROACH_OBJECT", "DESCEND_OBJECT", "GRASP", "LIFT", "TRANSPORT",
                     "DESCEND_SLOT", "RELEASE", "RETREAT", "DONE"]


def test_object_moved_during_descent_triggers_realign(fsm):
    d = Driver(fsm, make_est())
    assert d.run_until(S.DESCEND_OBJECT)
    d.est.capsule.pos[:2] += [0.03, 0.0]      # object nudged 3 cm
    d.tick(follow=False)
    assert fsm.state == S.APPROACH_OBJECT
    assert fsm.stats.num_realigns == 1


def test_slot_moved_during_descent_returns_to_transport(fsm):
    d = Driver(fsm, make_est())
    d.run_until(S.LIFT)
    d.carry = True
    assert d.run_until(S.DESCEND_SLOT)
    d.est.slot.pos[:2] += [0.0, 0.05]
    d.tick(follow=False)
    assert fsm.state == S.TRANSPORT


def test_failed_grasp_regrasps(fsm):
    d = Driver(fsm, make_est(), width=0.0)    # fingers closed on nothing
    d.run_until(S.LIFT)
    assert d.run_until(S.APPROACH_OBJECT)
    assert fsm.stats.num_regrasps == 1


def test_drop_during_transport(fsm):
    d = Driver(fsm, make_est())
    d.run_until(S.LIFT)
    d.carry = True
    assert d.run_until(S.TRANSPORT)
    d.carry = False
    d.est.capsule.pos = np.array([0.45, 0.0, TABLE + 0.008])   # fell on the table
    d.tick(follow=False)
    assert fsm.state == S.APPROACH_OBJECT


def test_open_loop_ignores_moved_slot(cfg):
    fsm = TaskFSM(cfg, HOME, TABLE, "open_loop")
    d = Driver(fsm, make_est())
    d.run_until(S.LIFT)
    d.carry = True
    d.run_until(S.TRANSPORT)
    d.est.slot.pos[:2] += [0.0, 0.05]        # open loop never re-reads the slot
    assert d.run_until(S.RELEASE)
    assert d.ee[1] == pytest.approx(0.15, abs=1e-6)
    assert fsm.stats.num_realigns == 0


def test_closed_loop_does_not_release_on_moving_slot(fsm):
    d = Driver(fsm, make_est())
    d.run_until(S.LIFT)
    d.carry = True
    assert d.run_until(S.DESCEND_SLOT)
    for _ in range(30):                       # slot creeping at 4 cm/s (8 mm per 0.2 s), below realign
        d.est.slot.pos[1] += 0.0008
        d.tick()
        assert fsm.state != S.RELEASE
