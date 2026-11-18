"""Task-space servo: resolved-rate DLS IK producing joint position targets."""
from __future__ import annotations

from typing import Any

import numpy as np

from .kinematics import clip_twist, dls_step, orientation_error, site_jacobian
from .sim import Sim


class TaskSpaceController:
    """One bounded step toward a desired EE pose per control tick.

    The joint target is integrated on itself (not on measured q) so position actuators
    track without steady-state lag, with an anti-windup band around measured q.
    """

    def __init__(self, sim: Sim, cfg: dict[str, Any]):
        self.sim = sim
        c = cfg["control"]
        self.kp = np.array([c["Kp_pos"]] * 3 + [c["Kp_rot"]] * 3)
        self.max_lin = c["max_lin_vel"]
        self.max_ang = c["max_ang_vel"]
        self.lam = c["lambda"]
        self.k_null = c["k_null"]
        self.max_lead = c["max_target_lead_rad"]
        self.home_speed = c["home_joint_speed"]
        self.q_home = sim.home_q.copy()
        self.q_target = sim.arm_q()
        self.last_error = np.zeros(6)

    def reset(self) -> None:
        self.q_target = self.sim.arm_q()
        self.last_error = np.zeros(6)

    def _write(self, q_target: np.ndarray) -> None:
        q = self.sim.arm_q()
        q_target = np.clip(q_target, q - self.max_lead, q + self.max_lead)
        lo, hi = self.sim.jnt_range[:, 0], self.sim.jnt_range[:, 1]
        self.q_target = np.clip(q_target, lo, hi)
        self.sim.set_arm_ctrl(self.q_target)

    def step(self, p_des: np.ndarray, q_des: np.ndarray) -> np.ndarray:
        """Command one control step toward (p_des, q_des). Returns the 6D error."""
        s = self.sim
        err = np.concatenate([p_des - s.ee_pos(), orientation_error(q_des, s.ee_mat())])
        v = clip_twist(self.kp * err, self.max_lin, self.max_ang)
        J = site_jacobian(s.model, s.data, s.ee_site, s.arm_dadr)
        dq = dls_step(J, v, self.lam, s.arm_q(), self.q_home, self.k_null)
        self._write(self.q_target + dq * s.dt_control)
        self.last_error = err
        return err

    def step_joint(self, q_goal: np.ndarray) -> float:
        """Joint-space move toward q_goal at bounded speed. Returns max joint error."""
        dq = np.clip(q_goal - self.q_target, -self.home_speed * self.sim.dt_control,
                     self.home_speed * self.sim.dt_control)
        self._write(self.q_target + dq)
        return float(np.max(np.abs(q_goal - self.sim.arm_q())))

    def hold(self) -> None:
        self.sim.set_arm_ctrl(self.q_target)
