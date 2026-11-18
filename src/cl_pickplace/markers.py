"""Debug geometry drawn into an MjvScene: viewer.user_scn or a Renderer scene."""
from __future__ import annotations

from collections import deque
from typing import Any

import mujoco
import numpy as np

from .episode import Episode
from .kinematics import quat_to_mat

G = mujoco.mjtGeom
EYE = np.eye(3)


def _rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class MarkerDrawer:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        hz = cfg["sim"]["control_hz"]
        self.trail: deque = deque(maxlen=int(cfg["viewer"]["trail_seconds"] * hz))
        self.marker_s = cfg["perturbations"]["marker_duration_s"]
        cap = cfg["scene"]["capsule"]
        self.cap_size = [cap["radius"] * 1.15, cap["half_length"], 0]
        sl = cfg["scene"]["slot"]
        self.slot_half = [sl["cavity"][0] / 2, sl["cavity"][1] / 2, sl["wall_height"] / 2]

    def reset(self) -> None:
        self.trail.clear()

    def record(self, ep: Episode) -> None:
        """Call once per control tick to extend the EE trail."""
        self.trail.append(ep.sim.ee_pos())

    @staticmethod
    def _add(scn: mujoco.MjvScene, gtype, size, pos, mat, rgba) -> mujoco.MjvGeom | None:
        if scn.ngeom >= scn.maxgeom:
            return None
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, gtype, np.asarray(size, float), np.asarray(pos, float),
                            np.asarray(mat, float).reshape(9), np.asarray(rgba, np.float32))
        scn.ngeom += 1
        return g

    @classmethod
    def _line(cls, scn, a, b, width, rgba, gtype=G.mjGEOM_CAPSULE) -> None:
        g = cls._add(scn, gtype, [0, 0, 0], [0, 0, 0], EYE, rgba)
        if g is not None:
            mujoco.mjv_connector(g, gtype, width, np.asarray(a, float), np.asarray(b, float))

    def draw(self, scn: mujoco.MjvScene, ep: Episode, show_targets: bool = True) -> None:
        hx, hy, hz = self.slot_half
        # EE trail, fading with age
        n = len(self.trail)
        for i, p in enumerate(list(self.trail)[::2]):
            a = 0.15 + 0.6 * (2 * i / max(n, 1))
            self._add(scn, G.mjGEOM_SPHERE, [0.0025, 0, 0], p, EYE, [0.2, 0.6, 1.0, a])

        if show_targets and ep.cmd.p_des is not None:
            R = quat_to_mat(ep.cmd.q_des)
            p = ep.cmd.p_des
            self._add(scn, G.mjGEOM_SPHERE, [0.009, 0, 0], p, EYE, [1.0, 0.55, 0.0, 0.45])
            for k, col in enumerate(([1, 0, 0, 0.8], [0, 1, 0, 0.8], [0, 0, 1, 0.8])):
                self._line(scn, p, p + 0.04 * R[:, k], 0.0015, col)

        if show_targets and ep.est is not None:
            c, s = ep.est.capsule, ep.est.slot
            # capsule estimate: translucent capsule along its long axis
            Rc = _rotz(c.yaw) @ np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0.0]])
            self._add(scn, G.mjGEOM_CAPSULE, self.cap_size, c.pos, Rc, [1, 1, 0, 0.35])
            # slot estimate: thin frame at the wall top
            Rs = _rotz(s.yaw)
            ctr = s.pos + [0, 0, 2 * hz]
            corners = [ctr + Rs @ np.array([sx * hx, sy * hy, 0])
                       for sx, sy in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
            for i in range(4):
                self._line(scn, corners[i], corners[(i + 1) % 4], 0.0012, [1, 1, 0, 0.8])

        # perturbation markers
        t = ep.t
        for ev in ep.perturber.events:
            if t - ev.t_start > self.marker_s:
                continue
            if ev.kind == "arm_push":
                b = ev.params["body"]
                p = ep.sim.data.xipos[b]
                f = ev.params["force"]
                self._line(scn, p - f / np.linalg.norm(f) * (0.04 + 0.004 * ev.magnitude), p,
                           0.012, [1, 0.1, 0.1, 0.9], G.mjGEOM_ARROW)
            elif ev.kind == "slot_move":
                sp = ep.sim.slot_pos()
                blink = 0.5 + 0.4 * np.sin(2 * np.pi * 4 * (t - ev.t_start))
                self._add(scn, G.mjGEOM_BOX, [hx * 1.3, hy * 1.3, 0.012], sp + [0, 0, 0.01],
                          _rotz(ep.sim.slot_yaw()), [1, 0.9, 0, 0.35 * blink])
            elif ev.kind == "object_nudge":
                cp = ep.sim.capsule_pos()
                self._add(scn, G.mjGEOM_SPHERE, [0.025, 0, 0], cp, EYE, [1, 0.3, 1, 0.25])
