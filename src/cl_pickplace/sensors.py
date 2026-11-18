"""Sensor boundary: RGB-D and calibrated camera transforms, without target poses."""
from dataclasses import dataclass
from contextlib import nullcontext

import mujoco
import numpy as np


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth: np.ndarray
    world_position: np.ndarray
    world_rotation: np.ndarray
    fovy_deg: float
    timestamp: float


class RGBDSensor:
    """MuJoCo sensor backend. Camera transforms represent calibrated fixed cameras
    or wrist-camera forward kinematics, not observed object/fixture body poses.
    """
    def __init__(self, sim, width, height, hide_sites=False, synchronize=False):
        self._sim = sim
        self.renderer = mujoco.Renderer(sim.model, height, width)
        self.opt = mujoco.MjvOption()
        if hide_sites:
            self.opt.sitegroup[:] = 0
        self.synchronize = synchronize
        self.capture_context = nullcontext

    def _scene(self, cid):
        self.renderer.update_scene(self._sim.data, camera=cid, scene_option=self.opt)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False

    def capture(self, name):
        with self.capture_context():
            return self._capture(name)

    def _capture(self, name):
        s = self._sim
        if self.synchronize:
            mujoco.mj_forward(s.model, s.data)
        cid = s.cams[name]
        self._scene(cid)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            self._scene(cid)
            depth = self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        return RGBDFrame(rgb, depth, s.data.cam_xpos[cid].copy(),
                         s.data.cam_xmat[cid].reshape(3, 3).copy(), float(s.model.cam_fovy[cid]), s.time)

    def close(self):
        self.renderer.close()
