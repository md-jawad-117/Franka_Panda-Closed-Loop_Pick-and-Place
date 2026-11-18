"""Model building (MjSpec), name resolution, reset and scene randomization."""
from __future__ import annotations

import logging
from typing import Any

import mujoco
import numpy as np

from .config import resolve
from .kinematics import axis_yaw_from_mat, lookat_quat, quat_mul, quat_from_axis_angle, wrap_pi

log = logging.getLogger(__name__)

ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATORS = [f"actuator{i}" for i in range(1, 8)]
GRIPPER_ACTUATOR = "actuator8"
FINGER_JOINTS = ["finger_joint1", "finger_joint2"]
ARM_BODIES = [f"link{i}" for i in range(0, 8)] + ["hand", "left_finger", "right_finger"]


def name2id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    """Resolve a name to an id, failing with a list of the available names."""
    i = mujoco.mj_name2id(model, obj, name)
    if i < 0:
        n = {
            mujoco.mjtObj.mjOBJ_BODY: model.nbody, mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
            mujoco.mjtObj.mjOBJ_SITE: model.nsite, mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
            mujoco.mjtObj.mjOBJ_CAMERA: model.ncam, mujoco.mjtObj.mjOBJ_GEOM: model.ngeom,
        }.get(obj, 0)
        names = [mujoco.mj_id2name(model, obj, k) for k in range(n)]
        raise KeyError(f"{obj.name} '{name}' not found. Available: {names}")
    return i


def _quat_xyz(axis: list[float], angle: float) -> np.ndarray:
    return quat_from_axis_angle(np.asarray(axis, float), angle)


def build_spec(cfg: dict[str, Any]) -> mujoco.MjSpec:
    sc = cfg["scene"]
    scene = mujoco.MjSpec.from_file(str(resolve(cfg["sim"]["scene_file"])))
    panda_path = resolve(cfg["sim"]["panda_file"])
    if not panda_path.exists():
        raise FileNotFoundError(
            f"{panda_path} not found. Run: python scripts/fetch_assets.py")
    panda = mujoco.MjSpec.from_file(str(panda_path))

    # The Menagerie keyframe only covers the robot; drop it so the merged model compiles.
    for key in list(panda.keys):
        panda.delete(key)

    # Menagerie ships a tracking light; the scene has its own lights (avoids double shadows).
    for light in list(panda.lights):
        panda.delete(light)

    hand = panda.body("hand")
    hand.add_site(name="ee_site", pos=[0, 0, cfg["sim"]["ee_site_offset"]],
                  size=[0.004, 0.004, 0.004], rgba=[1, 0.6, 0, 0.0], group=4)
    wc = cfg["sim"]["wrist_cam"]
    # Camera looks along its -z. Flip so it looks along hand +z, then tilt toward the TCP.
    q = quat_mul(_quat_xyz([1, 0, 0], np.pi), _quat_xyz([0, 1, 0], np.radians(wc["tilt_deg"])))
    hand.add_camera(name="wrist_cam", pos=wc["pos"], quat=q, fovy=wc["fovy"])

    if cfg["sim"]["gravity_compensation"]:
        for b in panda.bodies:
            b.gravcomp = 1.0

    scene.attach(panda, site=scene.site("robot_mount"), prefix="")

    scene.option.timestep = cfg["sim"]["timestep"]

    for name, c in cfg["sim"].get("cameras", {}).items():
        cam = scene.camera(name)
        cam.pos = c["pos"]
        cam.quat = lookat_quat(c["pos"], c["lookat"])
        cam.fovy = c["fovy"]

    # Capsule from config.
    c = sc["capsule"]
    g = scene.geom("capsule")
    g.size = [c["radius"], c["half_length"], 0]
    g.mass = c["mass"]
    g.friction = c["friction"]
    g.condim = c.get('contact_dim', 4)

    # Slot walls from config. Body origin = table surface under the cavity center.
    s = sc["slot"]
    cx, cy = s["cavity"][0] / 2, s["cavity"][1] / 2
    t, h, bt = s["wall_thickness"], s["wall_height"], s["bottom_thickness"]
    slot = scene.body("slot")
    bottom = scene.geom("slot_bottom")
    bottom.size = [cx + t, cy + t, bt / 2]
    bottom.pos = [0, 0, bt / 2]
    mat = "slot"
    zc = bt + h / 2
    for name, size, pos in [
        ("slot_wall_px", [t / 2, cy + t, h / 2], [cx + t / 2, 0, zc]),
        ("slot_wall_nx", [t / 2, cy + t, h / 2], [-cx - t / 2, 0, zc]),
        ("slot_wall_py", [cx, t / 2, h / 2], [0, cy + t / 2, zc]),
        ("slot_wall_ny", [cx, t / 2, h / 2], [0, -cy - t / 2, zc]),
    ]:
        slot.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos,
                      material=mat, contype=1, conaffinity=1)
    scene.site("slot_site").pos = [0, 0, bt]
    return scene


class Sim:
    """Owns model and data. All ids are resolved once here."""

    def __init__(self, cfg: dict[str, Any], spec: mujoco.MjSpec | None = None):
        self.cfg = cfg
        self.spec = build_spec(cfg) if spec is None else spec
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        O = mujoco.mjtObj

        self.arm_jnt = [name2id(m, O.mjOBJ_JOINT, n) for n in ARM_JOINTS]
        self.arm_qadr = np.array([m.jnt_qposadr[j] for j in self.arm_jnt])
        self.arm_dadr = np.array([m.jnt_dofadr[j] for j in self.arm_jnt])
        self.arm_act = np.array([name2id(m, O.mjOBJ_ACTUATOR, n) for n in ARM_ACTUATORS])
        self.grip_act = name2id(m, O.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR)
        self.finger_qadr = np.array(
            [m.jnt_qposadr[name2id(m, O.mjOBJ_JOINT, n)] for n in FINGER_JOINTS])
        self.jnt_range = m.jnt_range[self.arm_jnt].copy()
        self.ctrl_range = m.actuator_ctrlrange[self.arm_act].copy()

        self.ee_site = name2id(m, O.mjOBJ_SITE, "ee_site")
        self.capsule_body = name2id(m, O.mjOBJ_BODY, "capsule")
        self.capsule_geom = name2id(m, O.mjOBJ_GEOM, "capsule")
        cj = name2id(m, O.mjOBJ_JOINT, "capsule_free")
        self.capsule_qadr = m.jnt_qposadr[cj]
        self.capsule_dadr = m.jnt_dofadr[cj]
        self.slot_body = name2id(m, O.mjOBJ_BODY, "slot")
        self.slot_mocap = m.body_mocapid[self.slot_body]
        self.slot_site = name2id(m, O.mjOBJ_SITE, "slot_site")
        self.cams = {n: name2id(m, O.mjOBJ_CAMERA, n)
                     for n in ("wrist_cam", "front_cam", "top_cam")}
        self.link_bodies = {n: name2id(m, O.mjOBJ_BODY, n) for n in ARM_BODIES}

        self.home_q = np.array(cfg["sim"]["home_qpos"], float)
        self.n_substeps = int(round(cfg["sim"]["physics_hz"] / cfg["sim"]["control_hz"]))
        self.dt_control = self.n_substeps * m.opt.timestep
        self.table_z = cfg["scene"]["table_top_z"]
        log.info("Model compiled: nq=%d nv=%d nu=%d, actuators=%s", m.nq, m.nv, m.nu,
                 [mujoco.mj_id2name(m, O.mjOBJ_ACTUATOR, i) for i in range(m.nu)])

    # ---------------------------------------------------------------- state access
    @property
    def time(self) -> float:
        return float(self.data.time)

    def arm_q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qadr].copy()

    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site].copy()

    def ee_mat(self) -> np.ndarray:
        return self.data.site_xmat[self.ee_site].reshape(3, 3).copy()

    def capsule_pos(self) -> np.ndarray:
        return self.data.xpos[self.capsule_body].copy()

    def capsule_yaw(self) -> float:
        """Yaw of the capsule long axis (mod pi)."""
        return axis_yaw_from_mat(self.data.xmat[self.capsule_body].reshape(3, 3), axis=2)

    def capsule_speed(self) -> float:
        return float(np.linalg.norm(self.data.qvel[self.capsule_dadr:self.capsule_dadr + 3]))

    def slot_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.slot_site].copy()

    def slot_yaw(self) -> float:
        q = self.data.mocap_quat[self.slot_mocap]
        return float(2 * np.arctan2(q[3], q[0]))

    def finger_width(self) -> float:
        return float(self.data.qpos[self.finger_qadr].sum())

    # ---------------------------------------------------------------- commands
    def set_arm_ctrl(self, q_target: np.ndarray) -> None:
        q = np.clip(q_target, self.ctrl_range[:, 0], self.ctrl_range[:, 1])
        self.data.ctrl[self.arm_act] = q

    def set_gripper(self, ctrl: float) -> None:
        lo, hi = self.model.actuator_ctrlrange[self.grip_act]
        self.data.ctrl[self.grip_act] = float(np.clip(ctrl, lo, hi))

    def step(self, n: int = 1) -> None:
        mujoco.mj_step(self.model, self.data, nstep=n)

    # ---------------------------------------------------------------- scene setup
    def set_capsule_pose(self, x: float, y: float, yaw: float) -> None:
        """Place the capsule lying on the table, long axis at world yaw `yaw`."""
        r = self.cfg["scene"]["capsule"]["radius"]
        q = quat_mul(_quat_xyz([0, 0, 1], yaw), _quat_xyz([0, 1, 0], np.pi / 2))
        a = self.capsule_qadr
        self.data.qpos[a:a + 3] = [x, y, self.table_z + r + 1e-4]
        self.data.qpos[a + 3:a + 7] = q
        self.data.qvel[self.capsule_dadr:self.capsule_dadr + 6] = 0

    def set_slot_pose(self, x: float, y: float, yaw: float) -> None:
        self.data.mocap_pos[self.slot_mocap] = [x, y, self.table_z]
        self.data.mocap_quat[self.slot_mocap] = _quat_xyz([0, 0, 1], wrap_pi(yaw))

    def sample_poses(self, rng: np.random.Generator) -> dict[str, tuple[float, float, float]]:
        r = self.cfg["scene_randomization"]
        c, s = r["capsule"], r["slot"]
        cap = (rng.uniform(*c["x"]), rng.uniform(*c["y"]),
               np.pi / 2 + np.radians(rng.uniform(*c["yaw_deg"])))
        slot = (rng.uniform(*s["x"]), rng.uniform(*s["y"]), np.radians(rng.uniform(*s["yaw_deg"])))
        return {"capsule": cap, "slot": slot}

    def reset(self, rng: np.random.Generator | None = None,
              poses: dict[str, tuple[float, float, float]] | None = None) -> dict:
        """Reset robot to home, place capsule and slot, let the capsule settle."""
        mujoco.mj_resetData(self.model, self.data)
        if poses is None:
            poses = self.sample_poses(rng if rng is not None else np.random.default_rng())
        self.data.qpos[self.arm_qadr] = self.home_q
        self.data.qpos[self.finger_qadr] = 0.04
        self.set_arm_ctrl(self.home_q)
        self.set_gripper(self.cfg["gripper"]["open_ctrl"])
        self.set_capsule_pose(*poses["capsule"])
        self.set_slot_pose(*poses["slot"])
        mujoco.mj_forward(self.model, self.data)
        n = int(self.cfg["scene_randomization"]["settle_time_s"] / self.model.opt.timestep)
        self.step(n)
        self.data.time = 0.0
        return poses
