"""One pick-and-place episode: perception -> FSM -> controller every control tick,
perturbations applied every physics substep. Shared by demo, recorder and eval."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .controller import TaskSpaceController
from .evaluation import PlacementEvaluator
from .perception import Estimate, Perception
from .perturbations import KINDS, Perturber
from .sim import Sim
from .task_fsm import Command, Obs, S, TaskFSM


@dataclass
class TickLog:
    t: list = field(default_factory=list)
    state: list = field(default_factory=list)
    ee_pos: list = field(default_factory=list)
    p_des: list = field(default_factory=list)
    pos_err: list = field(default_factory=list)
    est_err: list = field(default_factory=list)    # perception error of capsule / slot (m)
    slot_lost_frames: list = field(default_factory=list)
    visual: list = field(default_factory=list)


def parse_perturb(arg: str | list[str] | None) -> list[str]:
    if not arg or arg == "none":
        return []
    if arg == "all":
        return list(KINDS)
    if isinstance(arg, str):
        arg = arg.split(",")
    bad = [a for a in arg if a not in KINDS]
    if bad:
        raise ValueError(f"unknown perturbation {bad}, choose from {KINDS}")
    return list(arg)


class Episode:
    def __init__(self, cfg: dict[str, Any], sim: Sim | None = None, mode: str | None = None,
                 perception_mode: str | None = None, perturb: str | list[str] | None = None):
        if cfg["control"]["mode"] != "ik_position":
            raise NotImplementedError(
                f"control.mode '{cfg['control']['mode']}': only 'ik_position' is implemented "
                "(the compliant 'osc' mode is optional milestone M7)")
        self.cfg = cfg
        self.sim = sim or Sim(cfg)
        self.mode = mode or cfg["task"]["mode"]
        self.perception_mode = perception_mode or cfg["perception"]["mode"]
        self.kinds = parse_perturb(perturb)
        self.rng = np.random.default_rng(0)
        self.ctrl = TaskSpaceController(self.sim, cfg)
        self.perception = Perception(self.sim, cfg, self.rng, self.perception_mode)
        self.fsm = TaskFSM(cfg, self.sim.home_q, self.sim.table_z, self.mode)
        self.perturber = Perturber(self.sim, cfg, self.rng)
        self.evaluator = PlacementEvaluator(self.sim, cfg)
        self.paused = False
        self.est: Estimate | None = None
        self.cmd = Command()
        self.log = TickLog()
        self.seed = 0
        self.poses: dict = {}
        self.result: dict | None = None

    # ------------------------------------------------------------------ lifecycle
    def reset(self, seed: int, poses: dict | None = None, capture: bool = True) -> None:
        self.seed = seed
        # Separate streams so the scene and perturbation schedule do not depend on mode.
        ss = np.random.SeedSequence(seed)
        scene_rng, pert_rng, noise_rng = [np.random.default_rng(s) for s in ss.spawn(3)]
        self.poses = self.sim.reset(scene_rng, poses)
        self.perception.rng = noise_rng
        self.perception.set_mode(self.perception_mode)
        self.perturber.rng = pert_rng
        self.perturber.reset(self.kinds)
        self.fsm.mode = self.mode
        self.fsm.reset()
        self.ctrl.reset()
        self.log = TickLog()
        self.result = None
        self.cmd = Command()
        self.evaluator.reset()
        self.est = self.perception.update() if capture else None

    @property
    def t(self) -> float:
        return self.sim.time

    @property
    def done(self) -> bool:
        return self.result is not None

    def grasped(self) -> bool:
        return self.fsm.state in (S.GRASP, S.LIFT, S.TRANSPORT, S.DESCEND_SLOT)

    # ------------------------------------------------------------------ stepping
    def control_tick(self, perception_ready: bool = False) -> None:
        s = self.sim
        self.fsm.require_visual_observations = self.perception_mode == 'learned'
        if not perception_ready:self.est = self.perception.update()
        if self.est is None:raise RuntimeError('A perception estimate must be prepared before control')
        if self.paused:
            self.ctrl.hold()
            return
        # FSM receives only robot measurements and a detached perception estimate.
        obs = Obs(self.t, s.ee_pos(), s.ee_mat(), s.arm_q(), s.finger_width(), self.est.copy())
        self.cmd = self.fsm.update(obs)
        if self.cmd.joint_goal is not None:
            self.ctrl.step_joint(self.cmd.joint_goal)
        else:
            self.ctrl.step(self.cmd.p_des, self.cmd.q_des)
        s.set_gripper(self.cmd.gripper)
        self._log(obs)

    def _log(self, obs: Obs) -> None:
        L, s = self.log, self.sim
        L.t.append(obs.t)
        L.state.append(int(self.fsm.state))
        L.ee_pos.append(obs.ee_pos)
        p_des = self.cmd.p_des if self.cmd.p_des is not None else obs.ee_pos
        L.p_des.append(p_des)
        L.pos_err.append(float(np.linalg.norm(p_des - obs.ee_pos)))
        L.est_err.append(self.evaluator.pose_errors(self.est))
        L.slot_lost_frames.append(obs.est.slot.lost_frames)
        if self.perception_mode=='learned':
            L.visual.append({'t':obs.t,'state':self.fsm.state.name,'guard':self.fsm.guard_action,
                             'capsule_visible':obs.est.capsule.visible,'slot_visible':obs.est.slot.visible,
                             'capsule_measurement_t':obs.est.capsule.measurement_t,'slot_measurement_t':obs.est.slot.measurement_t,
                             'capsule_source':obs.est.capsule.source,'slot_source':obs.est.slot.source,
                             'slot_observation':obs.est.slot.observation,
                             'placement_retries':self.fsm.placement_retries,'release_vetoes':self.fsm.release_vetoes,
                             'verification_status':self.fsm.verification_status,
                             'acquisition_view':self.fsm.acquisition_view,'recovery_grasp_attempts':self.fsm.recovery_grasp_attempts,
                             'finger_width_m':obs.finger_width,'capsule_estimated_yaw_rad':obs.est.capsule.yaw,
                             'ee_pos':obs.ee_pos.tolist(),'command_position':self.cmd.p_des.tolist() if self.cmd.p_des is not None else None,
                             'capsule_estimate':obs.est.capsule.pos.tolist(),'slot_estimate':obs.est.slot.pos.tolist()})

    def physics_period(self) -> None:
        # Phase gating belongs to the environment, not to controller observations.
        if self.perturber.protocol == 'state':
            self.perturber.update_schedule(self.t, self.fsm.state.name,
                float(self.sim.ee_pos()[2]-self.sim.capsule_pos()[2]), self.grasped(),
                float(self.sim.ee_pos()[2]-self.sim.slot_pos()[2]-self.cfg['task']['place_height_offset']))
        for _ in range(self.sim.n_substeps):
            self.perturber.apply(self.t)
            self.sim.step()

    def step(self, perception_ready: bool = False) -> None:
        """One control period: control tick + n physics substeps + termination check."""
        if self.done:
            self.physics_period()
            return
        self.control_tick(perception_ready=perception_ready)
        self.physics_period()
        self.evaluator.observe(self.fsm.state in (S.RELEASE, S.RETREAT, S.DONE, S.VERIFY, S.ABORT)
                               and self.cmd.gripper == self.cfg['gripper']['open_ctrl'])
        self._check_end()

    def _check_end(self) -> None:
        # The timeout applies even when a late perturbation is pending after DONE.
        if self.t >= self.cfg['task']['episode_timeout_s']:
            self.result = self.evaluate(timeout=True)
        elif self.fsm.state == S.ABORT:
            self.result = self.evaluate(timeout=False)
        elif self.fsm.state == S.DONE:
            window = self.cfg['task']['success_stable_window_s']
            if self.t - self.fsm.t_enter >= window and self.perturber.finished(self.t, window):
                self.result = self.evaluate(timeout=False)

    def run(self) -> dict:
        while not self.done:
            self.step()
        return self.result

    # ------------------------------------------------------------------ success
    def placement(self) -> tuple[bool, float]:
        """Geometry diagnostic; success additionally requires release/support/stability."""
        return self.evaluator.geometry()

    def evaluate(self, timeout: bool) -> dict:
        return self.evaluator.result(timeout, self.fsm.state.name)
