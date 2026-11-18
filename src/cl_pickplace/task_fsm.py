"""Pick-and-place FSM. Each tick it maps fresh observations to a desired EE pose and
gripper command. In open-loop mode the object/slot estimate is frozen at episode start
and all recovery transitions are disabled."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np

from .kinematics import (axis_yaw_from_mat, gripper_down_quat, orientation_error,
                         wrap_half_pi)
from .perception import Estimate
from .placement_recovery import PlacementRecovery
from .visual_acquisition import VisualAcquisition
from .rim_extraction import RimExtraction


def _rot2(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


class S(IntEnum):
    HOME = 0
    APPROACH_OBJECT = 1
    DESCEND_OBJECT = 2
    GRASP = 3
    LIFT = 4
    TRANSPORT = 5
    DESCEND_SLOT = 6
    RELEASE = 7
    RETREAT = 8
    DONE = 9
    VERIFY = 10
    ABORT = 11
    ACQUIRE = 12
    EXTRACT = 13


@dataclass
class Obs:
    t: float
    ee_pos: np.ndarray
    ee_mat: np.ndarray
    q_arm: np.ndarray
    finger_width: float
    est: Estimate


@dataclass
class Command:
    p_des: np.ndarray | None = None
    q_des: np.ndarray | None = None
    joint_goal: np.ndarray | None = None
    gripper: float = 255.0


@dataclass
class FsmStats:
    num_regrasps: int = 0
    num_realigns: int = 0
    transitions: list = field(default_factory=list)   # (t, from, to, reason)
    visual_guard_ticks: int = 0


class TaskFSM(PlacementRecovery,VisualAcquisition,RimExtraction):
    def __init__(self, cfg: dict[str, Any], home_q: np.ndarray, table_z: float,
                 mode: str | None = None):
        self.p = cfg["task"]
        self.g = cfg["gripper"]
        self.c = cfg["control"]
        self.pc = cfg["perception"]
        self.home_q = np.asarray(home_q, float)
        self.table_z = table_z
        self.mode = mode or self.p["mode"]
        self.require_visual_observations = self.pc.get('mode') == 'learned'
        self.reset()

    # ------------------------------------------------------------------ helpers
    @property
    def closed_loop(self) -> bool:
        return self.mode == "closed_loop"

    def reset(self, start_state: S = S.HOME) -> None:
        self.state = start_state
        self.t_enter = 0.0
        self.yaw_ref = 0.0
        self.frozen: Estimate | None = None
        self.hold_pos: np.ndarray | None = None
        self.p_des = np.zeros(3)
        self.q_des = gripper_down_quat(0.0)
        self.gripper = float(self.g["open_ctrl"])
        self.lift_checked = False
        self.stats = FsmStats()
        self.last_err = (0.0, 0.0)
        self.slot_hist: deque = deque()     # (t, slot xy) for the "slot still" check
        self.t_in_tol: float | None = None  # start of the release confirmation dwell
        self.in_hand: tuple[np.ndarray, float] | None = None   # capsule offset in hand frame
        self.guard_action = ''
        self.reobserve_pos = None
        self.observation_samples = deque()
        self.observation_arrived = None
        self.placement_reference = None
        self.observation_attempts = 0
        self.qualified_slot = None
        self.view_index = 0
        self.view_started = None
        self.placement_target = None
        self.reset_recovery()
        self.reset_acquisition()
        self.reset_extraction()

    def _go(self, new: S, t: float, reason: str = "") -> None:
        self.stats.transitions.append((t, self.state.name, new.name, reason))
        if new == S.APPROACH_OBJECT and any(word in reason for word in ('grasp check failed','object dropped','visual placement miss','visual placement uncertain')):
            self.acquisition_needed=True
            if (self.acquisition_enabled and self.placement_retries>0
                    and self.extraction_enabled
                    and self.extraction_attempts<self.acquisition_settings['rim_extraction']['max_attempts']
                    and any(word in reason for word in ('grasp check failed','object dropped'))):
                self.extraction_needed=True
        if new == S.DESCEND_OBJECT:
            self.retry_grasp_offset=self.proposed_grasp_offset
        if new == S.GRASP:
            self.grasp_attempt_counted=False
            self.in_hand = None
            self.observation_arrived=None;self.placement_reference=None
            self.qualified_slot=None;self.observation_samples.clear()
            self.view_index=0;self.view_started=None;self.placement_target=None
        self.state = new
        self.t_enter = t
        self.hold_pos = None
        self.lift_checked = False
        self.t_in_tol = None

    def _yaw(self, target: float) -> float:
        """Gripper yaw for a mod-pi target, continuous w.r.t. the previous command."""
        y = self.yaw_ref + wrap_half_pi(target - self.yaw_ref)
        if abs(y) > np.radians(110):          # stay well inside the joint 7 range
            y = wrap_half_pi(target)
        self.yaw_ref = y
        return y

    def _errors(self, obs: Obs) -> tuple[float, float, float]:
        e = self.p_des - obs.ee_pos
        rot = float(np.degrees(np.linalg.norm(orientation_error(self.q_des, obs.ee_mat))))
        return float(np.linalg.norm(e)), float(np.linalg.norm(e[:2])), rot

    def _targets(self, obs: Obs) -> Estimate:
        if self.closed_loop or self.frozen is None:
            return obs.est
        return self.frozen

    def _cmd(self) -> Command:
        return Command(p_des=self.p_des.copy(), q_des=self.q_des.copy(), gripper=self.gripper)

    def _place_xy_yaw(self, obs: Obs, est: Estimate) -> tuple[np.ndarray, float]:
        """TCP xy and yaw that put the capsule (not the TCP) over the slot.

        The in-hand offset (hand frame) is estimated during TRANSPORT only, smoothed, and
        then held for the final descent, where the capsule is often hidden."""
        slot_xy = est.slot.pos[:2]
        if not (self.closed_loop and self.p["in_hand_correction"]):
            return slot_xy.copy(), self._yaw(est.slot.yaw)
        hand_yaw = axis_yaw_from_mat(obs.ee_mat, 0)
        trusted = (est.capsule.source == 'gt' or self.p['in_hand_correction_with_camera']
                   or (est.capsule.source.startswith('learned:') and self.p.get('in_hand_correction_with_learned',True)))
        sample_time=est.capsule.measurement_t
        captured_now=sample_time is None or abs(obs.t-sample_time)<1e-7
        if self.state == S.TRANSPORT and self._capsule_seen(obs) and trusted and captured_now:
            off_hand = _rot2(-hand_yaw) @ (est.capsule.pos[:2] - obs.ee_pos[:2])
            yaw_off = wrap_half_pi(est.capsule.yaw - hand_yaw)
            if self.in_hand is None:
                self.in_hand = (off_hand, yaw_off)
            else:
                a = self.p["in_hand_alpha"]
                o, y = self.in_hand
                self.in_hand = ((1 - a) * o + a * off_hand, y + a * wrap_half_pi(yaw_off - y))
        if self.in_hand is None:
            return slot_xy.copy(), self._yaw(est.slot.yaw)
        off_hand, yaw_off = self.in_hand
        yaw = self._yaw(est.slot.yaw - yaw_off)
        return slot_xy - _rot2(yaw) @ off_hand, yaw

    def _slot_motion(self, obs: Obs) -> float:
        """Slot displacement over the last slot_still_window_s of *fresh* measurements
        (inf until the window is filled). Held estimates are skipped, so a slot that was
        moving when it left the view still counts as moving."""
        win = self.p["slot_still_window_s"]
        measured_t=obs.est.slot.measurement_t
        stamp=obs.t if measured_t is None else measured_t
        if obs.est.slot.visible and obs.est.slot.lost_frames == 0 and (not self.slot_hist or stamp>self.slot_hist[-1][0]+1e-9):
            self.slot_hist.append((stamp, obs.est.slot.pos[:2].copy()))
        while self.slot_hist and self.slot_hist[-1][0] - self.slot_hist[0][0] > win+1e-9:
            self.slot_hist.popleft()
        if not self.slot_hist or self.slot_hist[-1][0] - self.slot_hist[0][0] < win * 0.9:
            return float("inf")
        return float(np.linalg.norm(self.slot_hist[-1][1] - self.slot_hist[0][1]))

    def _capsule_seen(self, obs: Obs) -> bool:
        """Fresh capsule measurement (the fingers often hide it from the wrist camera)."""
        cap=obs.est.capsule
        captured_now=(not cap.source.startswith('learned:') or
                      (cap.measurement_t is not None and abs(obs.t-cap.measurement_t)<1e-7))
        return cap.visible and cap.lost_frames == 0 and captured_now

    def _width_ok(self, obs: Obs) -> bool:
        return self.g["min_grasp_width"] <= obs.finger_width <= self.g["max_grasp_width"]

    def capsule_attached(self, obs: Obs) -> bool:
        return float(np.linalg.norm(obs.est.capsule.pos - obs.ee_pos)) < self.g["grasp_dist_tol"]

    def grasp_ok(self, obs: Obs) -> bool:
        if not self._capsule_seen(obs):
            return self._width_ok(obs)
        return self._width_ok(obs) and self.capsule_attached(obs)

    def dropped(self, obs: Obs) -> bool:
        if not self._capsule_seen(obs):
            return obs.finger_width < self.g["min_grasp_width"]
        return float(np.linalg.norm(obs.est.capsule.pos - obs.ee_pos)) > self.p["drop_threshold_m"]

    # ------------------------------------------------------------------ main tick
    def _visual_ok(self, estimate, t):
        return (estimate.visible and estimate.lost_frames==0 and estimate.measurement_t is not None
                and -1e-9<=t-estimate.measurement_t<=self.pc['learned']['max_age_s']+1e-9)

    def _visual_guard(self, obs):
        """Never start at default coordinates or descend/release on stale learned poses."""
        self.guard_action=''
        cap_ok=self._visual_ok(obs.est.capsule,obs.t);slot_ok=self._visual_ok(obs.est.slot,obs.t)
        if self.frozen is None and not (cap_ok and slot_ok):
            self.guard_action='wait_initial_detections';self.stats.visual_guard_ticks+=1
            self.gripper=float(self.g['open_ctrl'])
            return Command(joint_goal=obs.q_arm.copy(),gripper=self.gripper)
        if not self.closed_loop:return None # frozen valid snapshot defines open-loop baseline
        if self.state==S.DESCEND_OBJECT and not cap_ok:
            cap=obs.est.capsule;settings=self.pc['learned']
            age=float('inf') if cap.measurement_t is None else obs.t-cap.measurement_t
            clearance=float(obs.ee_pos[2]-cap.pos[2])
            xy=float(np.linalg.norm(obs.ee_pos[:2]-cap.pos[:2]))
            rot=float(np.degrees(np.linalg.norm(orientation_error(gripper_down_quat(cap.yaw),obs.ee_mat))))
            if (cap.source.startswith('learned:') and 0<=age<=settings['grasp_commit_max_age_s']
                    and 0<=clearance<=settings['grasp_commit_clearance_m'] and xy<=settings['grasp_commit_xy_m']
                    and rot<=self.p['rot_tol_deg']):
                # Deliberate, bounded final grasp commitment during finger occlusion.
                # The retained pose is NOT relabelled as a new visible measurement.
                self.guard_action='bounded_grasp_commit'
                return None
        if self.state==S.DESCEND_SLOT and not slot_ok:
            slot=obs.est.slot;settings=self.pc['learned']
            age=float('inf') if slot.measurement_t is None else obs.t-slot.measurement_t
            clearance=float(obs.ee_pos[2]-slot.pos[2])
            xy=float(np.linalg.norm(obs.ee_pos[:2]-self.p_des[:2]))
            rot=float(np.degrees(np.linalg.norm(orientation_error(self.q_des,obs.ee_mat))))
            if (slot.source.startswith('learned:') and 0<age<=settings['place_commit_max_age_s']
                    and 0<=clearance<=settings['place_commit_clearance_m'] and xy<=settings['place_commit_xy_m']
                    and rot<=self.p['rot_tol_deg'] and self._slot_motion(obs)<self.p['slot_still_m']):
                # A short final descent/release on previously observed stationary
                # geometry. This is explicitly bounded, NOT a fresh observation.
                self.guard_action='bounded_place_commit'
                return None
        if self.active_placement and self.state in (S.TRANSPORT,S.DESCEND_SLOT):
            return None # active waypoint/age/motion policy handles destination loss
        target=None
        if self.state in (S.APPROACH_OBJECT,S.DESCEND_OBJECT) and not cap_ok:target='object'
        if self.state in (S.TRANSPORT,S.DESCEND_SLOT) and not slot_ok:target='destination'
        if target is None:
            self.reobserve_pos=None
            return None
        if self.state in (S.DESCEND_OBJECT,S.DESCEND_SLOT):
            self._go(S.APPROACH_OBJECT if target=='object' else S.TRANSPORT,obs.t,'learned target lost: re-observe '+target)
        if self.reobserve_pos is None:
            height=self.pc['search_height'] if target=='object' else self.p['carry_height']
            self.reobserve_pos=obs.ee_pos.copy();self.reobserve_pos[2]=max(obs.ee_pos[2],self.table_z+height)
        self.p_des=self.reobserve_pos.copy()
        self.gripper=float(self.g['open_ctrl'] if target=='object' else self.g['closed_ctrl'])
        self.t_in_tol=None;self.guard_action='reobserve_'+target;self.stats.visual_guard_ticks+=1
        return self._cmd()

    @property
    def active_placement(self):
        return (self.closed_loop and self.require_visual_observations
                and self.p.get('placement_policy', 'reactive_v1') == 'active_observe_v1')

    def _active_place(self, obs, est):
        """Observation waypoints followed by a bounded placement.

        Uses accepted visual samples and robot proprioception only. Stability is
        evidence over a finite window, not a guarantee against future target motion.
        All distances/deadlines are configured priors, never evaluator feedback.
        """
        settings=self.p['active_placement'];slot=obs.est.slot;t=obs.t
        # A temporally stable partial rectangle may have a biased centre. Use
        # the frozen geometry estimator's footprint evidence, without changing
        # its acceptance rules or reading actual simulator dimensions/poses.
        extent=np.asarray(slot.quality.get('footprint_extent_m',[]))
        nominal=np.asarray(settings['nominal_outer_extent_m'])
        complete=(extent.shape==(2,) and np.all(extent>=nominal*settings['minimum_extent_fraction']))
        fresh=self._visual_ok(slot,t) and complete
        if fresh:self.qualified_slot=(slot.pos.copy(),slot.yaw,slot.measurement_t)
        effective=est.copy()
        if self.qualified_slot is not None:
            effective.slot.pos=self.qualified_slot[0].copy();effective.slot.yaw=self.qualified_slot[1]
        xy,yaw=self._place_xy_yaw(obs,effective)
        self.placement_target=(effective.slot.pos.copy(),effective.slot.yaw,
                               self.qualified_slot[2] if self.qualified_slot is not None else slot.measurement_t)
        if fresh and (not self.observation_samples or slot.measurement_t>self.observation_samples[-1][0]+1e-9):
            self.observation_samples.append((slot.measurement_t,slot.pos.copy(),slot.yaw))
        while self.observation_samples and self.observation_samples[0][0]<t-settings['history_s']:
            self.observation_samples.popleft()
        self.gripper=float(self.g['closed_ctrl'])
        z_place=effective.slot.pos[2]+self.p['place_height_offset']
        if self.state==S.DESCEND_SLOT:
            age=float('inf') if self.qualified_slot is None else t-self.qualified_slot[2]
            ref=self.placement_reference
            changed=(fresh and ref is not None and (
                np.linalg.norm(slot.pos[:2]-ref[0][:2])>settings['motion_abort_m']
                or abs(wrap_half_pi(slot.yaw-ref[1]))>np.radians(settings['motion_abort_deg'])))
            if age>settings['commit_max_age_s'] or changed:
                self._go(S.TRANSPORT,t,'active observation: motion' if changed else 'active observation: age limit')
                self.observation_arrived=None;self.placement_reference=None
                self.view_started=None
                self.observation_attempts+=1
            else:
                self.p_des=np.array([xy[0],xy[1],z_place]);self.q_des=gripper_down_quat(yaw)
                self.guard_action='active_place_visible' if fresh else 'active_place_commit'
                e,_,r=self._errors(obs)
                if e<self.p['descend_tol_m'] and r<self.p['rot_tol_deg']:
                    if self.t_in_tol is None:self.t_in_tol=t
                    if t-self.t_in_tol>=self.p['release_confirm_s']:
                        self.release_reference=(obs.est.slot.pos.copy(),obs.est.slot.yaw)
                        self.release_opened_t=None
                        self._go(S.RELEASE,t,'active observation: at place height')
                else:self.t_in_tol=None
                return self._cmd()
        # Start overhead. If it never yields sufficient evidence, move through
        # declared lateral views rather than retrying one occluded pose forever.
        if self.view_started is None:self.view_started=t
        if t-self.view_started>=settings['view_timeout_s']:
            self.view_index=(self.view_index+1)%len(settings['view_offsets_xy_m'])
            self.view_started=t;self.observation_arrived=None
        offset=_rot2(yaw)@np.asarray(settings['view_offsets_xy_m'][self.view_index])
        self.p_des=np.array([xy[0]+offset[0],xy[1]+offset[1],z_place+settings['view_height_m']])
        self.q_des=gripper_down_quat(yaw);self.guard_action='active_observe_destination'
        self.stats.visual_guard_ticks+=1
        e,_,r=self._errors(obs)
        if e>=self.p['approach_tol_m'] or r>=self.p['rot_tol_deg']:
            self.observation_arrived=None
            return self._cmd()
        if self.observation_arrived is None:self.observation_arrived=t
        samples=[a for a in self.observation_samples if a[0]>=self.observation_arrived-1e-9]
        stable=(fresh and len(samples)>=settings['minimum_samples']
                and samples[-1][0]-samples[0][0]>=settings['stable_window_s']-1e-9
                and max(np.linalg.norm(a[1][:2]-samples[-1][1][:2]) for a in samples)<=settings['stable_position_m']
                and max(abs(wrap_half_pi(a[2]-samples[-1][2])) for a in samples)<=np.radians(settings['stable_yaw_deg']))
        if stable:
            self.placement_reference=(slot.pos.copy(),slot.yaw)
            self._go(S.DESCEND_SLOT,t,'active observation: confirmed stable destination')
        return self._cmd()

    def update(self, obs: Obs) -> Command:
        p, t = self.p, obs.t
        if self.acquisition_enabled:
            acquisition=self.acquisition_dispatch(obs)
            if acquisition is not None:return acquisition
        if self.recovery_enabled:
            if self.state == S.ABORT:
                self.gripper=float(self.g['open_ctrl'])
                return self._cmd()
            if self.recovery_settings.get('verify',True) and self.state in (S.VERIFY,S.DONE):
                return self.verification_tick(obs)
            veto=self.release_guard(obs)
            if veto is not None:return veto
        if self.require_visual_observations:
            guarded=self._visual_guard(obs)
            if guarded is not None:
                if self.acquisition_enabled and self.guard_action in ('reobserve_object','wait_initial_detections'):
                    if self.guard_action=='wait_initial_detections' and obs.t<self.acquisition_settings['initial_wait_s']:
                        return guarded
                    target='capsule' if not self._visual_ok(obs.est.capsule,obs.t) else 'fixture'
                    self.begin_acquisition(obs,target)
                    return self.acquisition_tick(obs)
                return guarded
        if self.frozen is None:
            self.frozen = obs.est.copy()
        est = self._targets(obs)
        st = self.state
        open_c, closed_c = float(self.g["open_ctrl"]), float(self.g["closed_ctrl"])

        if st == S.HOME:
            self.gripper = open_c
            if np.max(np.abs(obs.q_arm - self.home_q)) < self.c["home_tol_rad"]:
                self.yaw_ref = axis_yaw_from_mat(obs.ee_mat, 0)
                self._go(S.APPROACH_OBJECT, t, "at home")
                return self.update(obs)
            return Command(joint_goal=self.home_q.copy(), gripper=self.gripper)

        if st == S.APPROACH_OBJECT:
            self.gripper = open_c
            cap = est.capsule
            h = p["pre_grasp_height"]
            if self.closed_loop and cap.lost_frames >= self.pc["lost_frames_for_search"]:
                h = self.pc["search_height"]
            self.p_des = self.grasp_position(obs,est) + [0,0,h]
            self.q_des = gripper_down_quat(self._yaw(cap.yaw))
            e, _, r = self._errors(obs)
            if e < p["approach_tol_m"] and r < p["rot_tol_deg"] and cap.lost_frames == 0:
                self._go(S.DESCEND_OBJECT, t, "above object")
            return self._cmd()

        if st == S.DESCEND_OBJECT:
            self.gripper = open_c
            cap = est.capsule
            self.p_des = self.grasp_position(obs,est)
            self.q_des = gripper_down_quat(self._yaw(cap.yaw))
            e, eh, r = self._errors(obs)
            if self.closed_loop and eh > p["realign_threshold_m"]:
                self.stats.num_realigns += 1
                self._go(S.APPROACH_OBJECT, t, f"realign object {eh*1000:.0f}mm")
            elif e < p["descend_tol_m"] and r < p["rot_tol_deg"]:
                self._go(S.GRASP, t, "at grasp height")
            return self._cmd()

        if st == S.GRASP:
            if self.placement_retries>0 and not self.grasp_attempt_counted:
                self.recovery_grasp_attempts+=1;self.grasp_attempt_counted=True
            self.gripper = closed_c
            if t - self.t_enter >= p["grasp_settle_time_s"]:
                self.hold_pos = self.p_des.copy()
                self._go(S.LIFT, t, "closed")
                self.hold_pos = self.p_des.copy()
            return self._cmd()

        if st == S.LIFT:
            self.gripper = closed_c
            if self.hold_pos is None:
                self.hold_pos = self.p_des.copy()
            self.p_des = np.array([self.hold_pos[0], self.hold_pos[1],
                                   self.table_z + p["carry_height"]])
            e, _, r = self._errors(obs)
            if (self.closed_loop and not self.lift_checked
                    and obs.ee_pos[2] > self.hold_pos[2] + p["lift_check_height"]):
                self.lift_checked = True
                if not self.grasp_ok(obs):
                    self.stats.num_regrasps += 1
                    self.gripper = open_c
                    self._go(S.APPROACH_OBJECT, t, "grasp check failed")
                    return self._cmd()
            if e < p["approach_tol_m"] * 2 and (self.lift_checked or not self.closed_loop):
                self._go(S.TRANSPORT, t, "lifted")
            return self._cmd()

        if st in (S.TRANSPORT, S.DESCEND_SLOT):
            self.gripper = closed_c
            if self.closed_loop and self.dropped(obs):
                self.stats.num_regrasps += 1
                self.gripper = open_c
                self._go(S.APPROACH_OBJECT, t, "object dropped")
                return self._cmd()
            moved = self._slot_motion(obs) if self.closed_loop else 0.0
            still = moved < p["slot_still_m"]
            if self.active_placement:
                return self._active_place(obs,est)
            xy, yaw = self._place_xy_yaw(obs, est)
            if (st==S.TRANSPORT and self.closed_loop
                    and t-self.t_enter<p.get('transport_wait_s',0.0)):
                self.guard_action='fixed_transport_wait'
                return self._cmd()
            z_place = est.slot.pos[2] + p["place_height_offset"]
            if st == S.TRANSPORT:
                self.p_des = np.array([xy[0], xy[1], z_place + p["pre_place_height"]])
                self.q_des = gripper_down_quat(yaw)
                e, _, r = self._errors(obs)
                if e < p["approach_tol_m"] and r < p["rot_tol_deg"]:
                    self._go(S.DESCEND_SLOT, t, "above slot")
            else:
                self.p_des = np.array([xy[0], xy[1], z_place])
                self.q_des = gripper_down_quat(yaw)
                e, eh, r = self._errors(obs)
                # Near the bottom the wrist camera often loses the slot (too close, capsule
                # in the way): hold the estimate, unless the slot was moving when it was lost.
                # No recent fresh data at all (moved == inf) counts as unknown: back up
                # once the slot has been lost for a while, instead of waiting forever.
                lost = obs.est.slot.lost_frames
                lost_moving = lost > 0 and not still and (
                    moved != float("inf") or lost >= self.pc["lost_frames_for_search"])
                if self.closed_loop and eh > p["realign_threshold_m"]:
                    self.stats.num_realigns += 1
                    self._go(S.TRANSPORT, t, f"realign slot {eh*1000:.0f}mm")
                elif self.closed_loop and lost_moving:
                    self.stats.num_realigns += 1
                    self._go(S.TRANSPORT, t, "slot lost while moving or unknown")
                elif e < p["descend_tol_m"] and r < p["rot_tol_deg"] and still:
                    # closed loop: stay aligned for a short dwell before committing
                    if self.t_in_tol is None:
                        self.t_in_tol = t
                    dwell = p["release_confirm_s"] if self.closed_loop else 0.0
                    if t - self.t_in_tol >= dwell:
                        self.release_reference=(obs.est.slot.pos.copy(),obs.est.slot.yaw)
                        self.release_opened_t=None
                        self._go(S.RELEASE, t, "at place height")
                else:
                    self.t_in_tol = None
            return self._cmd()

        if st == S.RELEASE:
            if self.release_opened_t is None:self.release_opened_t=t
            self.gripper = open_c
            if t - self.t_enter >= p["release_settle_time_s"]:
                self._go(S.RETREAT, t, "released")
                self.hold_pos = self.p_des.copy()
            return self._cmd()

        if st == S.RETREAT:
            self.gripper = open_c
            if self.hold_pos is None:
                self.hold_pos = self.p_des.copy()
            self.p_des = self.hold_pos + [0, 0, p["retreat_height"]]
            e, _, _ = self._errors(obs)
            if e < p["approach_tol_m"] * 2:
                if self.recovery_enabled and self.recovery_settings.get('verify',True):
                    self._go(S.VERIFY,t,'retreated: inspect placement')
                    self.verification_started=None
                    self.verification_anchor=None
                    self.verification_samples.clear()
                    self.verification_last_pair=None
                else:
                    self._go(S.DONE, t, 'retreated')
            return self._cmd()

        # DONE: hold
        self.gripper = open_c
        return self._cmd()
