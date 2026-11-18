"""Privileged evaluation only: full capsule geometry, release, and settling.

Nothing from this module is fed into perception or the task controller.
"""
from collections import deque
from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class PlacementSample:
    t: float
    contained: bool
    supported: bool
    released: bool
    slow: bool
    clearance_m: float
    floor_gap_m: float
    place_error_m: float
    local_pos: np.ndarray
    local_axis: np.ndarray

    @property
    def valid(self):
        return self.contained and self.supported and self.released and self.slow


def capsule_bounds(center, axis, slot_center, slot_rotation, radius, half_length, cavity):
    """Exact capsule projection: line segment Minkowski-summed with a radius ball.

    Slot origin here is its floor surface. The slot is a shallow fixture, so the
    object's top need not be below the rim. No extra XY tolerance is introduced.
    """
    local = np.asarray(slot_rotation).T @ (np.asarray(center) - slot_center)
    direction = np.asarray(slot_rotation).T @ np.asarray(axis)
    extent = radius + half_length * np.abs(direction)
    clearance = float(np.min(np.asarray(cavity) / 2 - np.abs(local[:2]) - extent[:2]))
    floor_gap = float(local[2] - extent[2])
    return local, direction, clearance, floor_gap


class SettlingWindow:
    def __init__(self, duration, max_drift, max_axis_drift, max_gap):
        self.duration = duration
        self.max_drift = max_drift
        self.max_axis_drift = max_axis_drift
        self.max_gap = max_gap
        self.samples = deque()

    def reset(self):
        self.samples.clear()

    def add(self, sample):
        if self.samples and sample.t <= self.samples[-1].t:
            raise ValueError('Settling samples must have increasing timestamps')
        self.samples.append(sample)
        # Retain one sample at/before the window boundary to establish coverage.
        cutoff = sample.t - self.duration
        while len(self.samples) > 1 and self.samples[1].t <= cutoff + 1e-10:
            self.samples.popleft()

    def stable(self):
        samples = list(self.samples)
        if len(samples) < 2 or samples[-1].t - samples[0].t < self.duration - 1e-10:
            return False
        if any(b.t - a.t > self.max_gap for a, b in zip(samples, samples[1:])):
            return False
        if not all(s.valid for s in samples):
            return False
        p0, a0 = samples[0].local_pos, samples[0].local_axis
        # Capsule axes have no arrow: yaw and yaw+pi describe the same geometry.
        return all(np.linalg.norm(s.local_pos - p0) <= self.max_drift
                   and np.arccos(np.clip(abs(s.local_axis @ a0), 0, 1)) <= self.max_axis_drift
                   for s in samples)


class PlacementEvaluator:
    VERSION = 'full_capsule_supported_released_stable_v1'

    def __init__(self, sim, cfg):
        self.sim, self.cfg = sim, cfg
        p = cfg['task']
        self.window = SettlingWindow(p['success_stable_window_s'], p['success_max_drift_m'],
                                     np.radians(p['success_max_axis_drift_deg']), sim.dt_control * 1.5)
        self.sample = None
        self.bottom_geom = mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_GEOM, 'slot_bottom')
        self.robot_bodies = set(sim.link_bodies.values())

    def reset(self):
        self.window.reset()
        self.sample = None

    def observe(self, release_commanded):
        s, cfg = self.sim, self.cfg
        cap, slot, task = cfg['scene']['capsule'], cfg['scene']['slot'], cfg['task']
        center = s.data.geom_xpos[s.capsule_geom].copy()
        axis = s.data.geom_xmat[s.capsule_geom].reshape(3, 3)[:, 2].copy()
        rotation = s.data.xmat[s.slot_body].reshape(3, 3).copy()
        local, direction, clearance, floor_gap = capsule_bounds(
            center, axis, s.slot_pos(), rotation, cap['radius'], cap['half_length'], slot['cavity'])
        supported_contact, robot_contact = False, False
        for contact in s.data.contact:
            pair = (int(contact.geom1), int(contact.geom2))
            if s.capsule_geom not in pair:
                continue
            other = pair[1] if pair[0] == s.capsule_geom else pair[0]
            if other == self.bottom_geom and contact.dist <= task['success_support_tolerance_m']:
                supported_contact = True
            if int(s.model.geom_bodyid[other]) in self.robot_bodies and contact.dist <= 0:
                robot_contact = True
        v = s.data.qvel[s.capsule_dadr:s.capsule_dadr + 6]
        supported = abs(floor_gap) <= task['success_support_tolerance_m'] and supported_contact
        released = release_commanded and s.finger_width() >= task['success_min_open_width_m'] and not robot_contact
        slow = np.linalg.norm(v[:3]) < task['success_max_speed'] and np.linalg.norm(v[3:]) < task['success_max_angular_speed']
        sample = PlacementSample(s.time, bool(clearance >= -1e-10), bool(supported), bool(released),
                                 bool(slow), clearance, floor_gap, float(np.linalg.norm(local[:2])),
                                 local, direction)
        self.sample = sample
        self.window.add(sample)
        return sample

    def geometry(self):
        s = self.sim
        cap, slot = self.cfg['scene']['capsule'], self.cfg['scene']['slot']
        local, _, clearance, gap = capsule_bounds(
            s.data.geom_xpos[s.capsule_geom], s.data.geom_xmat[s.capsule_geom].reshape(3, 3)[:, 2],
            s.slot_pos(), s.data.xmat[s.slot_body].reshape(3, 3), cap['radius'], cap['half_length'], slot['cavity'])
        return bool(clearance >= -1e-10 and abs(gap) <= self.cfg['task']['success_support_tolerance_m']), float(np.linalg.norm(local[:2]))

    def pose_errors(self, estimate):
        return (float(np.linalg.norm(estimate.capsule.pos - self.sim.capsule_pos())),
                float(np.linalg.norm(estimate.slot.pos - self.sim.slot_pos())))

    def result(self, timeout, final_state):
        sample = self.sample
        if sample is None:
            raise RuntimeError('No evaluation samples collected')
        stable = self.window.stable()
        checks = {k: getattr(sample, k) for k in ('contained', 'supported', 'released', 'slow')}
        checks.update(stable=stable, task_done=final_state == 'DONE', within_timeout=not timeout)
        # Comparison diagnostic only; no legacy condition participates in success.
        s, slot = self.sim, self.cfg['scene']['slot']
        rel = s.capsule_pos() - s.slot_pos()
        yaw = s.slot_yaw()
        lx, ly = np.cos(yaw)*rel[0]+np.sin(yaw)*rel[1], -np.sin(yaw)*rel[0]+np.cos(yaw)*rel[1]
        legacy = (not timeout and abs(lx) < slot['cavity'][0]/2 and abs(ly) < slot['cavity'][1]/2
                  and rel[2] < slot['wall_height'] and s.capsule_speed() < self.cfg['task']['success_max_speed'])
        return {'success': bool(all(checks.values())), 'timeout': bool(timeout),
                'time_to_complete_s': s.time, 'final_place_error_m': sample.place_error_m,
                'final_state': final_state, 'failure_reason': ';'.join(k for k, v in checks.items() if not v),
                'evaluator_version': self.VERSION, 'legacy_center_success': bool(legacy),
                'minimum_xy_clearance_m': sample.clearance_m, 'floor_gap_m': sample.floor_gap_m,
                **checks}
