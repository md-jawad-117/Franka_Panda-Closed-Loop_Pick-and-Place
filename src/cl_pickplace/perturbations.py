"""Pre-generated environment interventions, independent of timed-trial policy.

Object nudges use external forces, never position/velocity edits. The fixture is
moved by a declared kinematic stage. Phase-triggered tests are a separate protocol.
"""
from dataclasses import dataclass, field
import copy
import hashlib
import json
import numpy as np
from .kinematics import wrap_pi

KINDS = ('arm_push', 'slot_move', 'object_nudge')


@dataclass
class Event:
    kind: str
    t_start: float
    t_end: float
    magnitude: float
    state: str = ''
    params: dict = field(default_factory=dict)
    unit: str = ''
    plan_index: int | None = None


@dataclass
class PlannedEvent:
    kind: str
    state: str
    delay: float
    params: dict
    magnitude: float
    duration: float
    start_s: float | None
    unit: str
    height_above_place_m: float | None = None
    fired: bool = False

    def description(self):
        return {k: copy.deepcopy(getattr(self, k)) for k in
                ('kind', 'state', 'delay', 'params', 'magnitude', 'duration', 'start_s', 'unit', 'height_above_place_m')}


class Perturber:
    def __init__(self, sim, cfg, rng):
        self.sim, self.cfg, self.rng = sim, cfg['perturbations'], rng
        self.bounds = cfg['scene_randomization']['workspace_bounds']
        self.events, self.active, self.schedule = [], [], []
        self._state_enter, self._kind_seeds, self._counts = {}, {}, {}
        self._force_bodies = set()
        self.protocol = self.cfg['schedule_mode']
        if self.protocol not in ('time', 'state'):
            raise ValueError(f'Unknown disturbance protocol: {self.protocol}')

    def reset(self, kinds=None):
        self.events.clear()
        self.active.clear()
        self.sim.data.xfrc_applied[:] = 0
        self._force_bodies.clear()
        self._state_enter.clear()
        self.protocol = self.cfg['schedule_mode']
        self._kind_seeds = {k: int(self.rng.integers(0, 2**63-1)) for k in KINDS}
        self._counts = {k: 0 for k in KINDS}
        self.schedule = [self._plan(k) for k in (kinds or [])]

    def _plan(self, kind, state=None, delay=None):
        if kind not in KINDS:
            raise ValueError(f'Unknown disturbance: {kind}')
        index = self._counts[kind]
        self._counts[kind] += 1
        rng = np.random.default_rng(np.random.SeedSequence([self._kind_seeds[kind], index]))
        c, sch = self.cfg['scheduled'][kind], self.cfg['scheduled']
        phase = self.protocol == 'state' or state is not None
        state = state or (str(rng.choice(c['states'])) if phase else '')
        if phase:
            limits = c.get('trigger_delay_by_state', {}).get(state, sch['trigger_delay_s'])
            delay = float(rng.uniform(*limits)) if delay is None else float(delay)
            start = None
        else:
            start, delay = float(rng.uniform(*c['time_s_range'])), 0.0
        if kind == 'arm_push':
            magnitude = float(rng.uniform(*c['force_N_range']))
            params = self._push_params(rng, magnitude)
            duration, unit = c['duration_s'], 'N'
        elif kind == 'object_nudge':
            magnitude = float(rng.uniform(*c['force_N_range']))
            params = self._nudge_params(rng, magnitude)
            duration, unit = c['duration_s'], 'N'
        else:
            magnitude = float(rng.uniform(*c['shift_m_range']))
            dyaw = float(np.radians(rng.uniform(*c['yaw_change_deg_range'])))
            params = self._slot_params(rng, magnitude, dyaw)
            duration, unit = c['move_duration_s'], 'm'
        height = c.get('trigger_height_by_state', {}).get(state) if phase else None
        return PlannedEvent(kind, state, delay, params, magnitude, float(duration), start, unit, height)

    def set_schedule(self, items):
        self.protocol = 'state'
        self.schedule = [self._plan(k, state=s, delay=d) for k, s, d in items]

    def manifest(self):
        return {'protocol': self.protocol, 'events': [p.description() for p in self.schedule],
                'object_intervention': 'external_horizontal_force', 'fixture_intervention': 'kinematic_stage'}

    def plan_id(self):
        return hashlib.sha256(json.dumps(self.manifest(), sort_keys=True, allow_nan=False).encode()).hexdigest()

    def delivery(self):
        return [{'kind': e.kind, 'actual_start_s': e.t_start, 'actual_end_s': e.t_end,
                 'magnitude': e.magnitude, 'unit': e.unit, 'state': e.state,
                 'params': e.params, 'plan_index': e.plan_index} for e in self.events]

    def finished(self, t, settling_window):
        if any(not p.fired for p in self.schedule):
            return False
        return not self.active and (not self.events or t >= max(e.t_end for e in self.events)+settling_window)

    def _push_params(self, rng, force, link=None):
        link = link or str(rng.choice(self.cfg['push_links']))
        angle = rng.uniform(0, 2*np.pi)
        return {'body': int(self.sim.link_bodies[link]), 'link': link,
                'force': (force*np.array([np.cos(angle), np.sin(angle), 0.0])).tolist()}

    def _nudge_params(self, rng, force):
        angle = rng.uniform(0, 2*np.pi)
        return {'body': int(self.sim.capsule_body),
                'force': (force*np.array([np.cos(angle), np.sin(angle), 0.0])).tolist()}

    def _slot_params(self, rng, shift, dyaw):
        x0, y0 = map(float, self.sim.data.mocap_pos[self.sim.slot_mocap][:2])
        a0 = self._stage_yaw()
        for _ in range(1000):
            angle = rng.uniform(0, 2*np.pi)
            x1, y1 = x0+shift*np.cos(angle), y0+shift*np.sin(angle)
            if self.bounds['x'][0] <= x1 <= self.bounds['x'][1] and self.bounds['y'][0] <= y1 <= self.bounds['y'][1]:
                return {'from': [x0, y0, a0], 'to': [float(x1), float(y1), wrap_pi(a0+dyaw)],
                        'dyaw_deg': float(np.degrees(dyaw))}
        raise ValueError('Cannot sample fixture displacement inside declared bounds')

    def _stage_yaw(self):
        # mocap controls are authoritative even before the next mj_forward/step.
        w,x,y,z=map(float,self.sim.data.mocap_quat[self.sim.slot_mocap])
        return float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))

    def _add(self, event):
        self.events.append(event)
        self.active.append(event)
        return event

    def push_arm(self, t, force, duration, link=None, state=''):
        return self._add(Event('arm_push', t, t+duration, float(force), state,
                               self._push_params(self.rng, force, link), unit='N'))

    def move_slot(self, t, shift, dyaw, duration, state=''):
        if duration<=0:raise ValueError('Fixture move duration must be positive')
        params=self._slot_params(self.rng, shift, dyaw)
        self._interrupt_slot_moves(t,params['from'],'manual_slot_move')
        return self._add(Event('slot_move', t, t+duration, float(shift), state,
                               params, unit='m'))

    def _interrupt_slot_moves(self,t,pose,reason):
        # A manual destination command retargets the single kinematic stage.
        # Do not let old trajectories continue overwriting its new command.
        for event in self.active:
            if event.kind=='slot_move':
                event.params.update({'interrupted_at_s':float(t),'requested_end_s':event.t_end,
                                     'interrupted_pose':list(pose),'interrupted_by':reason,
                                     'executed_shift_m':float(np.linalg.norm(np.asarray(pose)[:2]-np.asarray(event.params['from'])[:2]))})
                event.t_end=t
        self.active=[event for event in self.active if event.kind!='slot_move']

    def nudge_object(self, t, force, duration, state=''):
        return self._add(Event('object_nudge', t, t+duration, float(force), state,
                               self._nudge_params(self.rng, force), unit='N'))

    def _fire(self, index, t, state):
        p = self.schedule[index]
        p.fired = True
        params = copy.deepcopy(p.params)
        if p.kind == 'slot_move':
            planned_from, planned_to = np.asarray(params['from']), np.asarray(params['to'])
            current = np.r_[self.sim.data.mocap_pos[self.sim.slot_mocap][:2], self._stage_yaw()]
            if np.linalg.norm(current-planned_from) > 1e-9:
                # Repeated/manual phase interventions start where the stage actually is;
                # the offset was still sampled before execution. Record any clipping.
                destination = current[:2] + planned_to[:2]-planned_from[:2]
                clipped = np.clip(destination, [self.bounds['x'][0],self.bounds['y'][0]],
                                  [self.bounds['x'][1],self.bounds['y'][1]])
                params.update({'from': current.tolist(),
                               'to': [*map(float,clipped), wrap_pi(current[2]+wrap_pi(planned_to[2]-planned_from[2]))],
                               'clipped': bool(not np.array_equal(destination,clipped))})
            params['actual_shift_m'] = float(np.linalg.norm(np.asarray(params['to'])[:2]-np.asarray(params['from'])[:2]))
            self._interrupt_slot_moves(t,params['from'],'scheduled_slot_move')
        self._add(Event(p.kind, t, t+p.duration, p.magnitude, state, params, p.unit, index))

    def update_schedule(self, t, state, ee_above_capsule, grasped, ee_above_place=None):
        if self.protocol != 'state':
            return
        self._state_enter.setdefault(state, t)
        for i, p in enumerate(self.schedule):
            if p.fired or p.state != state or t-self._state_enter[state] < p.delay:
                continue
            if p.height_above_place_m is not None and (ee_above_place is None or ee_above_place > p.height_above_place_m):
                continue
            if p.kind == 'object_nudge' and (grasped or ee_above_capsule < self.cfg['nudge_min_clearance_m']):
                continue
            self._fire(i, t, state)

    def apply(self, t):
        if self.protocol == 'time':
            for i, p in enumerate(self.schedule):
                if not p.fired and t+1e-10 >= p.start_s:
                    self._fire(i, t, '')
        s = self.sim
        for body in self._force_bodies:
            s.data.xfrc_applied[body, :3] = 0
        self._force_bodies.clear()
        active = []
        for ev in self.active:
            u = 1.0 if ev.t_end <= ev.t_start else float(np.clip((t-ev.t_start)/(ev.t_end-ev.t_start), 0, 1))
            if ev.kind in ('arm_push', 'object_nudge'):
                if t < ev.t_end:
                    body = ev.params['body']
                    s.data.xfrc_applied[body, :3] += np.asarray(ev.params['force'])
                    self._force_bodies.add(body)
            elif ev.kind == 'slot_move':
                start, end = np.asarray(ev.params['from']), np.asarray(ev.params['to'])
                smooth = .5-.5*np.cos(np.pi*u)
                xy = start[:2]+smooth*(end[:2]-start[:2])
                yaw = start[2]+smooth*wrap_pi(end[2]-start[2])
                s.set_slot_pose(float(xy[0]), float(xy[1]), yaw)
            if t < ev.t_end:
                active.append(ev)
        self.active = active
