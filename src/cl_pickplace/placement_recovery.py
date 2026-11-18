"""Optional visual release veto and post-release inspection, without simulator access.

All dimensions are declared nominal priors. Visual containment cannot prove contact
or support; the independent physical evaluator remains the research success metric.
"""
from collections import deque
import numpy as np

from .kinematics import gripper_down_quat, wrap_half_pi


class PlacementRecovery:
    @property
    def recovery_settings(self):
        return self.p.get('placement_recovery', {})

    @property
    def recovery_enabled(self):
        return self.closed_loop and self.require_visual_observations and bool(self.recovery_settings)

    def reset_recovery(self):
        self.release_reference = None
        self.release_opened_t = None
        self.verification_started = None
        self.verification_anchor = None
        self.verification_samples = deque()
        self.verification_last_pair = None
        self.verification_status = 'not_started'
        self.verification_history = []
        self.placement_retries = 0
        self.release_vetoes = 0
        self.pending_veto_active = False

    def release_guard(self, obs):
        """Veto before opening, including the tick AFTER the RELEASE transition."""
        from .task_fsm import S
        if not self.recovery_enabled or not self.recovery_settings.get('release_guard', True):
            return None
        preopen = self.state == S.RELEASE and self.release_opened_t is None
        if self.state not in (S.TRANSPORT, S.DESCEND_SLOT) and not preopen:
            return None
        slot = obs.est.slot
        pending = slot.observation.get('motion_pending', False)
        changed = False
        if preopen and self.release_reference is not None and self._visual_ok(slot, obs.t):
            pos, yaw = self.release_reference
            changed = (np.linalg.norm(slot.pos - pos) > self.recovery_settings['release_change_m']
                       or abs(wrap_half_pi(slot.yaw-yaw)) > np.radians(self.recovery_settings['release_change_deg']))
        expired = (preopen and (slot.measurement_t is None or
                   obs.t-slot.measurement_t > self.pc['learned']['place_commit_max_age_s']+1e-9))
        if not (pending or changed or expired):
            self.pending_veto_active = False
            return None
        if not self.pending_veto_active:
            self.release_vetoes += 1
        self.pending_veto_active = True
        self.slot_hist.clear()  # old stationarity cannot qualify the new target
        self.t_in_tol = None
        reason = 'pending destination motion' if pending else ('destination changed before opening' if changed else 'release pose expired')
        if self.state != S.TRANSPORT:
            self._go(S.TRANSPORT, obs.t, 'release veto: '+reason)
        # Retract vertically first; an unconfirmed candidate is never a target.
        self.p_des = obs.ee_pos.copy()
        self.p_des[2] = max(self.p_des[2], self.table_z+self.p['carry_height'])
        self.gripper = float(self.g['closed_ctrl'])
        self.guard_action = 'release_veto_'+('pending_motion' if pending else 'changed_or_expired')
        self.stats.visual_guard_ticks += 1
        return self._cmd()

    def inspect_placement(self, obs):
        """Classify the estimated full capsule footprint, without target truth.

        A boundary band abstains. Height/tilt are visual plausibility checks,
        not proof of physical support. Capsule roll is unobservable and unused.
        """
        settings = self.recovery_settings
        cap, slot = obs.est.capsule, obs.est.slot
        if not (self._visual_ok(cap, obs.t) and self._visual_ok(slot, obs.t)):
            return 'unknown', None
        if slot.observation.get('motion_pending', False):
            return 'unknown', None
        if abs(cap.measurement_t-slot.measurement_t) > settings['pair_skew_s']:
            return 'unknown', None
        if min(cap.measurement_t, slot.measurement_t) < self.verification_started-1e-9:
            return 'unknown', None
        c, s = np.cos(slot.yaw), np.sin(slot.yaw)
        rotation = np.array([[c,s],[-s,c]])
        local = rotation @ (cap.pos[:2]-slot.pos[:2])
        axis = np.array([np.cos(cap.yaw),np.sin(cap.yaw),0.]) if cap.axis is None else np.asarray(cap.axis)
        axis = axis/(np.linalg.norm(axis)+1e-12)
        extent = settings['capsule_half_segment_m']*np.abs(rotation@axis[:2])+settings['capsule_radius_m']
        clearance = float(np.min(np.asarray(settings['cavity_m'])/2-np.abs(local)-extent))
        height_error = float(cap.pos[2]-slot.pos[2]-settings['capsule_radius_m'])
        tilt = float(np.degrees(np.arcsin(np.clip(abs(axis[2]),0,1))))
        detail = {'clearance_m':clearance, 'height_error_m':height_error, 'tilt_deg':tilt,
                  'capsule_position_m':cap.pos.tolist(), 'slot_position_m':slot.pos.tolist(),
                  'capsule_axis_world':axis.tolist(),
                  'capsule_yaw_rad':cap.yaw, 'slot_yaw_rad':slot.yaw,
                  'pair_t':min(cap.measurement_t,slot.measurement_t)}
        if (clearance >= settings['boundary_margin_m'] and abs(height_error) <= settings['height_tolerance_m']
                and tilt <= settings['max_tilt_deg']):
            return 'inside', detail
        if (clearance < -settings['boundary_margin_m'] or abs(height_error) > 2*settings['height_tolerance_m']
                or tilt > 2*settings['max_tilt_deg']):
            return 'outside', detail
        return 'ambiguous', detail

    def verification_tick(self, obs):
        from .task_fsm import S
        settings = self.recovery_settings
        if self.verification_started is None:
            self.verification_started = obs.t
        if self.verification_anchor is None:
            self.verification_anchor = obs.ee_pos.copy()
            self.verification_anchor[2] = max(obs.ee_pos[2],self.table_z+settings['view_height_m'])
        self.gripper = float(self.g['open_ctrl'])
        elapsed = obs.t-self.verification_started
        # A fixed, declared view schedule; independent of environment event times.
        view = int(elapsed/settings['view_dwell_s']) % len(settings['view_offsets_xy_m'])
        self.p_des = self.verification_anchor.copy()
        self.p_des[:2] += np.asarray(settings['view_offsets_xy_m'][view])
        if self.acquisition_enabled:
            self.p_des,self.q_des=self.observation_waypoint(obs,'capsule',view)
        status, detail = self.inspect_placement(obs)
        pair = None if detail is None else detail['pair_t']
        # Missing/ambiguous captures break evidence; cached poses add no samples.
        ambiguous_retry=(self.acquisition_enabled and status=='ambiguous'
                         and elapsed>=self.acquisition_settings.get('ambiguous_retry_after_s',float('inf')))
        eligible=status in ('inside','outside') or ambiguous_retry
        if not eligible:
            self.verification_samples.clear()
        if pair is not None and (self.verification_last_pair is None or pair > self.verification_last_pair+1e-9):
            self.verification_last_pair = pair
            self.verification_history.append({'t':obs.t, 'status':status, **detail})
            if eligible:
                self.verification_samples.append((pair,status,detail))
        while self.verification_samples and self.verification_samples[0][0] < obs.t-settings['evidence_history_s']:
            self.verification_samples.popleft()
        samples = list(self.verification_samples)
        stable = (len(samples) >= settings['minimum_samples']
                  and samples[-1][0]-samples[0][0] >= settings['stable_window_s']-1e-9
                  and all(row[1]==status for row in samples)
                  and all(b[0]-a[0] <= settings['max_sample_gap_s'] for a,b in zip(samples,samples[1:]))
                  and all(np.linalg.norm(np.asarray(row[2][key])-np.asarray(samples[-1][2][key])) <= settings['stable_position_m']
                          for row in samples for key in ('capsule_position_m','slot_position_m'))
                  and all(abs(wrap_half_pi(row[2][key]-samples[-1][2][key])) <= np.radians(settings['stable_yaw_deg'])
                          for row in samples for key in ('capsule_yaw_rad','slot_yaw_rad')))
        self.verification_status = status
        self.guard_action = 'verify_placement_'+status
        if stable and (status == 'outside' or ambiguous_retry):
            if self.placement_retries >= settings['max_retries']:
                self.verification_status = 'retry_limit'
                self._go(S.ABORT,obs.t,'visual placement retry limit')
            else:
                self.placement_retries += 1
                self.stats.num_regrasps += 1
                self.in_hand = None
                self.slot_hist.clear()
                self.reobserve_pos = None
                self.release_opened_t = None
                self.release_reference = None
                self.verification_started = None
                self.verification_anchor = None
                self.verification_samples.clear()
                self.verification_last_pair = None
                reason='visual placement uncertain: recenter' if ambiguous_retry else 'visual placement miss: retry from current visual capsule'
                self._go(S.APPROACH_OBJECT,obs.t,reason)
            return self._cmd() # stay clear and open on the retry transition
        if stable and status == 'inside' and elapsed >= settings['monitor_s']:
            if self.state != S.DONE:
                self._go(S.DONE,obs.t,'visually verified placement')
            return self._cmd()
        if self.state == S.DONE:
            self._go(S.VERIFY,obs.t,'visual placement evidence lost after verification')
        if elapsed >= settings['timeout_s']:
            self.verification_status = 'inconclusive'
            self._go(S.ABORT,obs.t,'visual placement verification inconclusive')
        return self._cmd()
