"""Resolve the safe-view reacquisition deadlock before a contact correction.

Known safe-height waypoints may be approached while the capsule is occluded.
Lowering still requires distinct, current learned observations in that new view.
No cached pose is relabelled or substituted for a new camera measurement.
"""
import numpy as np
from .centering_feedback import CenteringFSM
from .visual_centering import plan_centering
from .kinematics import gripper_down_quat,orientation_error,wrap_half_pi


class ReobservingCenterFSM(CenteringFSM):
    def reset(self,*args,**kwargs):
        super().reset(*args,**kwargs)
        self.centering_observe_samples=[];self.centering_observe_last=None;self.centering_replans=0

    def _centering_phase(self,obs,phase):
        super()._centering_phase(obs,phase)
        if phase=='raise':self.centering_replans=0
        if phase=='observe':self.centering_observe_samples=[];self.centering_observe_last=None

    def centering_tick(self,obs):
        phase=self.centering_phase
        if phase not in ('approach','observe'):return super().centering_tick(obs)
        settings=self.centering_settings;motion=settings['motion'];plan=self.centering_plan
        self.gripper=float(self.g['closed_ctrl']);self.q_des=gripper_down_quat(self._yaw(plan['yaw_rad']))
        self.p_des=np.r_[np.asarray(plan['start'])[:2],plan['clear_z']]
        self.guard_action='centering_safe_'+phase
        elapsed=obs.t-self.centering_phase_started
        arrived=(np.linalg.norm(obs.ee_pos-self.p_des)<=motion['arrival_position_m'] and
            np.degrees(np.linalg.norm(orientation_error(self.q_des,obs.ee_mat)))<=motion['arrival_rotation_deg'])
        if phase=='approach':
            if arrived:self._centering_phase(obs,'observe')
            elif elapsed>=motion['phase_timeout_s']:
                self._centering_retract(obs,'safe observation waypoint unreachable');return super().centering_tick(obs)
            return self._cmd()
        if not arrived:
            self.centering_observe_samples=[];self._centering_phase(obs,'approach');return self._cmd()
        cap,slot=obs.est.capsule,obs.est.slot
        synchronized=(self._visual_ok(cap,obs.t) and self._visual_ok(slot,obs.t)
            and cap.measurement_t>=self.centering_phase_started-1e-9
            and abs(cap.measurement_t-slot.measurement_t)<=self.recovery_settings['pair_skew_s']
            and not slot.observation.get('motion_pending',False)
            and slot.quality.get('estimator')=='observed_support_fixture_v17')
        if synchronized and (self.centering_observe_last is None or cap.measurement_t>self.centering_observe_last+1e-9):
            self.centering_observe_last=cap.measurement_t
            self.centering_observe_samples.append((cap.measurement_t,cap.pos.copy(),slot.pos.copy(),cap.yaw,slot.yaw))
            self.centering_observe_samples=self.centering_observe_samples[-settings['observation_samples']:]
        samples=self.centering_observe_samples
        stable=(synchronized and len(samples)>=settings['observation_samples'] and samples[-1][0]-samples[0][0]>=settings['observation_span_s']-1e-9
            and all(b[0]-a[0]<=self.recovery_settings['max_sample_gap_s']+1e-9 for a,b in zip(samples,samples[1:]))
            and all(np.linalg.norm(r[k]-samples[-1][k])<=self.recovery_settings['stable_position_m'] for r in samples for k in (1,2))
            and all(abs(wrap_half_pi(r[k]-samples[-1][k]))<=np.radians(self.recovery_settings['stable_yaw_deg']) for r in samples for k in (3,4)))
        if stable:
            agrees=(np.linalg.norm(cap.pos-np.asarray(plan['initial_capsule_m']))<=settings['revalidation_position_m']
                and np.linalg.norm(slot.pos-np.asarray(plan['goal_capsule_m']))<=settings['revalidation_position_m']
                and abs(wrap_half_pi(slot.yaw-plan['fixture_yaw_rad']))<=np.radians(settings['fixture_change_deg']))
            if agrees:
                self._centering_event(obs,'revalidated',capture_times=[r[0] for r in samples],original_plan_preserved=True)
                self._centering_phase(obs,'lower');return self._cmd()
            decision=plan_centering(obs,self.p,settings['plan_settings'])
            if decision['selected'] is not None and self.centering_replans<settings['maximum_safe_replans']:
                self.centering_replans+=1;self.centering_plan=decision['selected']
                self._centering_event(obs,'safe_replan',plan=self.centering_plan)
                self._centering_phase(obs,'approach');return self._cmd()
            self._centering_retract(obs,'safe-view target changed beyond replan budget');return super().centering_tick(obs)
        if elapsed>=settings['observation_timeout_s']:
            self._centering_retract(obs,'no stable new observation at safe waypoint');return super().centering_tick(obs)
        return self._cmd()
