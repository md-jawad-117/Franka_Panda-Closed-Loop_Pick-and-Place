"""v23: capsule-displacement guards need consistent repeated fresh evidence.

v22 (docs/BLIND_CENTERING_V22.md) stopped centering strokes on single fresh but
partly occluded capsule detections that were 6-20 mm wrong while the capsule had
not moved (seed 605 pre-contact guard, twice; seed 600 push-corridor guard).
Here the two capsule-displacement guards trip only after ``alarm_samples``
distinct fresh detections violate the guard and agree with each other within
``alarm_consistency_m``; a fresh non-violating detection clears the alarm.
Fixture guards, the blind budget, CAD path, stroke bound, timeouts and the
settled-inspection completion rule are the unchanged v22 ones. v22 source is
not modified, so its frozen cohort remains verifiable.
"""
import numpy as np
from .bounded_blind_centering import BoundedBlindCenterFSM
from .reobserving_centering import ReobservingCenterFSM
from .kinematics import gripper_down_quat,orientation_error,wrap_half_pi


class ConfirmedBlindCenterFSM(BoundedBlindCenterFSM):
    def reset(self,*args,**kwargs):
        super().reset(*args,**kwargs);self.capsule_alarms=[]

    def _centering_phase(self,obs,phase):
        super()._centering_phase(obs,phase)
        if phase in ('lower','push'):self.capsule_alarms=[]

    def _capsule_alarm(self,cap,violated):
        """Record one fresh capsule detection; True only for confirmed displacement."""
        t=float(cap.measurement_t)
        if self.capsule_alarms and t<=self.capsule_alarms[-1][0]+1e-9:return False
        if not violated:self.capsule_alarms=[];return False
        settings=self.centering_settings;self.capsule_alarms.append((t,np.asarray(cap.pos,float).copy()))
        self.capsule_alarms=self.capsule_alarms[-settings['alarm_samples']:]
        if len(self.capsule_alarms)<settings['alarm_samples']:return False
        last=self.capsule_alarms[-1][1]
        return all(np.linalg.norm(p[:2]-last[:2])<=settings['alarm_consistency_m'] for _,p in self.capsule_alarms)

    def _stroke_tick(self,obs,phase):
        settings=self.centering_settings;motion=settings['motion'];plan=self.centering_plan
        self.gripper=float(self.g['closed_ctrl']);self.guard_action='confirmed_blind_centering_'+phase
        q=gripper_down_quat(self._yaw(plan['yaw_rad']));self.q_des=q
        start=np.asarray(plan['start']);finish=np.asarray(plan['finish'])
        slot,cap=obs.est.slot,obs.est.capsule
        slot_fresh=self._fresh(slot,obs,settings['fixture_max_age_s']);cap_fresh=self._fresh(cap,obs,settings['capsule_max_age_s'])
        if slot_fresh and cap_fresh and abs(cap.measurement_t-slot.measurement_t)<=self.recovery_settings['pair_skew_s']:
            self.centering_evidence_t=max(self.centering_evidence_t or -np.inf,min(cap.measurement_t,slot.measurement_t))
        # Contradicting evidence always stops the stroke.
        if slot.observation.get('motion_pending',False):
            self._centering_retract(obs,'destination motion pending');return ReobservingCenterFSM.centering_tick(self,obs)
        if slot_fresh and (np.linalg.norm(slot.pos-np.asarray(plan['goal_capsule_m']))>settings['fixture_change_m']
                or abs(wrap_half_pi(slot.yaw-plan['fixture_yaw_rad']))>np.radians(settings['fixture_change_deg'])):
            self._centering_retract(obs,'destination moved');return ReobservingCenterFSM.centering_tick(self,obs)
        if (not slot_fresh or not cap_fresh) and not self._blind_ok(obs):
            self._centering_retract(obs,'blind budget exhausted');return ReobservingCenterFSM.centering_tick(self,obs)
        if phase=='lower':
            if cap_fresh and self._capsule_alarm(cap,np.linalg.norm(cap.pos[:2]-np.asarray(plan['initial_capsule_m'])[:2])>settings['revalidation_position_m']):
                self._centering_retract(obs,'capsule moved before contact');return ReobservingCenterFSM.centering_tick(self,obs)
            self.p_des=start.copy()
        else:
            direction=finish-start;distance=float(np.linalg.norm(direction));direction/=max(distance,1e-9)
            if cap_fresh:
                error=cap.pos[:2]-np.asarray(plan['goal_capsule_m'])[:2]
                n=direction[:2];lateral=float(np.linalg.norm(error-n*(error@n)))
                if self._capsule_alarm(cap,lateral>settings['maximum_lateral_error_m'] or np.linalg.norm(error)>plan['initial_distance_m']+settings['maximum_error_growth_m']):
                    self._centering_retract(obs,'capsule deviated from push corridor');return ReobservingCenterFSM.centering_tick(self,obs)
                if slot_fresh:
                    c,s=np.cos(slot.yaw),np.sin(slot.yaw);R=np.array([[c,s],[-s,c]])
                    axis=np.asarray(cap.axis) if cap.axis is not None else np.array([np.cos(cap.yaw),np.sin(cap.yaw),0.])
                    extent=settings['plan_settings']['capsule_radius_m']+settings['plan_settings']['capsule_half_segment_m']*np.abs(R@axis[:2])
                    clearance=float(np.min(np.asarray(self.recovery_settings['cavity_m'])/2-np.abs(R@error)-extent))
                    if np.linalg.norm(error)<=settings['goal_position_m'] and clearance>=self.recovery_settings['boundary_margin_m']:
                        self._centering_retract(obs,'visual centering target reached');return ReobservingCenterFSM.centering_tick(self,obs)
            travel=min(distance+settings['maximum_overtravel_m'],motion['speed_m_s']*(obs.t-self.centering_phase_started))
            self.p_des=start+direction*travel
            if travel>=distance+settings['maximum_overtravel_m'] and np.linalg.norm(obs.ee_pos-self.p_des)<=motion['arrival_position_m']:
                self._centering_retract(obs,'bounded stroke exhausted: inspect');return ReobservingCenterFSM.centering_tick(self,obs)
        elapsed=obs.t-self.centering_phase_started
        if elapsed>=(motion['push_timeout_s'] if phase=='push' else motion['phase_timeout_s']):
            self._centering_retract(obs,phase+' motion deadline');return ReobservingCenterFSM.centering_tick(self,obs)
        if phase=='lower' and (np.linalg.norm(obs.ee_pos-self.p_des)<=motion['arrival_position_m'] and
                np.degrees(np.linalg.norm(orientation_error(self.q_des,obs.ee_mat)))<=motion['arrival_rotation_deg']):
            self._centering_phase(obs,'push')
        return self._cmd()
