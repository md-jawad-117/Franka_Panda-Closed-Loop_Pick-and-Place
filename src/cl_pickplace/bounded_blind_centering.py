"""v22: bounded unobserved centering stroke after a stable fresh learned plan.

The v22 visibility assay (training/results/visibility-assay-v22) found that no
tested hand pose keeps the capsule detectable during the final ~4 cm of lowering
in any of the three development recovery states, and that the closed pusher
splits the wrist-camera capsule mask even from the inspection view that sees it
with the gripper open. v20/v21 require capsule age <=0.5 s through lowering and
pushing, so they can never complete the stroke.

This revision changes that one gate, explicitly:
- a plan still requires the unchanged stable, fresh, synchronized learned pair;
- missing capsule/fixture evidence is tolerated only while the time since the
  most recent fresh pair is within ``blind_budget_s``;
- any fresh *contradicting* evidence still stops the stroke (fixture moved or
  motion pending, capsule displaced before contact, capsule off the push
  corridor) exactly as before;
- the pusher path is unchanged CAD geometry above the fixture rim, with the
  unchanged bounded stroke and timeouts;
- completion is only ever declared by the unchanged settled visual inspection.
No cached pose is relabelled as a new measurement; evaluation truth is unused.
"""
import numpy as np
from .reobserving_centering import ReobservingCenterFSM
from .kinematics import gripper_down_quat,orientation_error,wrap_half_pi


class BoundedBlindCenterFSM(ReobservingCenterFSM):
    def reset(self,*args,**kwargs):
        super().reset(*args,**kwargs);self.centering_evidence_t=None

    def _fresh(self,est,obs,max_age):
        return est.measurement_t is not None and self._visual_ok(est,obs.t) and obs.t-est.measurement_t<=max_age

    def _centering_event(self,obs,event,**details):
        super()._centering_event(obs,event,**details)
        if event in ('planned','safe_replan') and self.centering_plan is not None:
            self.centering_evidence_t=float(self.centering_plan['observation_t'])
        elif event=='revalidated' and details.get('capture_times'):
            self.centering_evidence_t=float(max(details['capture_times']))

    def centering_tick(self,obs):
        phase=self.centering_phase;settings=self.centering_settings
        if phase=='observe' and obs.t-self.centering_phase_started>=settings['observation_timeout_s']:
            if self._blind_ok(obs):
                self._centering_event(obs,'blind_proceed',evidence_age_s=obs.t-self.centering_evidence_t)
                self._centering_phase(obs,'lower');return self._cmd()
        if phase not in ('lower','push'):return super().centering_tick(obs)
        return self._stroke_tick(obs,phase)

    def _blind_ok(self,obs):
        return self.centering_evidence_t is not None and obs.t-self.centering_evidence_t<=self.centering_settings['blind_budget_s']

    def _stroke_tick(self,obs,phase):
        settings=self.centering_settings;motion=settings['motion'];plan=self.centering_plan
        self.gripper=float(self.g['closed_ctrl']);self.guard_action='blind_centering_'+phase
        q=gripper_down_quat(self._yaw(plan['yaw_rad']));self.q_des=q
        start=np.asarray(plan['start']);finish=np.asarray(plan['finish'])
        slot,cap=obs.est.slot,obs.est.capsule
        slot_fresh=self._fresh(slot,obs,settings['fixture_max_age_s']);cap_fresh=self._fresh(cap,obs,settings['capsule_max_age_s'])
        if slot_fresh and cap_fresh and abs(cap.measurement_t-slot.measurement_t)<=self.recovery_settings['pair_skew_s']:
            self.centering_evidence_t=max(self.centering_evidence_t or -np.inf,min(cap.measurement_t,slot.measurement_t))
        # Contradicting evidence always stops the stroke.
        if slot.observation.get('motion_pending',False):
            self._centering_retract(obs,'destination motion pending');return super().centering_tick(obs)
        if slot_fresh and (np.linalg.norm(slot.pos-np.asarray(plan['goal_capsule_m']))>settings['fixture_change_m']
                or abs(wrap_half_pi(slot.yaw-plan['fixture_yaw_rad']))>np.radians(settings['fixture_change_deg'])):
            self._centering_retract(obs,'destination moved');return super().centering_tick(obs)
        if (not slot_fresh or not cap_fresh) and not self._blind_ok(obs):
            self._centering_retract(obs,'blind budget exhausted');return super().centering_tick(obs)
        if phase=='lower':
            if cap_fresh and np.linalg.norm(cap.pos[:2]-np.asarray(plan['initial_capsule_m'])[:2])>settings['revalidation_position_m']:
                self._centering_retract(obs,'capsule moved before contact');return super().centering_tick(obs)
            self.p_des=start.copy()
        else:
            direction=finish-start;distance=float(np.linalg.norm(direction));direction/=max(distance,1e-9)
            if cap_fresh:
                error=cap.pos[:2]-np.asarray(plan['goal_capsule_m'])[:2]
                n=direction[:2];lateral=float(np.linalg.norm(error-n*(error@n)))
                if lateral>settings['maximum_lateral_error_m'] or np.linalg.norm(error)>plan['initial_distance_m']+settings['maximum_error_growth_m']:
                    self._centering_retract(obs,'capsule deviated from push corridor');return super().centering_tick(obs)
                if slot_fresh:
                    c,s=np.cos(slot.yaw),np.sin(slot.yaw);R=np.array([[c,s],[-s,c]])
                    axis=np.asarray(cap.axis) if cap.axis is not None else np.array([np.cos(cap.yaw),np.sin(cap.yaw),0.])
                    extent=settings['plan_settings']['capsule_radius_m']+settings['plan_settings']['capsule_half_segment_m']*np.abs(R@axis[:2])
                    clearance=float(np.min(np.asarray(self.recovery_settings['cavity_m'])/2-np.abs(R@error)-extent))
                    if np.linalg.norm(error)<=settings['goal_position_m'] and clearance>=self.recovery_settings['boundary_margin_m']:
                        self._centering_retract(obs,'visual centering target reached');return super().centering_tick(obs)
            travel=min(distance+settings['maximum_overtravel_m'],motion['speed_m_s']*(obs.t-self.centering_phase_started))
            self.p_des=start+direction*travel
            if travel>=distance+settings['maximum_overtravel_m'] and np.linalg.norm(obs.ee_pos-self.p_des)<=motion['arrival_position_m']:
                self._centering_retract(obs,'bounded stroke exhausted: inspect');return super().centering_tick(obs)
        elapsed=obs.t-self.centering_phase_started
        if elapsed>=(motion['push_timeout_s'] if phase=='push' else motion['phase_timeout_s']):
            self._centering_retract(obs,phase+' motion deadline');return super().centering_tick(obs)
        if phase=='lower' and (np.linalg.norm(obs.ee_pos-self.p_des)<=motion['arrival_position_m'] and
                np.degrees(np.linalg.norm(orientation_error(self.q_des,obs.ee_mat)))<=motion['arrival_rotation_deg']):
            self._centering_phase(obs,'push')
        return self._cmd()
