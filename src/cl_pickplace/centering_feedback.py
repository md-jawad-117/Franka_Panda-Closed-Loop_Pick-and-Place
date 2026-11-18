"""Bounded object-centric centering, followed by fresh visual verification.

The policy sees only detached learned estimates and measured robot pose. Contacts,
forces, support/containment truth and independent evaluator results are unavailable.
"""
from collections import deque
import numpy as np
from .settled_inspection import InspectionFSM
from .visual_centering import plan_centering
from .task_fsm import S
from .kinematics import gripper_down_quat,mat_to_quat,orientation_error,wrap_half_pi


class CenteringFSM(InspectionFSM):
    @property
    def centering_settings(self):return self.p['visual_centering']

    def reset(self,start_state=S.HOME):
        super().reset(start_state)
        self.centering_active=False;self.centering_attempts=0;self.centering_history=[]
        self.centering_plan=None;self.centering_phase=None;self.centering_phase_started=None
        self.centering_origin=None;self.centering_raise_quat=None;self.centering_retreat=None
        self.centering_failure=None;self.margin_samples=deque();self.margin_last_pair=None;self.margin_view=None

    def _centering_event(self,obs,event,**details):
        self.centering_history.append({'t':obs.t,'event':event,**details})

    def _try_centering(self,obs,reason):
        settings=self.centering_settings
        if self.centering_attempts>=settings['maximum_attempts']:return None
        if obs.est.slot.quality.get('estimator')!='observed_support_fixture_v17':return None
        decision=plan_centering(obs,self.p,settings['plan_settings'])
        if decision['selected'] is None:
            self._centering_event(obs,'abstain',reason=decision['reason']);return None
        self.centering_plan=decision['selected'];self.centering_attempts+=1;self.centering_active=True
        self.centering_origin=obs.ee_pos.copy();self.centering_raise_quat=mat_to_quat(obs.ee_mat)
        self.centering_failure=None;self.centering_retreat=None
        self._go(S.EXTRACT,obs.t,'visual centering: '+reason)
        self._centering_phase(obs,'raise')
        self._centering_event(obs,'planned',attempt=self.centering_attempts,reason=reason,plan=self.centering_plan)
        return self.centering_tick(obs)

    def _inspection_retry(self,obs):
        if self.p.get('visual_centering'):
            cmd=self._try_centering(obs,'stable visual outside evidence')
            if cmd is not None:return cmd
        return super()._inspection_retry(obs)

    def verification_tick(self,obs):
        cmd=super().verification_tick(obs)
        if not self.p.get('visual_centering') or self.centering_active or self.state not in (S.VERIFY,S.DONE):return cmd
        if self.margin_view!=self.inspection_view:
            self.margin_view=self.inspection_view;self.margin_samples.clear();self.margin_last_pair=None
        if not self.verification_history:return cmd
        row=self.verification_history[-1];pair=row.get('pair_t')
        if abs(row['t']-obs.t)>1e-7 or pair is None:return cmd
        if self.margin_last_pair is not None and pair<=self.margin_last_pair+1e-9:return cmd
        self.margin_last_pair=pair;settings=self.recovery_settings
        borderline=(row['status']=='ambiguous' and abs(row['clearance_m'])<settings['boundary_margin_m']
                    and abs(row['height_error_m'])<=settings['height_tolerance_m'] and row['tilt_deg']<=settings['max_tilt_deg'])
        if borderline:self.margin_samples.append((pair,'ambiguous',row))
        else:self.margin_samples.clear()
        while self.margin_samples and self.margin_samples[0][0]<obs.t-settings['evidence_history_s']:self.margin_samples.popleft()
        # Reuse the unchanged distinct-capture/pose-stability thresholds.
        previous=self.inspection_samples;self.inspection_samples=self.margin_samples
        stable=self._inspection_stable('ambiguous');self.inspection_samples=previous
        if stable and obs.t-self.verification_started>=self.centering_settings['ambiguous_wait_s']:
            correction=self._try_centering(obs,'stable uncertain clearance: recenter')
            if correction is not None:return correction
        return cmd

    def _centering_phase(self,obs,phase):
        self.centering_phase=phase;self.centering_phase_started=obs.t
        self._centering_event(obs,'phase',phase=phase)

    def _centering_retract(self,obs,reason):
        self.centering_failure=reason;self.centering_retreat=obs.ee_pos.copy()
        self.centering_retreat[2]=max(self.centering_retreat[2],self.centering_plan['clear_z'])
        self._centering_event(obs,'stop_and_reobserve',reason=reason);self._centering_phase(obs,'retreat')

    def _finish_centering(self,obs):
        self._centering_event(obs,'trajectory_complete',uncertainty=self.centering_failure)
        self.centering_active=False;self.gripper=float(self.g['open_ctrl'])
        self.verification_started=None;self.inspection_pose=None;self.inspection_confirmed_t=None
        self.inspection_samples.clear();self.margin_samples.clear();self.margin_last_pair=None
        self._go(S.VERIFY,obs.t,'centering trajectory complete: fresh visual inspection')
        return self._cmd()

    def centering_tick(self,obs):
        settings=self.centering_settings;motion=settings['motion'];plan=self.centering_plan;phase=self.centering_phase
        self.gripper=float(self.g['closed_ctrl']);self.guard_action='visual_centering_'+phase
        q=gripper_down_quat(self._yaw(plan['yaw_rad']))
        start=np.asarray(plan['start']);finish=np.asarray(plan['finish']);safe=plan['clear_z']
        if phase in ('approach','lower'):
            slot=obs.est.slot;cap=obs.est.capsule
            age=float('inf') if slot.measurement_t is None else obs.t-slot.measurement_t
            cap_age=float('inf') if cap.measurement_t is None else obs.t-cap.measurement_t
            changed=(self._visual_ok(slot,obs.t) and (np.linalg.norm(slot.pos-np.asarray(plan['goal_capsule_m']))>settings['fixture_change_m']
                or abs(wrap_half_pi(slot.yaw-plan['fixture_yaw_rad']))>np.radians(settings['fixture_change_deg'])))
            if slot.observation.get('motion_pending',False) or age>settings['fixture_max_age_s'] or changed or cap_age>settings['capsule_max_age_s']:
                self._centering_retract(obs,'approach observation changed or expired');return self.centering_tick(obs)
        if phase=='raise':self.p_des=np.r_[self.centering_origin[:2],safe];self.q_des=self.centering_raise_quat.copy()
        elif phase=='orient':self.p_des=np.r_[self.centering_origin[:2],safe];self.q_des=q
        elif phase=='approach':self.p_des=np.r_[start[:2],safe];self.q_des=q
        elif phase=='lower':self.p_des=start.copy();self.q_des=q
        elif phase=='push':
            self.q_des=q
            slot=obs.est.slot;cap=obs.est.capsule
            age=float('inf') if slot.measurement_t is None else obs.t-slot.measurement_t
            changed=(self._visual_ok(slot,obs.t) and (np.linalg.norm(slot.pos-np.asarray(plan['goal_capsule_m']))>settings['fixture_change_m']
                or abs(wrap_half_pi(slot.yaw-plan['fixture_yaw_rad']))>np.radians(settings['fixture_change_deg'])))
            if slot.observation.get('motion_pending',False) or age>settings['fixture_max_age_s'] or changed:
                self._centering_retract(obs,'destination moved or unobserved');return self.centering_tick(obs)
            cap_age=float('inf') if cap.measurement_t is None else obs.t-cap.measurement_t
            if cap_age>settings['capsule_max_age_s']:
                self._centering_retract(obs,'capsule unobserved');return self.centering_tick(obs)
            direction=(finish-start);distance=float(np.linalg.norm(direction));direction/=max(distance,1e-9)
            if self._capsule_seen(obs):
                error=cap.pos[:2]-np.asarray(plan['goal_capsule_m'])[:2]
                n=direction[:2];lateral=float(np.linalg.norm(error-n*(error@n)))
                if lateral>settings['maximum_lateral_error_m'] or np.linalg.norm(error)>plan['initial_distance_m']+settings['maximum_error_growth_m']:
                    self._centering_retract(obs,'capsule deviated from push corridor');return self.centering_tick(obs)
                c,s=np.cos(slot.yaw),np.sin(slot.yaw);R=np.array([[c,s],[-s,c]])
                axis=np.asarray(cap.axis) if cap.axis is not None else np.array([np.cos(cap.yaw),np.sin(cap.yaw),0.])
                extent=settings['plan_settings']['capsule_radius_m']+settings['plan_settings']['capsule_half_segment_m']*np.abs(R@axis[:2])
                clearance=float(np.min(np.asarray(self.recovery_settings['cavity_m'])/2-np.abs(R@error)-extent))
                if np.linalg.norm(error)<=settings['goal_position_m'] and clearance>=self.recovery_settings['boundary_margin_m']:
                    self._centering_retract(obs,'visual centering target reached');return self.centering_tick(obs)
            travel=min(distance+settings['maximum_overtravel_m'],motion['speed_m_s']*(obs.t-self.centering_phase_started))
            self.p_des=start+direction*travel
            if travel>=distance+settings['maximum_overtravel_m'] and np.linalg.norm(obs.ee_pos-self.p_des)<=motion['arrival_position_m']:
                self._centering_retract(obs,'bounded stroke exhausted: inspect');return self.centering_tick(obs)
        else:
            self.q_des=q
            self.p_des=np.r_[finish[:2],safe] if self.centering_retreat is None else self.centering_retreat.copy()
        elapsed=obs.t-self.centering_phase_started
        limit=motion['push_timeout_s'] if phase=='push' else motion['phase_timeout_s']
        if elapsed>=limit:
            if phase=='retreat':
                self.centering_active=False;self.gripper=float(self.g['closed_ctrl']);self.p_des=obs.ee_pos.copy()
                self._go(S.ABORT,obs.t,'centering retreat timeout');return self._cmd()
            self._centering_retract(obs,phase+' motion deadline');return self.centering_tick(obs)
        if phase!='push':
            arrived=(np.linalg.norm(obs.ee_pos-self.p_des)<=motion['arrival_position_m'] and
                     np.degrees(np.linalg.norm(orientation_error(self.q_des,obs.ee_mat)))<=motion['arrival_rotation_deg'])
            if arrived:
                phases=['raise','orient','approach','lower','push','retreat'];index=phases.index(phase)
                if phase=='retreat':return self._finish_centering(obs)
                self._centering_phase(obs,phases[index+1])
        return self._cmd()

    def update(self,obs):
        if self.centering_active:return self.centering_tick(obs)
        return super().update(obs)
