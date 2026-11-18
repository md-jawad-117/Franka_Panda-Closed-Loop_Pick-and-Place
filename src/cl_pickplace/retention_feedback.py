"""Opt-in v10 retention experiment: learned positions and robot readings only.

This module has no simulation/contact API. Absence of a visual measurement is
unknown, never a positive grasp label. The unchanged v4 controller remains the
default. These finite thresholds are development settings, not calibrated bounds.
"""
from collections import deque
import numpy as np
from .task_fsm import TaskFSM, S
from .kinematics import gripper_down_quat, axis_yaw_from_mat


class RetentionFSM(TaskFSM):
    @property
    def retention_settings(self):return self.p['retention_feedback']

    def reset(self, start_state=S.HOME):
        super().reset(start_state)
        self.retention_history=[];self.retention_retries=0
        self.retention_phase='inactive';self.retention_anchor=None
        self.retention_reference=None;self.retention_samples=deque()
        self.retention_started=None;self.retention_last_seen=None
        self.retention_bad_since=None;self.retention_last_sample=None
        self.retention_mismatch=0;self.retention_view_yaw=None
        self.retention_probe_origin=None;self.retention_probe_cap=None
        self.retention_last_ground_sample=None;self.retention_failure=None
        self.retention_retract=None;self.retention_grasp_origin=None

    def _retention_event(self,obs,event,**details):
        self.retention_history.append({'t':obs.t,'event':event,'state':self.state.name,**details})

    def _new_capsule(self,obs):
        cap=obs.est.capsule
        valid=(cap.source.startswith('learned:') and cap.visible and cap.lost_frames==0
               and cap.measurement_t is not None and abs(cap.measurement_t-obs.t)<1e-7
               and np.all(np.isfinite(cap.pos)))
        return valid and (self.retention_last_sample is None or cap.measurement_t>self.retention_last_sample+1e-9)

    def _width_failure(self,obs):
        if self._width_ok(obs):self.retention_bad_since=None;return False
        if self.retention_bad_since is None:self.retention_bad_since=obs.t
        return obs.t-self.retention_bad_since>=self.retention_settings['width_debounce_s']-1e-9

    def _abort_retention(self,obs,reason):
        self.retention_phase='aborted';self.retention_failure=reason
        self._retention_event(obs,'abort',reason=reason)
        self.p_des=obs.ee_pos.copy();self.gripper=float(self.g['closed_ctrl'])
        self._go(S.ABORT,obs.t,'retention: '+reason)
        return self._cmd()

    def _retry_retention(self,obs,reason):
        if self.retention_retries>=self.retention_settings['maximum_retries']:
            return self._abort_retention(obs,'retry budget exhausted: '+reason)
        self.retention_retries+=1;self.stats.num_regrasps+=1
        self._retention_event(obs,'retry',reason=reason,attempt=self.retention_retries)
        self.retention_phase='retract_empty';self.retention_failure=reason
        self.retention_retract=obs.ee_pos.copy()
        self.retention_retract[2]=max(obs.ee_pos[2],self.table_z+self.p['carry_height'])
        self.retention_started=obs.t
        return self._retention_retract_tick(obs)

    def _retention_retract_tick(self,obs):
        self.p_des=self.retention_retract.copy();self.gripper=float(self.g['closed_ctrl'])
        self.guard_action='retention_retract_empty'
        if np.linalg.norm(obs.ee_pos-self.p_des)<=self.retention_settings['arrival_m']:
            self.retention_phase='inactive';self.retention_reference=None
            self.gripper=float(self.g['open_ctrl'])
            self._go(S.APPROACH_OBJECT,obs.t,'object dropped: retention feedback confirmed empty')
            self.acquisition_needed=True
            self._retention_event(obs,'reacquire')
        elif obs.t-self.retention_started>self.retention_settings['retract_timeout_s']:
            return self._abort_retention(obs,'empty-hand retraction timeout')
        return self._cmd()

    def _begin_probe(self,obs):
        settings=self.retention_settings
        self.retention_phase='probe';self.retention_started=obs.t
        self.retention_probe_origin=(obs.ee_pos.copy() if self.retention_grasp_origin is None
                                     else self.retention_grasp_origin.copy())
        self.retention_anchor=self.retention_probe_origin+np.array([0.,0.,settings['probe_height_m']])
        self.retention_view_yaw=axis_yaw_from_mat(obs.ee_mat)
        self.retention_probe_cap=self.retention_last_ground_sample
        self.retention_samples.clear();self.retention_bad_since=None
        self.retention_last_sample=None;self.retention_reference=None
        self._retention_event(obs,'probe_started')

    def _sample_retention(self,obs):
        if not self._new_capsule(obs):return None
        cap=obs.est.capsule;self.retention_last_sample=cap.measurement_t
        local=obs.ee_mat.T@(cap.pos-obs.ee_pos)
        distance=float(np.linalg.norm(local));self.retention_last_seen=obs.t
        self._retention_event(obs,'visual_sample',capsule_position_m=cap.pos.tolist(),
                              capsule_in_tcp_m=local.tolist(),finger_width_m=obs.finger_width)
        return local,distance

    def _probe_tick(self,obs):
        settings=self.retention_settings
        self.p_des=self.retention_anchor.copy();self.gripper=float(self.g['closed_ctrl'])
        self.guard_action='retention_test_lift'
        if self._width_failure(obs):return self._retry_retention(obs,'empty width during test lift')
        sample=self._sample_retention(obs)
        arrived=np.linalg.norm(obs.ee_pos-self.retention_anchor)<=settings['arrival_m']
        if arrived and sample is not None:
            local,distance=sample
            prior=self.retention_probe_cap
            # The pre-close reference must itself be a recent learned observation.
            origin_valid=(prior is not None and 0<=self.retention_started-prior[0]<=settings['preclose_reference_max_age_s'])
            rise=None if not origin_valid else float(obs.est.capsule.pos[2]-prior[1][2])
            lifted=origin_valid and rise>=settings['minimum_visual_rise_m']
            if distance<=settings['maximum_tcp_distance_m'] and lifted:
                if self.retention_samples and np.linalg.norm(local-self.retention_samples[-1][1])>settings['relative_drift_m']:
                    self.retention_samples.clear()
                self.retention_samples.append((obs.t,local.copy()))
                if len(self.retention_samples)>=settings['confirmation_samples']:
                    self.retention_reference=np.mean([x[1] for x in self.retention_samples],axis=0)
                    self.retention_phase='carrying';self.retention_mismatch=0
                    self._retention_event(obs,'probe_confirmed',visual_rise_m=rise)
                    self.hold_pos=self.retention_probe_origin.copy()
                    return super().update(obs)
            else:
                self.retention_samples.clear()
                self._retention_event(obs,'probe_not_confirmed',visual_rise_m=rise,distance_m=distance)
                if distance>settings['maximum_tcp_distance_m']:
                    return self._retry_retention(obs,'visual capsule detached during test lift')
        elapsed=obs.t-self.retention_started
        if arrived:
            view=int(elapsed/settings['view_dwell_s'])%len(settings['view_yaw_offsets_deg'])
            self.q_des=gripper_down_quat(self.retention_view_yaw+np.radians(settings['view_yaw_offsets_deg'][view]))
        if elapsed>=settings['probe_timeout_s']:
            return self._abort_retention(obs,'test lift visually unconfirmed')
        return self._cmd()

    def _carry_guard(self,obs):
        settings=self.retention_settings
        if self._width_failure(obs):return self._retry_retention(obs,'empty width while carrying')
        sample=self._sample_retention(obs)
        if sample is not None:
            local,distance=sample
            drift=float(np.linalg.norm(local-self.retention_reference))
            self.retention_mismatch=self.retention_mismatch+1 if drift>settings['relative_drift_m'] else 0
            if distance>settings['maximum_tcp_distance_m']:
                return self._retry_retention(obs,'visual capsule detached while carrying')
            if self.retention_mismatch>=settings['confirmation_samples']:
                return self._abort_retention(obs,'visual in-hand drift; grasp uncertain')
        age=obs.t-self.retention_last_seen
        if age>settings['carry_visual_max_age_s']:
            if self.retention_phase!='observe_hold':
                self.retention_phase='observe_hold';self.retention_anchor=obs.ee_pos.copy()
                self.retention_started=obs.t;self.retention_view_yaw=axis_yaw_from_mat(obs.ee_mat)
                self._retention_event(obs,'observation_hold')
            self.p_des=self.retention_anchor.copy();self.gripper=float(self.g['closed_ctrl'])
            view=int((obs.t-self.retention_started)/settings['view_dwell_s'])%len(settings['view_yaw_offsets_deg'])
            self.q_des=gripper_down_quat(self.retention_view_yaw+np.radians(settings['view_yaw_offsets_deg'][view]))
            self.guard_action='retention_observe_hold'
            if obs.t-self.retention_started>=settings['carry_observe_timeout_s']:
                return self._abort_retention(obs,'carrying visibility exhausted')
            return self._cmd()
        if self.retention_phase=='observe_hold':
            # Resume only if the fresh observation also agrees with the reference.
            if self.retention_mismatch:return self._abort_retention(obs,'drift after occlusion')
            self.retention_phase='carrying';self._retention_event(obs,'observation_resumed')
        return None

    def update(self,obs):
        # Explicit opt-in, with identical v4 behavior when disabled/open loop.
        if not self.p.get('retention_feedback') or not (self.closed_loop and self.require_visual_observations):
            return super().update(obs)
        if self.retention_phase=='aborted':
            self.p_des=obs.ee_pos.copy();self.gripper=float(self.g['closed_ctrl']);return self._cmd()
        if self.retention_phase=='retract_empty':return self._retention_retract_tick(obs)
        if self.state in (S.HOME,S.APPROACH_OBJECT,S.DESCEND_OBJECT) and self._new_capsule(obs):
            self.retention_last_sample=obs.est.capsule.measurement_t
            self.retention_last_ground_sample=(obs.t,obs.est.capsule.pos.copy())
        if self.state==S.GRASP:self.retention_grasp_origin=obs.ee_pos.copy()
        if self.state==S.LIFT and self.retention_phase not in ('probe','carrying','observe_hold'):
            self._begin_probe(obs)
        if self.state==S.LIFT and self.retention_phase=='probe':return self._probe_tick(obs)
        if self.state in (S.LIFT,S.TRANSPORT,S.DESCEND_SLOT) and self.retention_phase in ('carrying','observe_hold'):
            guard=self._carry_guard(obs)
            if guard is not None:return guard
        command=super().update(obs)
        if self.state not in (S.GRASP,S.LIFT,S.TRANSPORT,S.DESCEND_SLOT) and self.retention_phase in ('carrying','observe_hold'):
            self.retention_phase='inactive';self.retention_reference=None
        return command
