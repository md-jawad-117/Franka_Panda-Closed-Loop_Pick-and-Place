"""Stop-and-observe placement inspection using only the existing detached Obs.

Keeps containment/pose/stability thresholds unchanged. Camera motion is excluded
from evidence collection. A confirmed result has a short explicit freshness bound;
missing geometry is unknown, never a new positive observation.
"""
from collections import deque
import numpy as np
from .retention_feedback_v10r1 import RetentionFSM
from .task_fsm import S
from .kinematics import mat_to_quat,orientation_error,wrap_half_pi


class InspectionFSM(RetentionFSM):
    @property
    def inspection_settings(self):return self.p['settled_inspection']

    def reset(self,start_state=S.HOME):
        super().reset(start_state)
        self.inspection_pose=None;self.inspection_view=0;self.inspection_view_started=None
        self.inspection_arrived=None;self.inspection_last_capture=None
        self.inspection_confirmed_t=None;self.inspection_history=[]
        self.inspection_samples=deque()

    def _inspection_new_view(self,obs):
        self.inspection_samples.clear();self.inspection_arrived=None
        self.inspection_view_started=obs.t;self.inspection_last_capture=None
        if self.inspection_view==0:
            self.inspection_pose=(obs.ee_pos.copy(),mat_to_quat(obs.ee_mat))
        else:
            p,q=self.observation_waypoint(obs,'capsule',self.inspection_view-1)
            self.inspection_pose=(p.copy(),q.copy())
        self.inspection_history.append({'t':obs.t,'event':'view','view':self.inspection_view,
            'position_m':self.inspection_pose[0].tolist(),'quaternion':self.inspection_pose[1].tolist()})

    def _inspection_retry(self,obs):
        settings=self.recovery_settings
        if self.placement_retries>=settings['max_retries']:
            self.verification_status='retry_limit';self._go(S.ABORT,obs.t,'visual placement retry limit')
            return self._cmd()
        self.placement_retries+=1;self.stats.num_regrasps+=1;self.in_hand=None
        self.slot_hist.clear();self.reobserve_pos=None;self.release_opened_t=None;self.release_reference=None
        self.verification_started=None;self.verification_anchor=None;self.verification_samples.clear();self.verification_last_pair=None
        self.inspection_pose=None;self.inspection_confirmed_t=None;self.inspection_samples.clear()
        self._go(S.APPROACH_OBJECT,obs.t,'visual placement miss: retry from settled visual evidence')
        return self._cmd()

    def _inspection_stable(self,status):
        rows=list(self.inspection_samples);settings=self.recovery_settings
        return (len(rows)>=settings['minimum_samples']
            and rows[-1][0]-rows[0][0]>=settings['stable_window_s']-1e-9
            and all(r[1]==status for r in rows)
            and all(b[0]-a[0]<=settings['max_sample_gap_s']+1e-9 for a,b in zip(rows,rows[1:]))
            and all(np.linalg.norm(np.asarray(r[2][key])-np.asarray(rows[-1][2][key]))<=settings['stable_position_m']
                    for r in rows for key in ('capsule_position_m','slot_position_m'))
            and all(abs(wrap_half_pi(r[2][key]-rows[-1][2][key]))<=np.radians(settings['stable_yaw_deg'])
                    for r in rows for key in ('capsule_yaw_rad','slot_yaw_rad')))

    def verification_tick(self,obs):
        if not self.p.get('settled_inspection'):return super().verification_tick(obs)
        settings=self.inspection_settings;recovery=self.recovery_settings
        if self.verification_started is None:
            self.verification_started=obs.t;self.inspection_view=0;self.inspection_pose=None
            self.inspection_samples.clear();self.inspection_confirmed_t=None
        if self.inspection_pose is None:self._inspection_new_view(obs)
        self.p_des=self.inspection_pose[0].copy();self.q_des=self.inspection_pose[1].copy()
        self.gripper=float(self.g['open_ctrl']);self.guard_action='settled_inspection_move'
        arrived=(np.linalg.norm(obs.ee_pos-self.p_des)<=settings['arrival_position_m'] and
            np.degrees(np.linalg.norm(orientation_error(self.q_des,obs.ee_mat)))<=settings['arrival_rotation_deg'])
        if arrived:
            if self.inspection_arrived is None:
                self.inspection_arrived=obs.t;self.inspection_samples.clear()
                self.inspection_history.append({'t':obs.t,'event':'arrived','view':self.inspection_view})
        else:
            self.inspection_arrived=None;self.inspection_samples.clear()
        settled=arrived and obs.t-self.inspection_arrived>=settings['settle_s']-1e-9
        capture=obs.est.capsule.observation.get('capture_t')
        if capture is None:capture=obs.est.capsule.measurement_t
        new_capture=(capture is not None and (self.inspection_last_capture is None or capture>self.inspection_last_capture+1e-9))
        if new_capture:self.inspection_last_capture=capture
        if settled and new_capture and capture>=self.inspection_arrived+settings['settle_s']-1e-9:
            status,detail=self.inspect_placement(obs)
            # Pair must come from this actual capture, not a held pose of either target.
            same=(detail is not None and abs(detail['pair_t']-capture)<1e-7
                  and abs(obs.est.capsule.measurement_t-capture)<1e-7
                  and abs(obs.est.slot.measurement_t-capture)<1e-7)
            if not same:status='unknown';detail=None
            self.verification_status=status;self.guard_action='settled_inspection_'+status
            self.verification_history.append({'t':obs.t,'status':status,**({} if detail is None else detail)})
            if status not in ('inside','outside'):self.inspection_samples.clear()
            else:
                if self.inspection_samples and self.inspection_samples[-1][1]!=status:self.inspection_samples.clear()
                self.inspection_samples.append((capture,status,detail))
            if status in ('outside','ambiguous'):self.inspection_confirmed_t=None
            while self.inspection_samples and self.inspection_samples[0][0]<obs.t-recovery['evidence_history_s']:
                self.inspection_samples.popleft()
            if self._inspection_stable(status):
                if status=='outside':return self._inspection_retry(obs)
                self.inspection_confirmed_t=capture
                self.inspection_history.append({'t':obs.t,'event':'confirmed_inside','view':self.inspection_view})
        fresh_confirmation=(settled and self.inspection_confirmed_t is not None
            and obs.t-self.inspection_confirmed_t<=settings['confirmation_max_age_s']+1e-9)
        if fresh_confirmation:
            # No view change after confirmation. Contradictory stable evidence above
            # still triggers a physical retry; unknown observations expire explicitly.
            if obs.t-self.verification_started>=recovery['monitor_s'] and self.state!=S.DONE:
                self._go(S.DONE,obs.t,'settled visual inspection confirmed placement')
            self.guard_action='settled_inspection_confirmed';return self._cmd()
        if self.state==S.DONE:self._go(S.VERIFY,obs.t,'settled inspection confirmation expired')
        elapsed=obs.t-self.verification_started
        if elapsed>=settings['timeout_s']:
            self.verification_status='inconclusive';self._go(S.ABORT,obs.t,'settled visual inspection exhausted')
            return self._cmd()
        age=obs.t-(self.inspection_arrived if self.inspection_arrived is not None else self.inspection_view_started)
        deadline=settings['view_dwell_s'] if self.inspection_arrived is not None else settings['unreachable_view_s']
        if age>=deadline:
            self.inspection_view=(self.inspection_view+1)%settings['views'];self._inspection_new_view(obs)
        return self._cmd()
