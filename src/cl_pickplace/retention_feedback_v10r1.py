"""Corrected experimental retention guard, including the pre-opening tick.

Keeps the first frozen v10 implementation reproducible. A first visual mismatch
pauses immediately; the second distinct mismatch aborts. Visibility searches near
the placement surface retract before yawing. No new perception or physics input.
"""
import numpy as np
from .retention_feedback import RetentionFSM as FrozenRetentionFSM
from .task_fsm import S
from .kinematics import mat_to_quat


class RetentionFSM(FrozenRetentionFSM):
    def reset(self,start_state=S.HOME):
        super().reset(start_state)
        self.retention_warning_anchor=None;self.retention_warning_quat=None
        self.retention_hold_quat=None

    def _retry_retention(self,obs,reason):
        if reason.startswith('empty width') and obs.finger_width>self.g['max_grasp_width']:
            return self._abort_retention(obs,'excessive jaw width; grasp uncertain')
        return super()._retry_retention(obs,reason)

    def _carry_guard(self,obs):
        before=self.retention_phase
        cmd=super()._carry_guard(obs)
        if self.retention_phase=='observe_hold':
            if before!='observe_hold':
                self.retention_hold_quat=mat_to_quat(obs.ee_mat)
                # Avoid sweeping a held object through the cavity during a search.
                if self.state in (S.DESCEND_SLOT,S.RELEASE):
                    self.retention_anchor[2]=max(self.retention_anchor[2],self.table_z+self.p['carry_height'])
            if np.linalg.norm(obs.ee_pos-self.retention_anchor)>self.retention_settings['arrival_m']:
                self.p_des=self.retention_anchor.copy();self.q_des=self.retention_hold_quat.copy()
                self.guard_action='retention_observe_retract'
                return self._cmd()
        if cmd is not None:return cmd
        if self.retention_mismatch:
            if self.retention_warning_anchor is None:
                self.retention_warning_anchor=obs.ee_pos.copy();self.retention_warning_quat=mat_to_quat(obs.ee_mat)
                self._retention_event(obs,'slip_warning_hold')
            self.p_des=self.retention_warning_anchor.copy();self.q_des=self.retention_warning_quat.copy()
            self.gripper=float(self.g['closed_ctrl']);self.guard_action='retention_slip_warning_hold'
            return self._cmd()
        self.retention_warning_anchor=None;self.retention_warning_quat=None
        return None

    def update(self,obs):
        enabled=self.p.get('retention_feedback') and self.closed_loop and self.require_visual_observations
        active=self.retention_phase in ('carrying','observe_hold')
        if enabled and active and self.state==S.RELEASE and self.release_opened_t is None:
            guard=self._carry_guard(obs)
            if guard is not None:
                # Observation time is not an opening/settling dwell.
                self.t_enter=obs.t
                return guard
        reference=self.retention_reference
        cmd=super().update(obs)
        if enabled and active and self.state==S.RELEASE and self.release_opened_t is None:
            self.retention_phase='carrying';self.retention_reference=reference
        return cmd
