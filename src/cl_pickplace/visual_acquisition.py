"""Calibrated observation waypoints and bounded reacquisition; no simulator API."""
from collections import deque
import numpy as np
from .kinematics import gripper_down_quat,quat_to_mat,mat_to_quat,wrap_half_pi


class VisualAcquisition:
    @property
    def acquisition_settings(self):return self.p.get('visual_acquisition',{})

    @property
    def acquisition_enabled(self):
        return self.closed_loop and self.require_visual_observations and bool(self.acquisition_settings)

    def reset_acquisition(self):
        self.acquisition_started=None;self.acquisition_target='capsule'
        self.acquisition_samples=deque();self.acquisition_history=[]
        self.acquisition_needed=False;self.acquisition_view_started=None
        self.acquisition_view=0;self.acquisition_arrived=None
        self.recovery_grasp_attempts=0
        self.hand_camera_offset=None;self.hand_camera_rotation=None
        self.proposed_grasp_offset=None;self.retry_grasp_offset=None
        self.grasp_attempt_counted=False

    def observation_waypoint(self,obs,target,view):
        settings=self.acquisition_settings
        estimate=obs.est.capsule if target=='capsule' else obs.est.slot
        if estimate.measurement_t is None:
            position=np.array([*settings['initial_search_xy_m'][target],self.table_z])
            yaw=0.
        else:position=estimate.pos.copy();yaw=estimate.yaw
        offsets=settings['view_yaw_offsets_deg'];yaw=self._yaw(yaw+np.radians(offsets[view%len(offsets)]))
        q=gripper_down_quat(yaw);rotation=quat_to_mat(q)
        height=max(position[2]+settings['height_above_target_m'],self.table_z+settings['minimum_height_above_table_m'])
        desired=np.array([position[0],position[1],height])
        calibration=estimate.observation.get('wrist_world_from_camera')
        if calibration is not None and abs(estimate.observation.get('capture_t',-1)-obs.t)<1e-7:
            camera=np.asarray(calibration)
            # Robot FK and the observed camera transform define the fixed hand /
            # camera transform. Center the last observed target in the new view.
            self.hand_camera_offset=obs.ee_mat.T@(camera[:3,3]-obs.ee_pos)
            self.hand_camera_rotation=obs.ee_mat.T@camera[:3,:3]
        if self.hand_camera_offset is not None:
            # Vary actual optical elevation, not just orbit at the same fixed
            # oblique angle. A vertical view can reveal a capsule behind a rim.
            nominal_camera=rotation@self.hand_camera_rotation
            angle=np.radians(settings['view_tilt_deg'][view%len(settings['view_tilt_deg'])])
            forward=np.array([-np.sin(angle)*np.cos(yaw),-np.sin(angle)*np.sin(yaw),-np.cos(angle)])
            current=nominal_camera[:,2];v=np.cross(current,forward);cosine=float(current@forward)
            cross=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
            alignment=np.eye(3)+cross+cross@cross/(1+cosine)
            rotation=alignment@rotation;q=mat_to_quat(rotation)
            offset=rotation@self.hand_camera_offset;ray=(rotation@self.hand_camera_rotation)[:,2]
            if ray[2]<-.2:
                distance=(position[2]-height-offset[2])/ray[2]
                desired[:2]=position[:2]-offset[:2]-distance*ray[:2]
        desired[:2]+=np.asarray(settings['view_offsets_xy_m'][view%len(settings['view_offsets_xy_m'])])
        limits=np.asarray(settings['robot_search_bounds_xy_m'])
        desired[:2]=np.clip(desired[:2],limits[:,0],limits[:,1])
        return desired,q

    def begin_acquisition(self,obs,target='capsule'):
        from .task_fsm import S
        self.acquisition_started=obs.t;self.acquisition_target=target
        self.acquisition_samples.clear();self.acquisition_needed=False
        self.acquisition_view=0;self.acquisition_view_started=None;self.acquisition_arrived=None
        self.proposed_grasp_offset=None;self.retry_grasp_offset=None
        self._go(S.ACQUIRE,obs.t,'active visual acquisition: '+target)

    def acquisition_tick(self,obs):
        from .task_fsm import S
        settings=self.acquisition_settings
        self.gripper=float(self.g['open_ctrl'])
        elapsed=obs.t-self.acquisition_started
        estimate=obs.est.capsule if self.acquisition_target=='capsule' else obs.est.slot
        if self.acquisition_view_started is None:self.acquisition_view_started=obs.t
        if self.acquisition_arrived is not None and obs.t-self.acquisition_arrived>=settings['view_dwell_s']:
            self.acquisition_view=(self.acquisition_view+1)%len(settings['view_yaw_offsets_deg'])
            self.acquisition_view_started=obs.t;self.acquisition_arrived=None;self.acquisition_samples.clear()
        self.p_des,self.q_des=self.observation_waypoint(obs,self.acquisition_target,self.acquisition_view)
        e,_,r=self._errors(obs)
        arrived=e<settings['arrival_position_m'] and r<settings['arrival_rotation_deg']
        if arrived and self.acquisition_arrived is None:self.acquisition_arrived=obs.t
        # A view that cannot be reached also has a deadline.
        if not arrived and obs.t-self.acquisition_view_started>settings['unreachable_view_s']:
            self.acquisition_view=(self.acquisition_view+1)%len(settings['view_yaw_offsets_deg'])
            self.acquisition_view_started=obs.t;self.acquisition_arrived=None;self.acquisition_samples.clear()
        fresh=self._visual_ok(estimate,obs.t)
        if not fresh:self.acquisition_samples.clear()
        if (fresh and arrived and estimate.measurement_t>=self.acquisition_started-1e-9
                and (not self.acquisition_samples or estimate.measurement_t>self.acquisition_samples[-1][0]+1e-9)):
            self.acquisition_samples.append((estimate.measurement_t,estimate.pos.copy(),estimate.yaw))
        while self.acquisition_samples and self.acquisition_samples[0][0]<obs.t-settings['history_s']:
            self.acquisition_samples.popleft()
        samples=list(self.acquisition_samples)
        stable=(len(samples)>=settings['minimum_samples']
                and samples[-1][0]-samples[0][0]>=settings['stable_window_s']-1e-9
                and all(np.linalg.norm(a[1]-samples[-1][1])<=settings['stable_position_m']
                        and abs(wrap_half_pi(a[2]-samples[-1][2]))<=np.radians(settings['stable_yaw_deg']) for a in samples))
        self.guard_action='active_acquire_'+self.acquisition_target
        self.stats.visual_guard_ticks+=1
        initial_ok=(self.frozen is not None or (self._visual_ok(obs.est.capsule,obs.t) and self._visual_ok(obs.est.slot,obs.t)))
        if stable and fresh and arrived and initial_ok:
            self.acquisition_history.append({'t':obs.t,'outcome':'confirmed','target':self.acquisition_target,'view':self.acquisition_view})
            self._go(S.APPROACH_OBJECT,obs.t,'active acquisition: confirmed fresh target')
        elif elapsed>=settings['timeout_s']:
            self.acquisition_history.append({'t':obs.t,'outcome':'inconclusive','target':self.acquisition_target,'view':self.acquisition_view})
            self._go(S.ABORT,obs.t,'active visual acquisition exhausted')
        return self._cmd()

    def acquisition_dispatch(self,obs):
        from .task_fsm import S
        if (self.state==S.GRASP and not self.grasp_attempt_counted
                and self.recovery_grasp_attempts>=self.acquisition_settings['max_recovery_grasp_attempts']):
            self.gripper=float(self.g['open_ctrl']);self.p_des=obs.ee_pos.copy()
            self.p_des[2]=max(self.p_des[2],self.table_z+self.p['carry_height'])
            self._go(S.ABORT,obs.t,'bounded recovery grasp attempts exhausted')
            return self._cmd()
        if self.state==S.EXTRACT:return self.extraction_tick(obs)
        if (self.extraction_needed and self.state==S.APPROACH_OBJECT
                and self._visual_ok(obs.est.capsule,obs.t) and self._visual_ok(obs.est.slot,obs.t)):
            if self.begin_extraction(obs):return self.extraction_tick(obs)
        if self.acquisition_needed and self.state==S.APPROACH_OBJECT:self.begin_acquisition(obs)
        if self.state==S.ACQUIRE:return self.acquisition_tick(obs)
        return None

    def grasp_position(self,obs,est):
        from .task_fsm import S
        target=est.capsule.pos.copy();target[2]+=self.p['grasp_height_offset']
        if not (self.acquisition_enabled and self.placement_retries>0 and self.acquisition_settings.get('rim_aware_grasp')):
            return target
        settings=self.acquisition_settings;slot=est.slot
        # Commit the chosen grasp clearance for the short final approach. Losing
        # a box detection cannot suddenly lower the fingers into its rim.
        if self.state==S.DESCEND_OBJECT and self.retry_grasp_offset is not None:
            target[2]=est.capsule.pos[2]+self.retry_grasp_offset
            return target
        if not self._visual_ok(slot,obs.t):
            if self.proposed_grasp_offset is not None:target[2]=est.capsule.pos[2]+self.proposed_grasp_offset
            return target
        c,s=np.cos(slot.yaw),np.sin(slot.yaw)
        local=np.array([[c,s],[-s,c]])@(est.capsule.pos[:2]-slot.pos[:2])
        if np.all(np.abs(local)<np.asarray(settings['rim_near_region_m'])):
            rim=slot.pos[2]+settings['nominal_wall_height_m']
            target[2]=min(max(target[2],rim+settings['finger_tip_below_tcp_m']+settings['rim_clearance_m']),
                          est.capsule.pos[2]+settings['maximum_grasp_offset_m'])
        self.proposed_grasp_offset=float(target[2]-est.capsule.pos[2])
        return target
