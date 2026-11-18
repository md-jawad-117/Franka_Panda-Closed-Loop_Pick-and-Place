"""One bounded physical push to free a failed rim grasp, using visual poses only."""
import numpy as np
from .kinematics import gripper_down_quat


class RimExtraction:
    @property
    def extraction_enabled(self):
        return self.acquisition_enabled and self.acquisition_settings.get('rim_extraction',{}).get('enabled',False)

    def reset_extraction(self):
        self.extraction_needed=False;self.extraction_attempts=0
        self.extraction_plan=None;self.extraction_phase=None;self.extraction_started=None
        self.extraction_history=[]

    def begin_extraction(self,obs):
        from .task_fsm import S
        settings=self.acquisition_settings['rim_extraction'];cap,slot=obs.est.capsule,obs.est.slot
        if not self.extraction_enabled:return False
        if self.extraction_attempts>=settings['max_attempts']:
            self.extraction_needed=False
            return False
        vector=cap.pos[:2]-slot.pos[:2];length=np.linalg.norm(vector)
        if length<settings['minimum_direction_m'] or length>settings['maximum_distance_m']:
            self.extraction_needed=False
            return False
        vector=vector/length;limits=np.asarray(self.acquisition_settings['robot_search_bounds_xy_m'])
        plan=None
        for angle in (0.,np.pi/4,-np.pi/4):
            c,s=np.cos(angle),np.sin(angle);direction=np.array([[c,-s],[s,c]])@vector
            start=cap.pos[:2]-direction*settings['approach_distance_m']
            finish=cap.pos[:2]+direction*settings['push_distance_m']
            if not all(np.all(p>=limits[:,0]) and np.all(p<=limits[:,1]) for p in (start,finish)):continue
            height=max(slot.pos[2]+self.acquisition_settings['nominal_wall_height_m']+
                       self.acquisition_settings['finger_tip_below_tcp_m']+self.acquisition_settings['rim_clearance_m'],
                       cap.pos[2]+self.recovery_settings['capsule_radius_m']+
                       self.acquisition_settings['finger_tip_below_tcp_m']-settings['surface_overlap_m'])
            plan={'start':np.r_[start,height],'finish':np.r_[finish,height],
                  'clear_z':max(obs.ee_pos[2],self.table_z+self.p['carry_height']),
                  'fixture_yaw':slot.yaw,'fixture_floor_z':float(slot.pos[2]),
                  'yaw':float(np.arctan2(direction[1],direction[0])),'capsule_position_m':cap.pos.tolist(),
                  'fixture_position_m':slot.pos.tolist(),'observation_t':min(cap.measurement_t,slot.measurement_t)}
            break
        self.extraction_needed=False
        if plan is None:return False
        self.extraction_plan=plan;self.extraction_phase='raise';self.extraction_started=obs.t
        self.extraction_origin=obs.ee_pos.copy();self.extraction_attempts+=1
        self.extraction_history.append({'t':obs.t,'event':'planned','attempt':self.extraction_attempts,
                                        **{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in plan.items()}})
        self._go(S.EXTRACT,obs.t,'robot push: free failed rim grasp')
        return True

    def extraction_tick(self,obs):
        from .task_fsm import S
        settings=self.acquisition_settings['rim_extraction'];plan=self.extraction_plan
        self.gripper=float(self.g['closed_ctrl']);self.q_des=gripper_down_quat(self._yaw(plan['yaw']))
        phase=self.extraction_phase
        if phase=='raise':self.p_des=np.r_[self.extraction_origin[:2],plan['clear_z']]
        elif phase=='approach':self.p_des=np.r_[plan['start'][:2],plan['clear_z']]
        elif phase=='lower':self.p_des=plan['start'].copy()
        elif phase=='push':
            distance=float(np.linalg.norm(plan['finish'][:2]-plan['start'][:2]))
            progress=min(1.,(obs.t-self.extraction_push_started)*settings['push_speed_m_s']/distance)
            self.p_des=plan['start']+progress*(plan['finish']-plan['start'])
            c,s=np.cos(plan['fixture_yaw']),np.sin(plan['fixture_yaw'])
            local=np.array([[c,s],[-s,c]])@(obs.ee_pos[:2]-np.asarray(plan['fixture_position_m'])[:2])
            outer=np.asarray(self.recovery_settings['cavity_m'])/2+settings['nominal_wall_thickness_m']
            # Follow the lower table surface only once the known robot finger
            # footprint is beyond the nominal fixture walls.
            if np.any(np.abs(local)>outer+settings['finger_footprint_margin_m']):
                self.p_des[2]=plan['fixture_floor_z']-settings['nominal_bottom_thickness_m']+2*self.recovery_settings['capsule_radius_m']+self.acquisition_settings['finger_tip_below_tcp_m']-settings['surface_overlap_m']
        else:self.p_des=np.r_[plan['finish'][:2],plan['clear_z']]
        self.guard_action='physical_rim_push_'+phase
        e,_,r=self._errors(obs)
        if obs.t-self.extraction_started>=settings['timeout_s']:
            self.gripper=float(self.g['open_ctrl'])
            self.extraction_history.append({'t':obs.t,'event':'inconclusive'})
            self._go(S.ABORT,obs.t,'bounded robot push timeout')
        elif (e<settings['position_tolerance_m'] and r<settings['rotation_tolerance_deg']
              and (phase!='push' or progress>=1.)):
            phases=['raise','approach','lower','push','retreat'];index=phases.index(phase)
            if index<len(phases)-1:
                self.extraction_phase=phases[index+1]
                if self.extraction_phase=='push':self.extraction_push_started=obs.t
                self.extraction_history.append({'t':obs.t,'event':'phase','phase':self.extraction_phase})
            else:
                self.extraction_history.append({'t':obs.t,'event':'trajectory_complete'})
                self.gripper=float(self.g['open_ctrl']);self.acquisition_needed=True
                self._go(S.APPROACH_OBJECT,obs.t,'robot push complete: re-observe capsule')
        return self._cmd()
