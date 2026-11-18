"""v24: ignore capsule detections whose line of sight grazes the hand during a stroke.

v23 (docs/CONFIRMED_CENTERING_V23.md) still stopped seed 600 on two agreeing
top-camera detections biased 19.6 mm away from the fingers that partly hid the
capsule. Here, during lowering and pushing only, a capsule detection is treated
as no evidence (not fresh) when any point of the closed hand/finger CAD envelope
(config/hand_envelope_v24.npz, TCP frame) lies within ``occlusion_clearance_m``
of a camera->capsule sight line. Inputs: camera calibration, measured TCP pose and
the detection itself. Missing evidence is then handled by the unchanged v22 blind
budget; v23's two-detection confirmation, fixture guards and settled-inspection
completion are unchanged. v22/v23 sources are not modified.
"""
from pathlib import Path
import hashlib
import numpy as np
from .confirmed_blind_centering import ConfirmedBlindCenterFSM
from .hand_occlusion import capsule_samples,line_of_sight_clearance,hand_points_world

ROOT=Path(__file__).resolve().parents[2]


class OcclusionGatedCenterFSM(ConfirmedBlindCenterFSM):
    def reset(self,*args,**kwargs):
        super().reset(*args,**kwargs);self.occlusion_gated=[]
        if not hasattr(self,'_hand_envelope'):
            path=ROOT/self.centering_settings['hand_envelope']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=self.centering_settings['hand_envelope_sha256']:
                raise RuntimeError('hand envelope hash mismatch: '+str(path))
            self._hand_envelope=np.load(path)['points_tcp_frame_m'].astype(float)

    def _occluded(self,cap,obs):
        settings=self.centering_settings;source=cap.source or ''
        if source.endswith('top_cam'):camera=np.asarray(settings['top_cam_position_m'],float)
        elif source.endswith('wrist_cam'):camera=obs.ee_pos+obs.ee_mat@np.asarray(settings['wrist_cam_tcp_frame_m'],float)
        else:return True,None  # unknown view: no line-of-sight evidence, treat as gated
        axis=np.asarray(cap.axis,float) if cap.axis is not None else np.array([np.cos(cap.yaw),np.sin(cap.yaw),0.])
        points=hand_points_world(self._hand_envelope,obs.ee_pos,obs.ee_mat)
        clearance=line_of_sight_clearance(camera,capsule_samples(cap.pos,axis,settings['plan_settings']['capsule_half_segment_m']),points)
        return clearance<settings['occlusion_clearance_m'],clearance

    def _fresh(self,est,obs,max_age):
        fresh=super()._fresh(est,obs,max_age)
        if fresh and est is obs.est.capsule and self.centering_phase in ('lower','push'):
            gated,clearance=self._occluded(est,obs)
            if gated:
                if not self.occlusion_gated or self.occlusion_gated[-1]['measurement_t']!=est.measurement_t:
                    self.occlusion_gated.append({'t':obs.t,'measurement_t':est.measurement_t,'source':est.source,'clearance_m':clearance})
                return False
        return fresh
