"""Synchronized visible/amodal labels and clean/noisy metric depth."""
import mujoco
import numpy as np

from .dataset_geometry import camera_transform,intrinsics,transform


def depth_observation(clean, rng, noise):
    valid=np.isfinite(clean)&(clean>0)&(clean<noise['max_range_m'])
    valid &= rng.random(clean.shape)>=noise['dropout']
    observed=clean+rng.normal(0,noise['sigma_m'],clean.shape)
    valid &= observed>0
    return np.where(valid,observed,0).astype(np.float32),valid.astype(np.uint8)


class DatasetCapture:
    def __init__(self,sim,width=320,height=240,shadows=True):
        self.sim=sim
        self.width,self.height=width,height
        self.renderer=mujoco.Renderer(sim.model,height,width)
        self.opt=mujoco.MjvOption()
        self.opt.sitegroup[:]=0
        self.shadows=shadows
        self.entities={1:[sim.capsule_geom],2:np.where(sim.model.geom_bodyid==sim.slot_body)[0].tolist()}

    def _scene(self,camera,opt=None):
        self.renderer.update_scene(self.sim.data,camera=self.sim.cams[camera],scene_option=opt or self.opt)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW]=self.shadows
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]=False

    def _segmentation(self,camera,opt=None):
        self.renderer.enable_segmentation_rendering()
        try:
            self._scene(camera,opt)
            return self.renderer.render().copy()
        finally:
            self.renderer.disable_segmentation_rendering()

    def amodal(self,camera,geom_ids):
        groups=self.sim.model.geom_group.copy()
        opt=mujoco.MjvOption()
        opt.geomgroup[:]=0
        opt.geomgroup[5]=1
        opt.sitegroup[:]=0
        try:
            self.sim.model.geom_group[:]=0
            self.sim.model.geom_group[geom_ids]=5
            segmentation=self._segmentation(camera,opt)
            return np.isin(segmentation[:,:,0],geom_ids)&(segmentation[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))
        finally:
            self.sim.model.geom_group[:]=groups

    def capture(self,camera,rng,noise):
        s=self.sim
        # Synchronize derived poses/contacts with current qpos at this timestamp.
        mujoco.mj_forward(s.model,s.data)
        self._scene(camera)
        rgb=self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        try:
            self._scene(camera)
            depth=self.renderer.render().copy()
        finally:
            self.renderer.disable_depth_rendering()
        seg=self._segmentation(camera)
        mask=np.zeros((self.height,self.width),np.uint8)
        amodal=np.zeros((self.height,self.width,2),np.uint8)
        geometries=[]
        for cls,ids in self.entities.items():
            mask[np.isin(seg[:,:,0],ids)&(seg[:,:,1]==int(mujoco.mjtObj.mjOBJ_GEOM))]=cls
            amodal[:,:,cls-1]=self.amodal(camera,ids)
            for gid in ids:
                geometries.append({'geom_id':int(gid),'class_id':cls,
                                   'kind':'capsule' if cls==1 else 'box',
                                   'world_from_geom':transform(s.data.geom_xmat[gid].reshape(3,3),s.data.geom_xpos[gid]).tolist(),
                                   'size':s.model.geom_size[gid].tolist()})
        observed,valid=depth_observation(depth,rng,noise)
        cid=s.cams[camera]
        T=camera_transform(s.data.cam_xmat[cid].reshape(3,3),s.data.cam_xpos[cid])
        K=intrinsics(self.width,self.height,s.model.cam_fovy[cid])
        object_pose=np.asarray(geometries[0]['world_from_geom'])
        fixture_pose=transform(s.data.xmat[s.slot_body].reshape(3,3),s.slot_pos())
        visibility={}
        for cls in (1,2):
            visible=mask==cls
            full=amodal[:,:,cls-1].astype(bool)
            n,f=int(visible.sum()),int(full.sum())
            visibility[str(cls)]={'visible_pixels':n,'amodal_pixels':f,'visible_fraction':n/f if f else None,
                                  'in_frustum':bool(f),'depth_valid_pixels':int((visible&valid.astype(bool)).sum()),
                                  'amodal_touches_border':bool(full[0].any() or full[-1].any() or full[:,0].any() or full[:,-1].any())}
        annotation={'camera':camera,'timestamp_s':s.time,'width':self.width,'height':self.height,
                    'camera_convention':'OpenCV: +x right, +y down, +z forward; column-vector transforms',
                    'depth_definition':'optical-axis z, metres; observed invalid=0',
                    'K':K.tolist(),'world_from_camera':T.tolist(),
                    'world_from_object':object_pose.tolist(),'camera_from_object':(np.linalg.inv(T)@object_pose).tolist(),
                    'world_from_fixture_floor':fixture_pose.tolist(),'camera_from_fixture_floor':(np.linalg.inv(T)@fixture_pose).tolist(),
                    'geometries':geometries,'visibility':visibility,
                    'state':{'qpos':s.data.qpos.tolist(),'qvel':s.data.qvel.tolist(),
                             'mocap_pos':s.data.mocap_pos.tolist(),'mocap_quat':s.data.mocap_quat.tolist()},
                    'pose_symmetry':'capsule axial roll unobservable; long-axis direction equivalent modulo pi'}
        return rgb,observed,valid,mask,depth,seg,amodal,annotation

    def close(self):
        self.renderer.close()
