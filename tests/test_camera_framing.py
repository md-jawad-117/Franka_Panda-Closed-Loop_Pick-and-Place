import copy
import numpy as np
import mujoco
import pytest
from cl_pickplace.camera_framing import aim_wrist_at_grasp
from cl_pickplace.config import load_config
from cl_pickplace.sim import Sim
from cl_pickplace.dataset_geometry import camera_transform,project,intrinsics


def test_aiming_uses_static_hand_calibration_and_changes_only_camera_tilt():
    cfg=load_config('config/learned_reacquisition_v4.yaml');before=copy.deepcopy(cfg)
    revised=aim_wrist_at_grasp(cfg,.006)
    assert cfg==before and abs(revised['sim']['wrist_cam']['tilt_deg']-54.24611274556325)<1e-10
    revised['sim']['wrist_cam']['tilt_deg']=cfg['sim']['wrist_cam']['tilt_deg']
    assert revised==cfg


def test_actual_camera_optical_axis_centres_declared_grasp_region_without_target_reads():
    cfg=aim_wrist_at_grasp(load_config('config/learned_reacquisition_v4.yaml'),.006)
    sim=Sim(cfg);sim.reset(np.random.default_rng(12));mujoco.mj_forward(sim.model,sim.data)
    cid=sim.cams['wrist_cam'];T=camera_transform(sim.data.cam_xmat[cid].reshape(3,3),sim.data.cam_xpos[cid])
    point=sim.ee_pos()+sim.ee_mat()@np.array([0.,0.,.006])
    uv,depth=project(point[None],intrinsics(320,240,60),T)
    np.testing.assert_allclose(uv[0],[159.5,119.5],atol=1e-8);assert depth[0]>0


def test_invalid_mount_cannot_silently_generate_a_camera_profile():
    cfg=load_config('config/learned_reacquisition_v4.yaml');cfg['sim']['wrist_cam']['pos'][1]=.02
    with pytest.raises(ValueError):aim_wrist_at_grasp(cfg,.006)
