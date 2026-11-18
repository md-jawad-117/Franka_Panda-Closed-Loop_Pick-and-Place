import itertools,json
from pathlib import Path
import numpy as np
import mujoco
from cl_pickplace.camera_framing_wide import design,widen_wrist
from cl_pickplace.config import load_config
from cl_pickplace.sim import Sim
from cl_pickplace.dataset_geometry import camera_transform,intrinsics,project

ROOT=Path(__file__).resolve().parents[1]


def test_enclosing_grasp_box_fits_actual_camera_with_declared_border():
    cfg=load_config(ROOT/'config/learned_reacquisition_v4.yaml')
    settings=json.loads((ROOT/'config/framing_wide_protocol_v13.json').read_text())['camera_settings']
    planned=design(cfg,settings);wide=widen_wrist(cfg,settings)
    assert 60<planned['fovy_deg']<=100 and wide['sim']['wrist_cam']['tilt_deg']==cfg['sim']['wrist_cam']['tilt_deg']
    assert cfg['sim']['wrist_cam']['fovy']==60
    sim=Sim(wide);sim.reset(np.random.default_rng(12));mujoco.mj_forward(sim.model,sim.data)
    hand=sim.link_bodies['hand'];world=np.asarray(list(itertools.product(*planned['hand_frame_box_m'])))@sim.data.xmat[hand].reshape(3,3).T+sim.data.xpos[hand]
    cid=sim.cams['wrist_cam'];T=camera_transform(sim.data.cam_xmat[cid].reshape(3,3),sim.data.cam_xpos[cid])
    uv,z=project(world,intrinsics(320,240,planned['fovy_deg']),T)
    assert np.all(z>0) and np.all(uv>=6) and np.all(uv<=np.array([313,233]))


def test_fov_design_changes_no_physics_or_perception_threshold():
    cfg=load_config(ROOT/'config/learned_reacquisition_v4.yaml')
    settings=json.loads((ROOT/'config/framing_wide_protocol_v13.json').read_text())['camera_settings']
    result=widen_wrist(cfg,settings);result['sim']['wrist_cam']['fovy']=cfg['sim']['wrist_cam']['fovy']
    assert result==cfg
