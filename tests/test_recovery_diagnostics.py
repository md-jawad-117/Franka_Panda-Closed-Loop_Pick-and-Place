"""Read-only evaluation logging and supervision/sensor-boundary checks."""
import json,sys
from pathlib import Path
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from cl_pickplace.config import load_config
from cl_pickplace.episode import Episode
from cl_pickplace.task_fsm import Obs
from cl_pickplace.dataset_capture import DatasetCapture
from cl_pickplace.sensors import RGBDSensor
from cl_pickplace.dataset_geometry import camera_transform,intrinsics
from recovery_diagnostics_v5 import axis_error_deg,frame_diagnostic,capture_labels_without_forward,FIELDS
from probe_grasp_feasibility_v5 import action_candidates,select_checkpoint
from probe_gripper_preshape_v6 import candidates


def test_axis_diagnostic_respects_capsule_symmetry_and_missing_axis():
    assert axis_error_deg([0,0,-2],[0,0,1])==0
    assert axis_error_deg([1,0,0],[0,0,1])==90
    assert axis_error_deg(None,[0,0,1]) is None


def test_feasibility_actions_use_only_the_explicit_visual_interface():
    protocol=json.loads((ROOT/'config/grasp_feasibility_protocol_v5.json').read_text())
    visual={'capsule_position_m':[.5,.1,.411],'capsule_axis_world':[1,0,0],
            'gripper_quaternion':[0,1,0,0],'recorded_target_m':[.5,.1,.423]}
    before=json.dumps(visual);actions=action_candidates(visual,protocol)
    assert len(actions)==13 and json.dumps(visual)==before
    assert actions[0]['position_m']==visual['recorded_target_m']
    assert all(.492-1e-9<=a['position_m'][0]<=.508+1e-9 for a in actions)


def test_checkpoint_uses_last_descent_before_actual_closure_not_failed_search():
    sequence=[('APPROACH_OBJECT','DESCEND_OBJECT',255),('DESCEND_OBJECT','DESCEND_OBJECT',255),
              ('DESCEND_OBJECT','APPROACH_OBJECT',255),('APPROACH_OBJECT','DESCEND_OBJECT',255),
              ('DESCEND_OBJECT','DESCEND_OBJECT',255),('DESCEND_OBJECT','GRASP',255),('GRASP','GRASP',0)]
    frames=[{'state_before':a,'state_after':b,'gripper_command':g,'placement_retries':1} for a,b,g in sequence]
    assert select_checkpoint(frames)==4


def test_preshape_grid_is_bounded_by_calibrated_actuator_commands():
    protocol=json.loads((ROOT/'config/gripper_preshape_assay_v6.json').read_text())
    visual={'capsule_position_m':[.5,.1,.411],'capsule_axis_world':[1,0,0],
            'gripper_quaternion':[0,1,0,0],'recorded_target_m':[.5,.1,.423]}
    actions=candidates(visual,protocol)
    assert len(actions)==46 and actions[0]['approach_ctrl']==255
    assert all(0<=action['approach_ctrl']<=255 for action in actions)
    assert min(action['approach_joint_width_m'] for action in actions)==.014


def test_frame_diagnostics_leave_physics_rng_and_policy_unchanged():
    ep=Episode(load_config(ROOT/'config/learned_reacquisition_v4.yaml'),perception_mode='ground_truth')
    try:
        ep.reset(3);ep.control_tick()
        ep.est.capsule.axis=ep.sim.data.geom_xmat[ep.sim.capsule_geom].reshape(3,3)[:,2].copy()
        obs=Obs(ep.t,ep.sim.ee_pos(),ep.sim.ee_mat(),ep.sim.arm_q(),ep.sim.finger_width(),ep.est.copy())
        spec=mujoco.mjtState.mjSTATE_INTEGRATION
        before=np.zeros(mujoco.mj_stateSize(ep.sim.model,spec));mujoco.mj_getState(ep.sim.model,ep.sim.data,before,spec)
        rng=json.dumps(ep.perception.rng.bit_generator.state);transitions=list(ep.fsm.stats.transitions)
        record=frame_diagnostic(ep,obs,'HOME')
        after=np.zeros_like(before);mujoco.mj_getState(ep.sim.model,ep.sim.data,after,spec)
        np.testing.assert_array_equal(before,after)
        assert rng==json.dumps(ep.perception.rng.bit_generator.state) and transitions==ep.fsm.stats.transitions
        assert record['evaluation_only']['capsule_axis_error_deg'] is not None
        assert set(record['evaluation_only']['contacts'])=={'finger_capsule','finger_fixture','hand_fixture','capsule_fixture','finger_table'}
    finally:ep.perception.close()


def test_separate_label_renderer_matches_sensor_and_restores_model_groups():
    ep=Episode(load_config(ROOT/'config/learned_reacquisition_v4.yaml'),perception_mode='ground_truth')
    sensor=capture=None
    try:
        ep.reset(3);sensor=RGBDSensor(ep.sim,320,240,hide_sites=True,synchronize=True);arrays={}
        for camera in ('wrist_cam','top_cam'):
            frame=sensor.capture(camera);depth=frame.depth
            arrays.update({f'rgb_{camera}':frame.rgb,f'depth_{camera}':depth,f'valid_{camera}':(depth>0).astype(np.uint8),
                           f'K_{camera}':intrinsics(320,240,frame.fovy_deg),f'T_{camera}':camera_transform(frame.world_rotation,frame.world_position)})
        capture=DatasetCapture(ep.sim,320,240,shadows=False);groups=ep.sim.model.geom_group.copy()
        qpos=ep.sim.data.qpos.copy();qvel=ep.sim.data.qvel.copy();stamp=ep.t
        labels,geometry=capture_labels_without_forward(ep.sim,capture,arrays)
        assert set(arrays)==FIELDS and not set(labels)&FIELDS
        assert labels['semantic_top_cam'].dtype==np.uint8 and len(geometry)==6
        assert set(np.unique(labels['semantic_top_cam']))<=set((0,1,2))
        np.testing.assert_array_equal(groups,ep.sim.model.geom_group)
        np.testing.assert_array_equal(qpos,ep.sim.data.qpos);np.testing.assert_array_equal(qvel,ep.sim.data.qvel)
        assert ep.t==stamp
    finally:
        if sensor is not None:sensor.close()
        if capture is not None:capture.close()
        ep.perception.close()
