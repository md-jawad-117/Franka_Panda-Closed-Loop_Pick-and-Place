"""Geometry, observation boundary, CAD and isolated-IK regression checks."""
import copy,json,sys
from pathlib import Path
import mujoco
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from cl_pickplace.grasp_planner import (box_gaps,fixture_boxes,gripper_boxes,rotation,
    interpolate_rotation,candidates,plan,screen_action,validate_observation)
from cl_pickplace.config import load_config
from cl_pickplace.sim import Sim
from evaluate_grasp_planner_v7 import robot_templates,kinematic_check


def protocol():return json.loads((ROOT/'config/grasp_planner_protocol_v7.json').read_text())


def observation():
    return {'t':1.,'capsule_t':.9,'fixture_t':.9,'capsule_visible':True,'fixture_visible':True,
            'motion_pending':False,'capsule_position_m':[.45,-.1,.408],
            'capsule_axis_world':[1.,0,0],'fixture_position_m':[.55,.2,.403],
            'fixture_yaw_rad':0.,'ee_position_m':[.45,-.1,.52],
            'ee_rotation':np.diag([1.,-1.,-1.]).tolist(),'finger_width_m':.08}


def test_obb_sat_separates_diagonal_rotated_boxes_and_detects_overlap():
    a=np.eye(3)[None,:,:];b=rotation([0,0,1],np.pi/4)[None,:,:]
    half=np.array([[.01,.02,.03]])
    separated=box_gaps([[0,0,0]],a,half,[[.04,0,0]],b,half)[0,0]
    overlap=box_gaps([[0,0,0]],a,half,[[.01,0,0]],b,half)[0,0]
    assert separated>0 and overlap<0
    # SAT is invariant to a common world rotation and translation.
    r=rotation([1,2,3],.73);shift=np.array([.3,-.2,.6])
    value=box_gaps([shift],np.einsum('ij,njk->nik',r,a),half,
                   [r@np.array([.04,0,0])+shift],np.einsum('ij,njk->nik',r,b),half)[0,0]
    assert value==pytest.approx(separated,abs=1e-10)


def test_fixture_floor_convention_preserves_bottom_and_four_walls():
    cad=load_config()['scene']['slot'];c,a,h=fixture_boxes(np.array([.5,.1,.403]),0,cad)
    assert len(c)==5 and c[0,2]+h[0,2]==pytest.approx(.403)
    assert np.allclose(c[1:,2]+h[1:,2],.413)


def test_rotation_interpolation_handles_half_turn_and_so3():
    a=np.eye(3);b=rotation([1,2,3],np.pi)
    np.testing.assert_allclose(interpolate_rotation(a,b,1),b,atol=1e-7)
    mid=interpolate_rotation(a,b,.5)
    np.testing.assert_allclose(mid.T@mid,np.eye(3),atol=1e-9)
    assert np.linalg.det(mid)==pytest.approx(1)


def test_nearly_identical_orientations_stay_finite_under_strict_numpy_checks():
    r=rotation([0,0,1],.5911343038391399)@np.diag([1.,-1.,-1.])
    with np.errstate(invalid='raise',divide='raise'):
        for theta in (0.,1e-12,1e-9,1e-7,.1,np.pi):
            end=r@rotation([1,2,3],theta)
            for f in (0,.25,.5,1):
                value=interpolate_rotation(r,end,f)
                assert np.isfinite(value).all()
                np.testing.assert_allclose(value.T@value,np.eye(3),atol=1e-8)


@pytest.mark.parametrize('mutation,reason',[
    ({'capsule_t':.5},'stale_observation'),({'fixture_t':None},'stale_observation'),
    ({'motion_pending':True},'unconfirmed_or_missing_observation'),
    ({'capsule_visible':False},'unconfirmed_or_missing_observation'),
    ({'capsule_axis_world':[0,0,1]},'unobservable_horizontal_axis'),
    ({'fixture_position_m':[np.nan,0,0]},'nonfinite_observation'),
    ({'evaluation_only':{}},'invalid_sensor_interface')])
def test_untrusted_or_incomplete_observations_abstain(mutation,reason):
    obs=observation();obs.update(mutation)
    result=plan(obs,{}, {},protocol(),protocol()['modes']['bounded_geometry'])
    assert result['status']=='abstain' and result['reason']==reason


def test_registered_candidates_include_tilts_symmetry_and_do_not_mutate_input():
    obs=observation();before=copy.deepcopy(obs);actions=candidates(obs,protocol())
    assert len(actions)==1350 and obs==before
    assert len({a['name'] for a in actions})==1350
    assert {a['tilt'] for a in actions}=={'vertical','roll_negative','roll_positive','pitch_negative','pitch_positive'}
    for action in actions:
        r=np.asarray(action['rotation']);np.testing.assert_allclose(r.T@r,np.eye(3),atol=1e-10)
        assert np.linalg.det(r)==pytest.approx(1)


def test_cad_boxes_enclose_actual_collision_vertices_at_all_gripper_widths():
    sim=Sim(load_config());templates=robot_templates(sim);m=sim.model;d=mujoco.MjData(m)
    d.qpos[sim.arm_qadr]=sim.home_q
    for width in (0,.024,.04,.08):
        d.qpos[sim.finger_qadr]=width/2;mujoco.mj_kinematics(m,d)
        r=d.site_xmat[sim.ee_site].reshape(3,3);p=d.site_xpos[sim.ee_site]
        centers,axes,half=gripper_boxes(templates,p,r,width)
        for k,g in enumerate(templates['geometry_ids']):
            if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
                mid=int(m.geom_dataid[g]);s=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);v=m.mesh_vert[s:s+n]
            else:
                import itertools
                v=np.array(list(itertools.product((-1,1),repeat=3)))*m.geom_size[g]
            world=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            assert np.all(np.abs((world-centers[k])@axes[k])<=half[k]+1e-7)


def test_cad_and_ik_are_invariant_to_hidden_live_target_poses():
    sim=Sim(load_config());sim.reset(poses={'capsule':(.44,-.1,.3),'slot':(.52,.2,.1)})
    t=robot_templates(sim);obs=observation();action=candidates(obs,protocol())[0];arm=sim.arm_q()
    initial=sim.data.qpos.copy();first=kinematic_check(sim,arm,obs,action,protocol())
    np.testing.assert_array_equal(initial,sim.data.qpos)
    sim.set_capsule_pose(.6,.3,2);sim.set_slot_pose(.35,.02,-2);mujoco.mj_forward(sim.model,sim.data)
    assert robot_templates(sim)==t
    assert kinematic_check(sim,arm,obs,action,protocol())==first


def test_clearance_checks_the_closing_sweep_even_when_open_fingers_clear():
    # A simple pair of closing finger boxes straddles a fixture wall when open.
    obs=observation();obs['capsule_position_m']=[.5,.1,.42];obs['fixture_position_m']=[.5,.1,.403]
    templates={'centers':[[0,0,0],[0,0,0]],'axes':[np.eye(3).tolist()]*2,
               'half_sizes':[[.002,.002,.002]]*2,'width_directions':[[0,1,0],[0,-1,0]]}
    action={'position_m':[.5,.1+.0175,.408],'rotation':np.eye(3).tolist(),'approach_joint_width_m':.08}
    result=screen_action(obs,action,templates,load_config()['scene']['slot'],protocol(),protocol()['modes']['nominal_geometry'])
    assert not result['feasible'] and result['reason']=='grasp_or_closure_clearance'
