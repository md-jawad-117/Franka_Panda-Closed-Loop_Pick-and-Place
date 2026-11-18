"""Independent CAD support, collision distance and static interception checks."""
import copy,json,sys
from pathlib import Path
import mujoco
import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from cl_pickplace.config import load_config
from cl_pickplace.sim import Sim
from cl_pickplace.grasp_collision import extract_robot_cad,GraspCollisionWorld
from cl_pickplace.grasp_capture_planner import registered_candidates,capsule_scenarios,plan
from cl_pickplace.grasp_planner import rotation
from cl_pickplace.kinematics import quat_to_mat
from evaluate_grasp_capture_v8 import calibration
from test_grasp_planner import observation,protocol as parent_protocol


def protocol():return json.loads((ROOT/'config/grasp_capture_protocol_v8.json').read_text())


def obs():
    base=observation();base.update(recorded_target_m=[.45,-.1,.419],recorded_rotation=base['ee_rotation'])
    return base


def small_cad():
    hand={'type':'box','body':'hand','position_m':[0,0,.2],'quaternion':[1,0,0,0],'half_size':[.001,.001,.001]}
    return [hand]+[{'type':'box','body':side,'position_m':[0,sign*.0015,0],
             'quaternion':[1,0,0,0],'half_size':[.01,.002,.01]}
            for side,sign in (('left',1),('right',-1))]


def world(cad):
    return GraspCollisionWorld(cad,load_config()['scene']['slot'],.4,parent_protocol()['capsule_prior'],
                               {'position_m':[1.5,.8,.403],'yaw_rad':.3})


def vertices(cad,width):
    result=[]
    import itertools
    for row in cad:
        local=np.array(row['vertices']) if row['type']=='mesh' else np.array(list(itertools.product((-1,1),repeat=3)))*row['half_size']
        p=np.array(row['position_m']);p[1]+={'hand':0,'left':1,'right':-1}[row['body']]*width/2
        result.append(local@quat_to_mat(np.array(row['quaternion'])).T+p)
    return result


def test_compiled_convex_meshes_preserve_original_cad_support_at_three_widths():
    sim=Sim(load_config());cad=extract_robot_cad(sim.model,calibration(sim));query=world(cad)
    r=rotation([1,2,3],.37);p=np.array([.45,-.1,.55]);directions=np.random.default_rng(723).normal(size=(20,3))
    directions/=np.linalg.norm(directions,axis=1)[:,None]
    for width in (0,.024,.08):
        expected=vertices(cad,width);query.update(p,r,width)
        for row,g,points in zip(cad,query.robot_ids,expected):
            m=query.model;d=query.data
            if row['type']=='mesh':
                mid=int(m.geom_dataid[g]);s=int(m.mesh_vertadr[mid]);n=int(m.mesh_vertnum[mid]);v=m.mesh_vert[s:s+n]
            else:
                import itertools
                v=np.array(list(itertools.product((-1,1),repeat=3)))*m.geom_size[g]
            actual=v@d.geom_xmat[g].reshape(3,3).T+d.geom_xpos[g]
            expected_world=points@r.T+p
            np.testing.assert_allclose((actual@directions.T).max(0),(expected_world@directions.T).max(0),atol=2e-7,rtol=0)


def test_plane_collision_distance_matches_independent_vertex_support():
    sim=Sim(load_config());cad=extract_robot_cad(sim.model,calibration(sim));query=world(cad)
    for angle,height,width in ((0,.411,.08),(.2,.42,.024),(-.3,.43,.04)):
        r=np.diag([1.,-1.,-1.])@rotation([1,0,0],angle);p=np.array([.45,-.1,height])
        points=np.concatenate(vertices(cad,width));true_gap=min(.01,float((points@r.T+p)[:,2].min()-.4))
        predicted=query.environment_clearance(p,r,width)
        assert predicted==pytest.approx(true_gap,abs=2e-7)


def test_static_pad_interception_requires_two_opposing_contacts():
    query=world(small_cad());settings=protocol()['capture'];p=np.array([.45,-.1,.5]);r=np.eye(3)
    query.set_capsule(p,[1,0,0]);good=query.capture(p,r,.04,settings)
    assert good['geometrically_capturable'] and set(good['interception']['pad_contacts'])=={'left','right'}
    query.set_capsule(p+[0,0,.05],[1,0,0]);assert not query.capture(p,r,.04,settings)['geometrically_capturable']
    query.set_capsule(p+[0,.02,0],[1,0,0]);assert not query.capture(p,r,.04,settings)['geometrically_capturable']


def test_query_world_and_cad_do_not_read_or_modify_live_target_poses():
    sim=Sim(load_config());sim.reset(poses={'capsule':(.44,-.1,.2),'slot':(.52,.2,.1)})
    before=sim.data.qpos.copy();cad=extract_robot_cad(sim.model,calibration(sim));query=world(cad)
    first=query.environment_clearance([.45,-.1,.415],np.diag([1.,-1.,-1.]),.04)
    np.testing.assert_array_equal(before,sim.data.qpos)
    sim.set_capsule_pose(.6,.3,2);sim.set_slot_pose(.35,.02,-2);mujoco.mj_forward(sim.model,sim.data)
    assert cad==extract_robot_cad(sim.model,calibration(sim))
    assert first==query.environment_clearance([.45,-.1,.415],np.diag([1.,-1.,-1.]),.04)


def test_registered_visual_reference_and_perturbations_are_explicit_and_immutable():
    observed=obs();before=copy.deepcopy(observed);actions=registered_candidates(observed,parent_protocol())
    assert len(actions)==1351 and observed==before
    reference=next(a for a in actions if a['name']=='recorded_visual_reference')
    assert reference['position_m']==observed['recorded_target_m']
    probes=capsule_scenarios(observed,protocol(),True)
    assert len(probes)==13 and len({p[0] for p in probes})==13
    for _,p,a in probes:assert np.linalg.norm(a)==pytest.approx(1) and np.isfinite(p).all()


def test_extra_truth_field_is_rejected_before_query_world_access():
    observed=obs();observed['true_target_pose']=[0,0,0]
    decision=plan(observed,None,None,None,parent_protocol(),protocol(),protocol()['modes']['convex_point_capture'])
    assert decision['status']=='abstain' and decision['reason']=='invalid_sensor_interface'
