from pathlib import Path
import copy,json,numpy as np
from cl_pickplace.fixture_exit import plan_exit
from cl_pickplace.config import load_config
from cl_pickplace.task_fsm import Obs
from cl_pickplace.perception import PoseEstimate,Estimate

ROOT=Path(__file__).resolve().parents[1]


def sample():
    cfg=load_config(ROOT/'config/learned_reacquisition_v4.yaml')
    settings=json.loads((ROOT/'config/exit_assay_protocol_v15.json').read_text())['settings']
    cap=PoseEstimate(np.array([.5,.16,.411]),0,True,0,'learned:test',1.,np.array([1.,0,0]))
    slot=PoseEstimate(np.array([.5,.15,.403]),0,True,0,'learned:test',1.)
    obs=Obs(1.,np.array([.5,.16,.55]),np.eye(3),np.zeros(7),.08,Estimate(cap,slot))
    return cfg,settings,obs


def test_exit_target_leaves_capsule_and_pickup_clearance_outside_fixture():
    cfg,settings,obs=sample();old=copy.deepcopy(obs.est)
    result=plan_exit(obs,cfg['task'],settings);plan=result['selected'];assert plan is not None
    direction=np.array([np.cos(plan['yaw']),np.sin(plan['yaw'])]);extent=.015*abs(direction@obs.est.capsule.axis[:2])+.008
    goal=np.asarray(plan['finish'])[:2]+(extent+settings['tool_forward_extent_m'])*direction
    outer=np.array([.06,.035])/2+.003
    assert np.max(np.abs(goal-obs.est.slot.pos[:2])-outer-np.array([.023,.008]))>=.012-1e-9
    np.testing.assert_equal(obs.est.capsule.axis,old.capsule.axis);np.testing.assert_equal(obs.est.capsule.pos,old.capsule.pos)


def test_exit_abstains_on_stale_or_pending_fixture_instead_of_using_truth():
    cfg,settings,obs=sample();obs.t=1.3
    assert plan_exit(obs,cfg['task'],settings)['selected'] is None
    obs.t=1.;obs.est.slot.observation['motion_pending']=True
    assert plan_exit(obs,cfg['task'],settings)['reason']=='fixture_motion_pending'


def test_exit_search_respects_robot_and_object_goal_bounds():
    cfg,settings,obs=sample();cfg['task']['visual_acquisition']['robot_search_bounds_xy_m']=[[.499,.501],[.159,.161]]
    assert plan_exit(obs,cfg['task'],settings)['selected'] is None
