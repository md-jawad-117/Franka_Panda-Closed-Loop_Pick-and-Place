"""Prospective checkpoint selection and observation/outcome boundary checks."""
import copy,json,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from fresh_grasp_validation_v9 import eligibility,observation,usability,outcome
from test_grasp_planner import protocol,observation as base_observation


def sample():
    base=base_observation()
    frame={'state_before':'DESCEND_OBJECT','state_after':'DESCEND_OBJECT','placement_retries':1,
           't':base['t'],'capsule_visible':True,'capsule_estimate_m':base['capsule_position_m'],
           'capsule_estimated_axis':base['capsule_axis_world'],'ee_position_m':base['ee_position_m'],
           'ee_rotation':base['ee_rotation'],'finger_width_m':base['finger_width_m'],
           'command_position_m':[.45,-.1,.410],'command_quaternion':[0,1,0,0],
           'evaluation_only':{'capsule_true_position_m':[9,9,9]}}
    visual={'capsule_measurement_t':.9,'slot_measurement_t':.9,'slot_visible':True,
            'slot_observation':{'motion_pending':False},'slot_estimate':base['fixture_position_m'],
            'slot_estimated_yaw_rad':0.}
    return frame,visual


def test_first_opportunity_is_not_replaced_by_later_successful_closure():
    first,_=sample();first['state_after']='APPROACH_OBJECT'
    later,_=sample();later['outcome']='future_success'
    frames=[first,later]
    chosen=next(i for i,f in enumerate(frames) if eligibility(f,'release_phase'))
    assert chosen==0
    obs=observation(first,sample()[1]);assert usability(first,obs,protocol())=='controller_left_first_descent'


def test_recovery_requires_retry_and_missing_opportunity_stays_missing():
    f,_=sample();f['placement_retries']=0
    assert eligibility(f,'nominal') and not eligibility(f,'phase')
    f['state_before']='ACQUIRE_OBJECT';f['placement_retries']=1
    assert not eligibility(f,'phase')


def test_observation_ignores_true_pose_and_never_invents_missing_axis():
    f,v=sample();a=observation(f,v);f['evaluation_only']['capsule_true_position_m']=[-8,-8,-8]
    assert observation(f,v)==a and usability(f,a,protocol()) is None
    f['capsule_estimated_axis']=None
    assert usability(f,observation(f,v),protocol())=='missing_capsule_axis'


def test_stale_observation_is_reported_instead_of_selecting_later_capture():
    f,v=sample();v['capsule_measurement_t']=.2
    assert usability(f,observation(f,v),protocol())=='stale_observation'


def test_retention_with_contact_is_never_contact_free():
    assert outcome({'retained_lift':True,'collision_free_retained_lift':False})=='retained_with_environment_contact'
    assert outcome({'retained_lift':False,'collision_free_retained_lift':False})=='failed_lift'


def test_registered_cohort_counts_scene_seeds_separately_from_conditions():
    p=json.loads((ROOT/'config/fresh_grasp_protocol_v9.json').read_text())
    assert len(set(p['seeds']))==6 and len(p['conditions'])==4 and p['expected_episodes']==24
    assert set(p['seeds']).isdisjoint({500,501,502,503,504,505})
