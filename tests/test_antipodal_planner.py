from pathlib import Path
import json,numpy as np
from cl_pickplace.antipodal_planner_v18 import candidates,supported_capture

ROOT=Path(__file__).resolve().parents[1]


class CapsuleContacts:
    def __init__(self,height=0):self.height=height
    def set_capsule(self,center,axis):self.center=np.array(center);self.axis=np.array(axis)
    def capture(self,position,rotation,width,settings):
        u=np.cross(self.axis,[0.,0,1]);u/=np.linalg.norm(u)
        v=np.cross(u,self.axis);v/=np.linalg.norm(v)
        radius=np.sqrt(.008**2-self.height**2)
        return {'geometrically_capturable':True,'interception':{'pad_contacts':{
            'left':{'position_m':(self.center+radius*u+self.height*v).tolist()},
            'right':{'position_m':(self.center-radius*u+self.height*v).tolist()}}}}


def inputs():
    protocol=json.loads((ROOT/'config/antipodal_protocol_v18.json').read_text())
    obs={'capsule_position_m':[0,0,0],'capsule_axis_world':[1,0,0],'recorded_rotation':np.diag([1,-1,-1]).tolist()}
    action={'position_m':[0,0,.002],'rotation':obs['recorded_rotation'],'approach_joint_width_m':.024}
    return protocol,obs,action


def test_opposing_equatorial_contacts_pass_and_downward_pair_is_rejected():
    p,o,a=inputs();assert supported_capture(CapsuleContacts(),o,a,p,{})['passed']
    assert not supported_capture(CapsuleContacts(.005),o,a,p,{})['passed']


def test_opposing_contacts_at_rounded_lower_pad_edge_do_not_pass_support_gate():
    p,o,a=inputs();a['position_m'][2]=.010
    assert not supported_capture(CapsuleContacts(),o,a,p,{})['passed']


def test_tilt_family_preserves_observation_and_is_properly_ranked():
    p,o,a=inputs();before=json.dumps(o);rows=candidates(o,p)
    assert len(rows)==180 and json.dumps(o)==before
    assert rows[0]['rank_cost']==0 and rows[0]['approach_joint_width_m']==.024
    for row in rows:
        R=np.array(row['rotation']);np.testing.assert_allclose(R.T@R,np.eye(3),atol=1e-12)
        assert abs(np.linalg.det(R)-1)<1e-12
