"""Tilted grasp probes with opposed, supported pad contacts in a separate CAD world."""
import numpy as np
from .grasp_capture_planner import screen,base_observation,capsule_scenarios
from .grasp_planner import validate_observation,rotation


def candidates(obs,protocol):
    center=np.asarray(obs['capsule_position_m']);axis=np.array(obs['capsule_axis_world'],dtype=float,copy=True);axis/=np.linalg.norm(axis)
    base=np.asarray(obs['recorded_rotation']);actions=[]
    for roll in protocol['roll_deg']:
        R=base@rotation(np.array([1.,0,0]),np.radians(roll))
        for width in protocol['joint_widths_m']:
            for height in protocol['center_approach_offsets_m']:
                for axial in protocol['axial_offsets_m']:
                    position=center+axis*axial-R[:,2]*height
                    cost=abs(height-.002)/.004+abs(axial)/.008+abs(roll)/30+abs(width-.024)/.016
                    actions.append({'name':f'roll{roll:+03d}_w{width*1000:.0f}_h{height*1000:.0f}_a{axial*1000:+.0f}',
                        'position_m':position.tolist(),'rotation':R.tolist(),'approach_joint_width_m':width,'rank_cost':float(cost)})
    assert len(actions)==protocol['candidate_count']
    return sorted(actions,key=lambda a:(a['rank_cost'],a['name']))


def supported_capture(world,obs,action,protocol,capture_settings):
    results=[];R=np.asarray(action['rotation']);tcp=np.asarray(action['position_m'])
    for name,center,axis in capsule_scenarios(obs,protocol,True):
        world.set_capsule(center,axis)
        result=world.capture(tcp,R,action['approach_joint_width_m'],capture_settings)
        normals=[];local_z=[]
        if result['geometrically_capturable']:
            for contact in result['interception']['pad_contacts'].values():
                point=np.asarray(contact['position_m']);delta=point-center
                segment=np.clip(delta@axis,-protocol['capsule_half_segment_m'],protocol['capsule_half_segment_m'])
                inward=center+axis*segment-point;inward/=max(np.linalg.norm(inward),1e-12)
                normals.append(inward);local_z.append(float(((point-tcp)@R)[2]))
        opposing=float(np.linalg.norm(sum(normals))) if len(normals)==2 else None
        good=(len(normals)==2 and opposing<=protocol['maximum_normal_sum_norm'] and max(local_z)<=protocol['maximum_contact_tcp_z_m'])
        results.append({'scenario':name,'capturable':result['geometrically_capturable'],'normal_sum_norm':opposing,
            'maximum_contact_tcp_z_m':max(local_z) if local_z else None,'supported_antipodal':bool(good)})
        if not good:break
    return {'passed':all(r['supported_antipodal'] for r in results),'scenarios':results}


def plan(obs,world,parent,protocol,capture_settings):
    reason=validate_observation(base_observation(obs),parent)
    if reason:return {'status':'abstain','reason':reason,'ranked':[],'rejections':{}}
    ranked=[];rejections={};actions=candidates(obs,protocol)
    for action in actions:
        support=supported_capture(world,obs,action,protocol,capture_settings)
        if not support['passed']:
            rejections['unsupported_or_nonopposing_capture']=rejections.get('unsupported_or_nonopposing_capture',0)+1;continue
        clearance=screen(obs,action,world.environment_clearance,parent,protocol['minimum_clearance_m'])
        if not clearance['feasible']:
            rejections[clearance['reason']]=rejections.get(clearance['reason'],0)+1;continue
        ranked.append({**action,'support':support,'clearance':clearance})
        if len(ranked)>=protocol['maximum_ik_candidates']:break
    return {'status':'candidates' if ranked else 'abstain','reason':'requires_endpoint_ik' if ranked else 'no_supported_clear_grasp',
        'ranked':ranked,'rejections':rejections,'candidate_count':len(actions)}
