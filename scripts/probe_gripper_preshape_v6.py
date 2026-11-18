"""Declared partial-jaw/axial-grasp assay, retaining the original v5 mechanics."""
from pathlib import Path
import argparse,copy,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cl_pickplace.sim import Sim
from probe_grasp_feasibility_v5 import trial,sha


def candidates(visual,protocol):
    cap=np.asarray(visual['capsule_position_m']);axis=np.asarray(visual['capsule_axis_world'])[:2];axis=axis/np.linalg.norm(axis)
    q=visual['gripper_quaternion'];calibration=protocol['gripper_calibration']
    result=[{'name':'recorded_target_open','position_m':visual['recorded_target_m'],'quaternion':q,
             'approach_joint_width_m':calibration['full_open_joint_width_m'],'approach_ctrl':calibration['full_open_ctrl']}]
    for width in protocol['approach_joint_widths_m']:
        control=calibration['closed_ctrl']+(calibration['full_open_ctrl']-calibration['closed_ctrl'])*width/calibration['full_open_joint_width_m']
        control=float(np.clip(control,calibration['closed_ctrl'],calibration['full_open_ctrl']))
        for height in protocol['center_height_offsets_m']:
            for offset in protocol['axial_offsets_m']:
                result.append({'name':f'w{width*1000:02.0f}_h{height*1000:02.0f}_a{offset*1000:+03.0f}',
                               'position_m':(cap+np.r_[offset*axis,height]).tolist(),'quaternion':q,
                               'approach_joint_width_m':width,'approach_ctrl':control,
                               'center_height_offset_m':height,'axial_offset_m':offset})
    assert len(result)==protocol['entry_count_per_checkpoint']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',required=True);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();protocol_path=ROOT/'config/gripper_preshape_assay_v6.json';protocol=json.loads(protocol_path.read_text())
    assert args.case in protocol['cases'];base=ROOT/'training/results/recovery-diagnostics-v5'
    previous=base/'feasibility'/args.case/'manifest.json';manifest=json.loads(previous.read_text())
    for rel,checksum in manifest['source_sha256'].items():assert sha(ROOT/rel)==checksum,rel
    source=base/args.case;collector=json.loads((source/'manifest.json').read_text())
    for rel,checksum in collector['frozen_v4_source_sha256'].items():assert sha(ROOT/rel)==checksum,rel
    original=json.loads((ROOT/collector['original_episode_path']).read_text());visual=manifest['planning_observation']
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True);(args.out/'manifest.json').write_text(json.dumps({
        'case':args.case,'protocol':protocol,'v5_assay':manifest,'source_checkpoint_manifest_sha256':sha(previous),
        'new_source_sha256':{rel:sha(ROOT/rel) for rel in ('scripts/probe_gripper_preshape_v6.py','config/gripper_preshape_assay_v6.json')},
        'scope':'Selected-state mechanical assay only; same visual targets and initial physics, no claim of closed-loop policy success.'},indent=2))
    sim=Sim(original['config']);rows=[]
    with np.load(source/'evaluation_only/control_states.npz',allow_pickle=False) as states:
        for action in candidates(visual,protocol):
            cfg=copy.deepcopy(original['config'])
            # Only the assay's preposition/descent command changes. No simulator
            # collision, geometry, friction, capsule state or actuation limits change.
            cfg['gripper']['open_ctrl']=action['approach_ctrl']
            result=trial(sim,cfg,states,manifest['checkpoint_index'],action,visual['fixture_position_m'],manifest['protocol'])
            (args.out/(action['name']+'.json')).write_text(json.dumps(result,allow_nan=False))
            rows.append({key:value for key,value in result.items() if key!='trace'})
            print(json.dumps({'case':args.case,'action':action['name'],'retained':result['retained_lift'],
                              'collision_free_retained':result['collision_free_retained_lift'],'rise_mm':result['max_rise_mm']}),flush=True)
    (args.out/'summary.json').write_text(json.dumps({'case':args.case,'trials':rows},indent=2))


if __name__=='__main__':main()
