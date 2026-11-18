"""Post-initial-result component ablation; all earlier geometry parameters frozen."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from evaluate_grasp_capture_v8 import prepare,calibration,planning_world,select
from evaluate_grasp_planner_v7 import sha,robot_templates,physical_trial
from cl_pickplace.sim import Sim
from cl_pickplace.grasp_collision import extract_robot_cad
from cl_pickplace.grasp_capture_planner import base_observation,capture_check
from probe_grasp_feasibility_v5 import trial as original_trial


def ablation_protocol():
    parent=json.loads((ROOT/'config/grasp_capture_protocol_v8.json').read_text())
    overlay=json.loads((ROOT/'config/grasp_capture_ablation_v8.json').read_text())
    assert overlay['extends']=='config/grasp_capture_protocol_v8.json'
    return {**parent,**overlay}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',required=True);p.add_argument('--out',required=True,type=Path);args=p.parse_args()
    source,old,original,obs,parent,_=prepare(args.case);protocol=ablation_protocol();assert args.case in protocol['cases']
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True);cfg=original['config'];sim=Sim(cfg);cad=extract_robot_cad(sim.model,calibration(sim));templates=robot_templates(sim)
    previous=json.loads((ROOT/'training/results/grasp-capture-v8'/args.case/'manifest.json').read_text())
    for rel,digest in previous['source_sha256'].items():assert sha(ROOT/rel)==digest,rel
    extra=['scripts/evaluate_capture_ablation_v8.py','config/grasp_capture_ablation_v8.json']
    m={**previous,'protocol':protocol,'source_sha256':{**previous['source_sha256'],**{rel:sha(ROOT/rel) for rel in extra}},
       'initial_comparison_manifest_sha256':sha(ROOT/'training/results/grasp-capture-v8'/args.case/'manifest.json')}
    (args.out/'manifest.json').write_text(json.dumps(m,indent=2,allow_nan=False));decisions={};rows=[]
    with np.load(source/'evaluation_only/control_states.npz',allow_pickle=False) as states:
        arm=states['qpos'][old['checkpoint_index'],sim.arm_qadr].copy()
        for name,mode in protocol['modes'].items():
            world=planning_world(cad,cfg,obs,parent,mode);decision=select(sim,arm,obs,parent,protocol,mode,world,templates);decisions[name]=decision
            (args.out/(name+'_decision.json')).write_text(json.dumps(decision,indent=2,allow_nan=False))
            print(json.dumps({'case':args.case,'mode':name,'status':decision['status'],'selected':None if decision['selected'] is None else decision['selected']['name'],'screened':decision['screened_count']}),flush=True)
        for name,decision in decisions.items():
            physical=None;diagnostic=None
            if decision['selected']:
                world=planning_world(cad,cfg,obs,parent,protocol['modes']['obb_point_capture'])
                diagnostic=capture_check(world,obs,decision['selected'],protocol,False)
                physical=physical_trial(sim,cfg,states,old['checkpoint_index'],base_observation(obs),decision['selected'],parent,old['protocol'])
                (args.out/(name+'_physical.json')).write_text(json.dumps(physical,allow_nan=False))
            rows.append({'mode':name,'status':decision['status'],'selected':decision['selected'],'capture_diagnostic':diagnostic,
                         'physical':None if physical is None else {k:v for k,v in physical.items() if k!='trace'}})
            print(json.dumps({'case':args.case,'mode':name,'physical':None if physical is None else {k:physical[k] for k in ('retained_lift','collision_free_retained_lift','fixture_contact_ticks')}}),flush=True)
        reference={'name':'recorded_visual_reference','position_m':obs['recorded_target_m'],'rotation':obs['recorded_rotation'],'approach_joint_width_m':.08}
        common=physical_trial(sim,cfg,states,old['checkpoint_index'],base_observation(obs),reference,parent,old['protocol'])
        (args.out/'recorded_reference_common_physical.json').write_text(json.dumps(common,allow_nan=False))
        control=original_trial(sim,cfg,states,old['checkpoint_index'],{'name':'recorded_target','position_m':obs['recorded_target_m'],'quaternion':old['planning_observation']['gripper_quaternion']},obs['fixture_position_m'],old['protocol'])
        expected=json.loads((ROOT/'training/results/recovery-diagnostics-v5/feasibility'/args.case/'recorded_target.json').read_text());assert expected==control
    (args.out/'summary.json').write_text(json.dumps({'case':args.case,'rows':rows,'original_reference_exact_replay':True,
        'common_reference':{k:v for k,v in common.items() if k!='trace'},'original_reference':{k:v for k,v in control.items() if k!='trace'}},indent=2,allow_nan=False))


if __name__=='__main__':main()
