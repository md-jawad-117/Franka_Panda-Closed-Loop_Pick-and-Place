"""Isolated learned-pose/CAD planner experiment; no frozen controller is modified."""
from pathlib import Path
import argparse, copy, hashlib, json, sys, time
import numpy as np
import mujoco

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from cl_pickplace.sim import Sim
from cl_pickplace.controller import TaskSpaceController
from cl_pickplace.kinematics import mat_to_quat, orientation_error, site_jacobian, dls_step
from cl_pickplace.grasp_planner import plan, waypoints
from probe_grasp_feasibility_v5 import restore_checkpoint, trial as reference_trial
from recovery_diagnostics_v5 import contact_diagnostics


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def robot_templates(sim):
    """CAD calibration only, in a new data object; never read live scene poses."""
    m=sim.model;d=mujoco.MjData(m);d.qpos[sim.arm_qadr]=sim.home_q
    d.qpos[sim.finger_qadr]=0.;mujoco.mj_kinematics(m,d)
    tcp=d.site_xpos[sim.ee_site];r=d.site_xmat[sim.ee_site].reshape(3,3)
    centers=[];axes=[];half=[];directions=[];gids=[]
    bodies={sim.link_bodies['hand']:0,sim.link_bodies['left_finger']:1,sim.link_bodies['right_finger']:-1}
    for g in range(m.ngeom):
        body=int(m.geom_bodyid[g])
        if body not in bodies or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
        gr=d.geom_xmat[g].reshape(3,3)
        if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH:
            mesh=int(m.geom_dataid[g]);start=int(m.mesh_vertadr[mesh]);n=int(m.mesh_vertnum[mesh])
            v=m.mesh_vert[start:start+n];lo=v.min(0);hi=v.max(0);c=(lo+hi)/2;h=(hi-lo)/2
        elif m.geom_type[g]==mujoco.mjtGeom.mjGEOM_BOX:c=np.zeros(3);h=m.geom_size[g].copy()
        else:raise ValueError('Unsupported gripper collision geometry')
        centers.append(((d.geom_xpos[g]+gr@c-tcp)@r).tolist());axes.append((r.T@gr).tolist());half.append(h.tolist())
        directions.append([0,bodies[body],0]);gids.append(g)
    return {'centers':centers,'axes':axes,'half_sizes':half,'width_directions':directions,'geometry_ids':gids}


def kinematic_check(sim, arm_q, obs, action, protocol):
    """Joint-limited endpoint IK in independent robot data; no scene/contact reads.

    This does not certify swept arm/self collision or dynamic trackability.
    """
    d=mujoco.MjData(sim.model);d.qpos[sim.arm_qadr]=arm_q
    settings=protocol['ik'];results=[]
    for name,p,r,width in waypoints(obs,action,protocol):
        target=mat_to_quat(r);arrived=False
        for i in range(settings['iterations']):
            mujoco.mj_kinematics(sim.model,d);mujoco.mj_comPos(sim.model,d)
            err=np.r_[p-d.site_xpos[sim.ee_site],orientation_error(target,d.site_xmat[sim.ee_site].reshape(3,3))]
            pos=float(np.linalg.norm(err[:3]));angle=float(np.degrees(np.linalg.norm(err[3:])))
            if pos<=settings['position_tolerance_m'] and angle<=settings['rotation_tolerance_deg']:
                arrived=True;break
            J=site_jacobian(sim.model,d,sim.ee_site,sim.arm_dadr)
            dq=dls_step(J,err,settings['damping'])
            d.qpos[sim.arm_qadr]=np.clip(d.qpos[sim.arm_qadr]+np.clip(dq,-settings['maximum_joint_step_rad'],settings['maximum_joint_step_rad']),sim.jnt_range[:,0],sim.jnt_range[:,1])
        results.append({'phase':name,'arrived':arrived,'iterations':i+1,'position_error_m':pos,'rotation_error_deg':angle})
        if not arrived:return {'feasible':False,'phases':results}
    return {'feasible':True,'phases':results}


def physical_trial(sim,cfg,states,index,obs,action,protocol,retention):
    restore_checkpoint(sim,states,index);controller=TaskSpaceController(sim,cfg)
    controller.q_target=sim.data.ctrl[sim.arm_act].copy();cap0=sim.capsule_pos().copy();start=sim.time
    records=[];phase_results=[];reason='completed'
    for name,target,r,width in waypoints(obs,action,protocol):
        phase_start=sim.time;arrived=False;q=mat_to_quat(r)
        control=float(np.clip(255*width/.08,0,255))
        for _ in range(int(np.ceil(protocol['phase_deadlines_s'][name]/sim.dt_control))):
            controller.step(target,q);sim.set_gripper(control);sim.step(sim.n_substeps)
            mujoco.mj_forward(sim.model,sim.data)
            contacts=contact_diagnostics(sim);cap=sim.capsule_pos();position=float(np.linalg.norm(target-sim.ee_pos()))
            angle=float(np.degrees(np.linalg.norm(orientation_error(q,sim.ee_mat()))))
            bodies=set()
            for c in sim.data.contact:
                if c.dist<=0 and sim.capsule_geom in (c.geom1,c.geom2):
                    other=int(c.geom2 if c.geom1==sim.capsule_geom else c.geom1);bodies.add(int(sim.model.geom_bodyid[other]))
            bilateral=all(sim.link_bodies[n] in bodies for n in ('left_finger','right_finger'))
            fixture=contacts['finger_fixture']+contacts['hand_fixture'];table=contacts['finger_table']
            penetration=max((c['penetration_m'] for c in fixture+table),default=0.)
            records.append({'t':sim.time-start,'phase':name,'capsule_position_m':cap.tolist(),'ee_position_m':sim.ee_pos().tolist(),
                            'rise_m':float(cap[2]-cap0[2]),'capsule_tcp_distance_m':float(np.linalg.norm(cap-sim.ee_pos())),
                            'finger_width_m':sim.finger_width(),'bilateral_contact':bilateral,
                            'finger_fixture_contact':bool(contacts['finger_fixture']),'hand_fixture_contact':bool(contacts['hand_fixture']),
                            'finger_table_contact':bool(table),'maximum_fixture_penetration_m':max((c['penetration_m'] for c in fixture),default=0.),
                            'maximum_environment_penetration_m':penetration,'position_error_m':position,'rotation_error_deg':angle})
            if penetration>retention['maximum_finger_fixture_penetration_m']:
                reason='collision_penetration_limit';break
            arrived=position<retention['position_tolerance_m'] and angle<retention['rotation_tolerance_deg']
            if name not in ('close','hold','preshape') and arrived:break
        phase_results.append({'phase':name,'arrived':arrived,'duration_s':sim.time-phase_start})
        if reason!='completed':break
        if name not in ('close','hold','preshape') and not arrived:reason=name+'_tracking_timeout';break
    window=[r for r in records if r['t']>=records[-1]['t']-retention['retention_window_s']-1e-9]
    success=(reason=='completed' and len(window)>=2 and window[-1]['t']-window[0]['t']>=retention['retention_window_s']-1e-8 and
             all(r['rise_m']>=retention['minimum_capsule_rise_m'] and r['capsule_tcp_distance_m']<=retention['maximum_capsule_tcp_distance_m'] and
                 r['bilateral_contact'] and cfg['gripper']['min_grasp_width']<=r['finger_width_m']<=cfg['gripper']['max_grasp_width'] for r in window))
    fixture_ticks=sum(r['finger_fixture_contact'] or r['hand_fixture_contact'] for r in records)
    table_ticks=sum(r['finger_table_contact'] for r in records)
    return {'action':action,'termination':reason,'phase_results':phase_results,'retained_lift':bool(success),
            'collision_free_retained_lift':bool(success and fixture_ticks==0 and table_ticks==0),
            'fixture_contact_ticks':fixture_ticks,'finger_table_contact_ticks':table_ticks,
            'max_rise_mm':1000*max(r['rise_m'] for r in records),'trace':records}


def load_case(case):
    base=ROOT/'training/results/recovery-diagnostics-v5';source=base/case
    manifest=json.loads((base/'feasibility'/case/'manifest.json').read_text())
    collector=json.loads((source/'manifest.json').read_text())
    for rel,digest in {**manifest['source_sha256'],**collector['frozen_v4_source_sha256']}.items():assert sha(ROOT/rel)==digest,rel
    original=json.loads((ROOT/collector['original_episode_path']).read_text());i=manifest['checkpoint_index'];v=original['visual_trace'][i]
    old=manifest['planning_observation']
    obs={'t':v['t'],'capsule_t':v['capsule_measurement_t'],'fixture_t':v['slot_measurement_t'],
         'capsule_visible':v['capsule_visible'],'fixture_visible':v['slot_visible'],
         'motion_pending':v['slot_observation']['motion_pending'],
         'capsule_position_m':old['capsule_position_m'],'capsule_axis_world':old['capsule_axis_world'],
         'fixture_position_m':v['slot_estimate'],'fixture_yaw_rad':v['slot_estimated_yaw_rad'],
         'ee_position_m':v['ee_pos'],'ee_rotation':None,'finger_width_m':v['finger_width_m']}
    frames=[json.loads(line) for line in (source/'evaluation_only/frames.jsonl').read_text().splitlines()]
    obs['ee_rotation']=frames[i]['ee_rotation']
    return source,manifest,original,obs


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--case',required=True);parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args();pp=ROOT/'config/grasp_planner_protocol_v7.json';protocol=json.loads(pp.read_text());assert args.case in protocol['cases']
    source,old,original,obs=load_case(args.case)
    if args.out.exists():raise FileExistsError(args.out)
    args.out.mkdir(parents=True);cfg=original['config'];sim=Sim(cfg);templates=robot_templates(sim)
    assert protocol['capsule_prior']=={'radius_m':cfg['scene']['capsule']['radius'],'half_length_m':cfg['scene']['capsule']['half_length']}
    assert protocol['table_z_m']==cfg['scene']['table_top_z']
    paths=[pp,Path(__file__),ROOT/'src/cl_pickplace/grasp_planner.py',source/'complete.json',source/'evaluation_only/control_states.npz',source/'evaluation_only/frames.jsonl',ROOT/'scripts/probe_grasp_feasibility_v5.py']
    manifest={'case':args.case,'protocol':protocol,'visual_observation':obs,'robot_cad':templates,'checkpoint_index':old['checkpoint_index'],
              'inputs_sha256':{p.relative_to(ROOT).as_posix():sha(p) for p in paths},'frozen_v4_sha256':json.loads((source/'manifest.json').read_text())['frozen_v4_source_sha256'],
              'scope':protocol['scope'],'selection_boundary':'Only visual_observation, robot CAD, static scene dimensions, design protocol and measured arm joints enter selection. Target state only initializes/evaluates the physical action.'}
    (args.out/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False))
    rows=[]
    with np.load(source/'evaluation_only/control_states.npz',allow_pickle=False) as states:
        arm_q=states['qpos'][old['checkpoint_index'],sim.arm_qadr].copy()
        for mode,bounds in protocol['modes'].items():
            start=time.perf_counter();decision=plan(obs,templates,cfg['scene']['slot'],protocol,bounds)
            chosen=None;checks=[]
            for action in decision['ranked']:
                check=kinematic_check(sim,arm_q,obs,action,protocol);checks.append({'action':action['name'],**check})
                if check['feasible']:chosen=action;break
            elapsed=time.perf_counter()-start
            decision.update(selected=chosen,kinematic_checks=checks,planning_wall_seconds=elapsed,
                            status='execute_assay' if chosen else 'abstain')
            if chosen is None and decision['ranked']:decision['reason']='ranked_kinematic_candidates_exhausted'
            (args.out/(mode+'_decision.json')).write_text(json.dumps(decision,indent=2,allow_nan=False))
            result=None
            if chosen:
                result=physical_trial(sim,cfg,states,old['checkpoint_index'],obs,chosen,protocol,old['protocol'])
                (args.out/(mode+'_physical.json')).write_text(json.dumps(result,allow_nan=False))
            rows.append({'mode':mode,'status':decision['status'],'reason':decision['reason'],
                         'screened_count':decision.get('screened_count',0),'selected':chosen,
                         'planning_wall_seconds':elapsed,'physical':None if result is None else {k:v for k,v in result.items() if k!='trace'}})
            print(json.dumps({'case':args.case,**rows[-1]}),flush=True)
        # Exact original assay control, separately labeled (its trajectory differs).
        action={'name':'recorded_target','position_m':old['planning_observation']['recorded_target_m'],
                'quaternion':old['planning_observation']['gripper_quaternion']}
        reference=reference_trial(sim,cfg,states,old['checkpoint_index'],action,obs['fixture_position_m'],old['protocol'])
        previous=json.loads((ROOT/'training/results/recovery-diagnostics-v5/feasibility'/args.case/'recorded_target.json').read_text())
        assert reference==previous,'Original reference assay changed'
    summary={'case':args.case,'rows':rows,'original_reference_exact_replay':True,'reference':{k:v for k,v in reference.items() if k!='trace'}}
    (args.out/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))


if __name__=='__main__':main()
