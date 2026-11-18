"""Run single episodes of the v24 closed-loop controller (headless).

Same configuration and episode setup as scripts/benchmark_occlusion_v24.py, but
without frozen-cohort manifests or result archives. Example:

    python run_v24.py --seed 605 --condition release_phase
    python run_v24.py --seed 600 --condition nominal --method confirmed_v23
"""
from pathlib import Path
import argparse,json,sys,time
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from cl_pickplace.episode import Episode
from cl_pickplace.confirmed_blind_centering import ConfirmedBlindCenterFSM
from cl_pickplace.occlusion_gated_centering import OcclusionGatedCenterFSM
from benchmark_occlusion_v24 import configurations,PROTOCOL
from benchmark_reacquisition_v4 import recovery_evaluation_trace

CONDITIONS=('nominal','late','phase','release_phase')


def run_episode(seed,condition,method='occlusion_v24'):
    cfg=configurations(json.loads(PROTOCOL.read_text()),condition)[method]
    ep=Episode(cfg,mode='closed_loop',perception_mode='learned',perturb='none' if condition=='nominal' else 'slot_move')
    ep.fsm=(OcclusionGatedCenterFSM if method=='occlusion_v24' else ConfirmedBlindCenterFSM)(cfg,ep.sim.home_q,ep.sim.table_z,'closed_loop')
    ep._log=recovery_evaluation_trace(ep,ep._log)
    try:
        start=time.perf_counter();ep.reset(seed)
        while not ep.done:ep.step()
        return {'seed':seed,'condition':condition,'method':method,'result':ep.result,
                'transitions':ep.fsm.stats.transitions,'centering_attempts':getattr(ep.fsm,'centering_attempts',0),
                'wall_seconds':round(time.perf_counter()-start,1)}
    finally:ep.perception.close()


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--condition',choices=CONDITIONS,default='nominal')
    p.add_argument('--method',choices=['occlusion_v24','confirmed_v23'],default='occlusion_v24')
    p.add_argument('--json',action='store_true',help='print the full result record as JSON')
    a=p.parse_args();row=run_episode(a.seed,a.condition,a.method)
    if a.json:print(json.dumps(row,indent=2));return
    r=row['result']
    print(f"seed {a.seed} {a.condition} {a.method}: {'SUCCESS' if r['success'] else 'FAIL'} "
          f"(final state {r['final_state']}, {r['time_to_complete_s']:.1f} s, min clearance {1000*r['minimum_xy_clearance_m']:.2f} mm"
          +('' if r['success'] else f", failed: {r['failure_reason']}")+f") [{row['wall_seconds']} s wall]")


if __name__=='__main__':main()
