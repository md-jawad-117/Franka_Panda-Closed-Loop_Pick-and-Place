"""Interactive live demo in the MuJoCo viewer.

  python run_demo.py
  python run_demo.py --mode open_loop
  python run_demo.py --perception camera
"""
from __future__ import annotations

import argparse
import logging
import queue
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import glfw  # noqa: E402
import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402
import numpy as np  # noqa: E402

from cl_pickplace.config import load_config  # noqa: E402
from cl_pickplace.episode import Episode  # noqa: E402
from cl_pickplace.markers import MarkerDrawer  # noqa: E402
from cl_pickplace.interactive import drain_commands, auto_reset_due  # noqa: E402

CONTROLS = """
Controls
  Mouse : double-click a body to select it, then Ctrl + right drag to push it
          (arm links or capsule). The green slot is a mocap body: select it and
          Ctrl + right drag to move it, Ctrl + left drag to rotate it.
  P     : push the arm (random link, random horizontal direction)
  S     : move/retarget the slot (latest move replaces the previous one)
  O     : nudge the capsule (only when not grasped)
  R     : reset with new random capsule and slot poses
  F8    : toggle closed-loop / open-loop (applies on next reset)
  V     : cycle camera / learned / ground_truth and reset this episode
  SPACE : pause / resume control (physics keeps running)
  TAB / Shift+TAB : show the viewer's left / right UI panels
"""

KEYS = {ord("P"): "push", ord("S"): "slot", ord("O"): "nudge", ord("R"): "reset",
        glfw.KEY_F8: "mode", ord("V"): "perception", 32: "pause"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--mode", choices=["closed_loop", "open_loop"], default=None)
    ap.add_argument("--perception", choices=["ground_truth", "camera", "learned"], default=None)
    ap.add_argument("--perturb", default="none",
                    help="scheduled perturbations each episode: none | all | comma list")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=None,
                    help="close the viewer after this many wall seconds (for testing)")
    ap.add_argument('--auto-reset',action=argparse.BooleanOptionalAction,default=None,
                    help='automatically start a new scene after completion/timeout (profile default)')
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    cfg = load_config(args.config)
    ep = Episode(cfg, mode=args.mode, perception_mode=args.perception, perturb=args.perturb)
    pc = cfg["perturbations"]
    vc = cfg["viewer"]
    automatic_reset=vc.get('auto_reset',False) if args.auto_reset is None else args.auto_reset
    seed = args.seed
    ep.reset(seed)
    drawer = MarkerDrawer(cfg)
    next_mode = ep.mode
    keyq: queue.Queue = queue.Queue()
    last_event = "-"
    done_wall: float | None = None

    print(CONTROLS)
    print('Episode restart: automatic' if automatic_reset else 'Episode restart: manual — press R after success or timeout')

    def key_callback(keycode: int) -> None:
        if keycode in KEYS:
            keyq.put(KEYS[keycode])

    def handle(cmd: str) -> None:
        nonlocal seed, next_mode, last_event, done_wall
        t = ep.t
        if ep.done and cmd in ('push','slot','nudge'):
            last_event='episode finished; press R to start a new scene'
            return
        if cmd == "push":
            ev = ep.perturber.push_arm(t, pc["push_force_N"], pc["push_duration_s"])
            last_event = f"push {ev.params['link']} {ev.magnitude:.0f}N @{t:.1f}s"
        elif cmd == "slot":
            dyaw = np.radians(pc["slot_yaw_change_deg"]) * ep.perturber.rng.choice([-1, 1])
            ep.perturber.move_slot(t, pc["slot_shift_m"], dyaw, pc["slot_move_duration_s"])
            last_event = f"slot retargeted @{t:.1f}s"
        elif cmd == "nudge":
            if ep.grasped():
                last_event = "nudge ignored (grasped)"
            elif (ep.est is None or not ep.est.capsule.visible or ep.est.capsule.lost_frames > 0
                  or ep.sim.ee_pos()[2]-ep.est.capsule.pos[2] < pc['nudge_min_clearance_m']):
                last_event = 'nudge ignored (insufficient visible clearance)'
            else:
                ep.perturber.nudge_object(t, pc["nudge_force_N"], pc["nudge_duration_s"])
                last_event = f"object nudged @{t:.1f}s"
        elif cmd == "reset":
            seed += 1
            ep.mode = next_mode
            ep.reset(seed,capture=False)
            drawer.reset()
            done_wall = None
            last_event = f"reset seed {seed}"
        elif cmd == "mode":
            next_mode = "open_loop" if next_mode == "closed_loop" else "closed_loop"
            last_event = f"next reset: {next_mode}"
        elif cmd == "perception":
            choices=('camera','learned','ground_truth')
            ep.perception_mode=choices[(choices.index(ep.perception_mode)+1)%len(choices)]
            ep.reset(seed,capture=False)
            drawer.reset()
            done_wall=None
            last_event = f"perception: {ep.perception_mode}"
        elif cmd == "pause":
            ep.paused = not ep.paused
            last_event = "paused" if ep.paused else "resumed"

    try:
        with mujoco.viewer.launch_passive(ep.sim.model, ep.sim.data, key_callback=key_callback,
                                          show_left_ui=False, show_right_ui=False) as viewer:
            viewer.cam.lookat[:] = [0.45, 0.03, 0.45]
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 1.4, 145, -28
            wall0, sim0 = time.perf_counter(), ep.t
            start = wall0
            last_sync = last_status = 0.0

            def refresh(waiting=False):
                nonlocal last_sync,last_status
                if not viewer.is_running():raise InterruptedError('Viewer closed')
                if args.duration and time.perf_counter()-start>args.duration:raise InterruptedError('Demo duration finished')
                now=time.perf_counter()
                if now-last_sync>=1/vc['sync_hz']:
                    with viewer.lock():
                        viewer.user_scn.ngeom=0
                        drawer.draw(viewer.user_scn,ep,vc['show_targets'])
                    viewer.sync() # never called while the viewer lock is held
                    last_sync=now
                if now-last_status>=1/vc['status_hz']:
                    guard_labels={'reobserve_destination':'waiting for destination vision','reobserve_object':'waiting for capsule vision',
                                  'wait_initial_detections':'waiting for initial detections','bounded_grasp_commit':'final grasp: brief occlusion',
                                  'bounded_place_commit':'final placement: brief occlusion'}
                    activity=('processing vision' if waiting else 'press R to restart' if ep.done and not automatic_reset
                              else guard_labels.get(ep.fsm.guard_action,ep.fsm.guard_action) or 'running')
                    line=f'[{ep.mode}/{ep.perception_mode}] t={ep.t:5.1f}s {ep.fsm.state.name} | {activity} | {last_event}'
                    columns=max(40,shutil.get_terminal_size((160,20)).columns-1)
                    sys.stdout.write('\r'+line[:columns].ljust(columns));sys.stdout.flush()
                    last_status=now

            ep.perception.set_live_hooks(viewer.lock,lambda:refresh(waiting=True))
            while viewer.is_running():
                if args.duration and time.perf_counter()-start>args.duration:break
                with viewer.lock():
                    for command in drain_commands(keyq):
                        handle(command)
                        wall0,sim0=time.perf_counter(),ep.t
                # Only camera capture touches MuJoCo under the lock. Neural work
                # waits outside it and services viewer refreshes without advancing
                # physics or fabricating new observations.
                if not ep.done:ep.est=ep.perception.update()
                with viewer.lock():
                    if not (ep.done and not automatic_reset):
                        ep.step(perception_ready=True)
                        drawer.record(ep)
                    if ep.done and done_wall is None:
                        done_wall=time.perf_counter();result=ep.result
                        outcome='SUCCESS' if result['success'] else 'TIMEOUT' if result['timeout'] else 'FAIL'
                        last_event=f'{outcome}: {result["final_place_error_m"]*1000:.1f}mm'
                        print('\n'+last_event+(' — press R to start a new scene' if not automatic_reset else ' — auto reset enabled'))
                    if auto_reset_due(done_wall,time.perf_counter(),automatic_reset,cfg['task']['auto_reset_s']):
                        handle('reset')
                        wall0,sim0=time.perf_counter(),ep.t
                refresh()
                if ep.done and not automatic_reset:
                    time.sleep(1/vc['sync_hz'])
                    continue
                lag=(ep.t-sim0)-(time.perf_counter()-wall0)
                if lag>0:time.sleep(min(lag,.02))
    except InterruptedError:
        pass
    finally:
        print()
        ep.perception.close()


if __name__ == "__main__":
    main()
