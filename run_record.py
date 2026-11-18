"""Headless run that records a presentation video.

  python run_record.py --seed 3 --perturb all
  python run_record.py --compare --seed 3 --perturb all
  python run_record.py --presentation            (slot moved twice + one arm push)
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from cl_pickplace.config import load_config, resolve  # noqa: E402
from cl_pickplace.episode import Episode  # noqa: E402
from cl_pickplace.recorder import record_episodes  # noqa: E402

PRESENTATION_SCHEDULE = [
    ("arm_push", "APPROACH_OBJECT", 0.35),
    ("slot_move", "TRANSPORT", 0.45),
    ("slot_move", "DESCEND_SLOT", 0.15),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--mode", choices=["closed_loop", "open_loop"], default=None)
    ap.add_argument("--perception", choices=["ground_truth", "camera", "learned"], default=None)
    ap.add_argument("--perturb", default="all", help="none | all | comma list of arm_push,slot_move,object_nudge")
    ap.add_argument("--compare", action="store_true", help="open-loop vs closed-loop side by side")
    ap.add_argument("--presentation", action="store_true",
                    help="fixed schedule: arm push, then slot moved twice")
    ap.add_argument("--camera", default=None, help="front_cam | top_cam | wrist_cam")
    ap.add_argument("--no-pip", action="store_true", help="no wrist_cam picture-in-picture")
    ap.add_argument("--out", default=None, help="output .mp4 path")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    cfg = load_config(args.config)
    rc = cfg["recording"]
    camera = args.camera or rc["camera"]
    pip = None if args.no_pip else rc["pip_camera"]
    perturb = "none" if args.presentation else args.perturb

    if args.compare:
        modes = ["open_loop", "closed_loop"]
        w, h = rc["width"] * 3 // 4, (rc["height"] * 3 // 4) // 16 * 16
    else:
        modes = [args.mode or cfg["task"]["mode"]]
        w, h = rc["width"], rc["height"]

    eps = []
    for m in modes:
        ep = Episode(cfg, mode=m, perception_mode=args.perception, perturb=perturb)
        ep.reset(args.seed)
        if args.presentation:
            ep.perturber.set_schedule(PRESENTATION_SCHEDULE)
        eps.append(ep)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = "compare" if args.compare else modes[0]
    if args.presentation:
        tag += "_presentation"
    out = Path(args.out) if args.out else resolve(rc["out_dir"]) / f"{stamp}_{tag}_{args.seed}.mp4"
    print(f"Recording {tag}, seed {args.seed} -> {out}")
    t0 = time.time()
    results = record_episodes(eps, cfg, out, w, h, camera, pip)
    for m, ep, r in zip(modes, eps, results):
        evs = ", ".join(f"{e.kind}@{e.t_start:.2f}s" for e in ep.perturber.events) or "none"
        print(f"  {m:12s} success={r['success']}  t={r['time_to_complete_s']:.2f}s  "
              f"place_err={r['final_place_error_m']*1000:.1f}mm  perturbations: {evs}")
    print(f"Done in {time.time() - t0:.1f}s: {out}")


if __name__ == "__main__":
    main()
