"""Batch evaluation: open-loop vs closed-loop under perturbations.

  python run_eval.py --episodes 100 --mode both --perturb all --headless
Outputs results/<run_id>/episodes.csv, summary.json, config.yaml and plots/.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from cl_pickplace.config import load_config, resolve  # noqa: E402
from cl_pickplace.episode import Episode, parse_perturb  # noqa: E402
from cl_pickplace.metrics import episode_row, summarize  # noqa: E402
from cl_pickplace.perturbations import KINDS  # noqa: E402
from cl_pickplace.task_fsm import S  # noqa: E402
from cl_pickplace.provenance import run_metadata  # noqa: E402

COLORS = {"closed_loop": "#2a9d8f", "open_loop": "#e76f51"}


def run_episodes(cfg, mode, perturb, perception, n, base_seed, live=False):
    ep = Episode(cfg, mode=mode, perception_mode=perception, perturb=perturb)
    rows, example, audits = [], None, []
    viewer = None
    if live:
        import mujoco.viewer
        viewer = mujoco.viewer.launch_passive(ep.sim.model, ep.sim.data)
    for i in range(n):
        ep.reset(base_seed + i)
        while not ep.done:
            ep.step()
            if viewer is not None:
                viewer.sync()
        row = episode_row(ep, ep.result)
        audits.append({'seed': ep.seed, 'mode': ep.mode, 'perception': ep.perception_mode,
                       'initial_poses': ep.poses, 'plan_id': ep.perturber.plan_id(),
                       'plan': ep.perturber.manifest(), 'delivered': ep.perturber.delivery(),
                       'most_recent_slot_lost_frames_at_event': [
                           ep.log.slot_lost_frames[max(0,int(np.searchsorted(ep.log.t,e.t_start,side='right'))-1)]
                           for e in ep.perturber.events],
                       'result': ep.result})
        if perception=='learned':
            audits[-1]['transitions']=ep.fsm.stats.transitions
            audits[-1]['learned_perception']=ep.perception.learned_summary()
            audits[-1]['visual_trace']=ep.log.visual
            audits[-1]['learned_capture_trace']=ep.perception._learned.history
        rows.append(row)
        if example is None and ep.perturber.events:
            example = {"t": list(ep.log.t), "err": list(ep.log.pos_err),
                       "state": list(ep.log.state),
                       "events": [(e.kind, e.t_start) for e in ep.perturber.events],
                       "seed": ep.seed, "success": row["success"]}
        ok = sum(r["success"] for r in rows)
        sys.stdout.write(f"\r  {mode:11s} {'+'.join(ep.kinds) or 'none':32s} "
                         f"{i + 1:4d}/{n}  success {ok}/{i + 1}")
        sys.stdout.flush()
    print()
    if viewer is not None:
        viewer.close()
    ep.perception.close()
    return rows, example, audits


def plot_success_vs_magnitude(rows, out: Path) -> None:
    kinds = [k for k in KINDS if any(not np.isnan(r[f"{k}_magnitude"]) for r in rows)]
    if not kinds:
        return
    fig, axes = plt.subplots(1, len(kinds), figsize=(4.5 * len(kinds), 3.6), squeeze=False)
    for ax, k in zip(axes[0], kinds):
        unit = "N" if k in ('arm_push', 'object_nudge') else "m"
        # Prefer runs with only this perturbation, so other perturbations do not mix in.
        alone = [r for r in rows if r["perturb"] == k]
        src = alone or rows
        ax.set_title(k.replace("_", " ") + ("" if alone else " (combined runs)"))
        for mode in sorted({r["mode"] for r in src}):
            rs = [r for r in src if r["mode"] == mode and not np.isnan(r[f"{k}_magnitude"])]
            if not rs:
                continue
            mags = np.array([r[f"{k}_magnitude"] for r in rs])
            succ = np.array([r["success"] for r in rs], float)
            edges = np.quantile(mags, np.linspace(0, 1, 5))
            centers, rates = [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                m = (mags >= lo) & (mags <= hi)
                if m.any():
                    centers.append(mags[m].mean())
                    rates.append(succ[m].mean())
            ax.plot(centers, rates, "o-", color=COLORS.get(mode), label=mode)
        ax.set_xlabel(f"magnitude [{unit}]")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("success rate")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_recovery_hist(rows, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 3.8))
    for k, c in zip(KINDS, ["#e63946", "#f4a261", "#9b5de5"]):
        v = np.array([r[f"{k}_recovery_s"] for r in rows if r["mode"] == "closed_loop"], float)
        v = v[~np.isnan(v)]
        if len(v):
            ax.hist(v, bins=20, alpha=0.6, color=c, label=f"{k} (n={len(v)})")
    ax.set_xlabel("recovery time [s]")
    ax.set_ylabel("count")
    ax.set_title("Closed-loop recovery time per perturbation")
    if ax.get_legend_handles_labels()[0]:
        ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def plot_error_timeseries(example, out: Path) -> None:
    if example is None:
        return
    fig, ax = plt.subplots(figsize=(9, 3.8))
    t, err = np.array(example["t"]), np.array(example["err"]) * 1000
    ax.plot(t, err, color="#264653", lw=1.2, label="EE position error to current target")
    st = np.array(example["state"])
    for i in np.nonzero(np.diff(st))[0]:
        ax.axvline(t[i + 1], color="0.8", lw=0.6, zorder=0)
        ax.text(t[i + 1], ax.get_ylim()[1] if False else err.max() * 1.02,
                S(st[i + 1]).name.replace("_", "\n"), fontsize=5.5, color="0.4", va="bottom")
    cols = {"arm_push": "#e63946", "slot_move": "#f4a261", "object_nudge": "#9b5de5"}
    for kind, te in example["events"]:
        ax.axvline(te, color=cols[kind], lw=2, label=kind)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error [mm]")
    ax.set_title(f"Closed-loop episode seed {example['seed']} "
                 f"({'success' if example['success'] else 'fail'})", fontsize=10)
    h, l = ax.get_legend_handles_labels()
    uniq = dict(zip(l, h))
    ax.legend(uniq.values(), uniq.keys(), fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def load_rows(csv_path: Path) -> list[dict]:
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out = {}
            for k, v in r.items():
                if v in ("True", "False"):
                    out[k] = v == "True"
                else:
                    try:
                        out[k] = float(v)
                    except ValueError:
                        out[k] = v
            rows.append(out)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--mode", choices=["open_loop", "closed_loop", "both"], default="both")
    ap.add_argument("--perturb", default="all",
                    help="none | arm_push | slot_move | object_nudge | all | per_type "
                         "(per_type = none + each kind alone + all)")
    ap.add_argument("--perception", choices=["ground_truth", "camera", "learned"], default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument('--protocol', choices=['time', 'state'], default=None,
                    help='time = matched exogenous trials; state = separate phase-triggered stress tests')
    ap.add_argument("--headless", action="store_true", help="no viewer (default)")
    ap.add_argument("--live", action="store_true", help="watch in the viewer (slow)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--replot", default=None, metavar="RUN_DIR",
                    help="only redraw the magnitude and recovery plots from RUN_DIR/episodes.csv")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    if args.replot:
        d = Path(args.replot)
        rows = load_rows(d / "episodes.csv")
        plot_success_vs_magnitude([r for r in rows if r["perturb"] != "none"],
                                  d / "plots" / "success_vs_perturbation_magnitude.png")
        plot_recovery_hist(rows, d / "plots" / "recovery_time_hist.png")
        print(f"Replotted {d}")
        return

    cfg = load_config(args.config)
    if args.protocol:
        cfg['perturbations']['schedule_mode'] = args.protocol
    n = cfg['eval']['episodes'] if args.episodes is None else args.episodes
    if n <= 0:
        ap.error('--episodes must be positive')
    modes = ["open_loop", "closed_loop"] if args.mode == "both" else [args.mode]
    if args.perturb == "per_type":
        perturbs = ["none", *KINDS, "all"]
    else:
        parse_perturb(args.perturb)
        perturbs = [args.perturb]
    perception = args.perception or cfg["perception"]["mode"]

    run_id = args.run_id or time.strftime("%Y%m%d_%H%M%S") + f"_{args.mode}_{args.perturb}_{perception}"
    out = resolve(cfg["eval"]["out_dir"]) / run_id
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite an evaluation run: {out}')
    (out / "plots").mkdir(parents=True, exist_ok=True)
    with open(out / "config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump({"args": vars(args), "config": cfg}, f, sort_keys=False)
    (out/'provenance.json').write_text(json.dumps(run_metadata(cfg, perception), indent=2, allow_nan=False))

    print(f"Evaluating {n} episodes x {modes} x {perturbs}, perception={perception}")
    t0 = time.time()
    rows, example, audits = [], None, []
    for pert in perturbs:
        for mode in modes:
            r, ex, au = run_episodes(cfg, mode, pert, perception, n, args.seed, live=args.live)
            rows += r
            audits += au
            if mode == "closed_loop" and ex is not None and (example is None or pert == "all"):
                example = ex

    with open(out / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(rows)
    with open(out/'events.jsonl', 'w', encoding='utf-8') as f:
        for audit in audits:
            f.write(json.dumps(audit, allow_nan=False) + '\n')
    with open(out / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plot_success_vs_magnitude([r for r in rows if r["perturb"] != "none"],
                              out / "plots" / "success_vs_perturbation_magnitude.png")
    plot_recovery_hist(rows, out / "plots" / "recovery_time_hist.png")
    plot_error_timeseries(example, out / "plots" / "ee_error_timeseries_example.png")

    print(f"\n{'group':38s} {'success':>9s}  {'95% CI':>15s}  {'place err':>9s}  {'time':>6s}")
    for k, g in summary.items():
        lo, hi = g["success_rate_ci95"]
        pe = g.get("final_place_error_m", {}).get("mean", float("nan")) * 1000
        tt = g.get("time_to_complete_s_successful", {}).get("mean", float("nan"))
        print(f"{k:38s} {g['success_rate']*100:8.1f}%  [{lo*100:5.1f}, {hi*100:5.1f}]  "
              f"{pe:7.1f}mm  {tt:5.1f}s")
    print(f"\nResults in {out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
