"""Per-episode metrics and summary statistics."""
from __future__ import annotations

from typing import Any

import numpy as np

from .episode import Episode
from .task_fsm import S


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    L = float(ab @ ab)
    u = 0.0 if L < 1e-12 else float(np.clip((p - a) @ ab / L, 0, 1))
    return float(np.linalg.norm(p - (a + u * ab)))


def perturbation_metrics(ep: Episode) -> list[dict[str, Any]]:
    """For every perturbation event:
    recovery_time_s: from perturbation start until the EE is back within approach_tol of
        its current target, or the FSM has moved past the state it was perturbed in.
    max_deviation_m: max distance of the EE from the straight path between its position at
        perturbation start and the current target, until recovery.
    """
    L = ep.log
    if not L.t:
        return []
    t = np.array(L.t)
    ee = np.array(L.ee_pos)
    pd = np.array(L.p_des)
    err = np.array(L.pos_err)
    st = np.array(L.state)
    tol = ep.cfg["task"]["approach_tol_m"]
    out = []
    for ev in ep.perturber.events:
        i0 = int(np.searchsorted(t, ev.t_start))
        if i0 >= len(t):
            continue
        s0 = st[i0]
        i_end = int(np.searchsorted(t, ev.t_end))
        rec = None
        for i in range(max(i_end, i0 + 1), len(t)):
            if err[i] < tol or st[i] > s0 or st[i] == S.DONE:
                rec = i
                break
        last = rec if rec is not None else len(t) - 1
        dev = max(_seg_dist(ee[i], ee[i0], pd[i]) for i in range(i0, last + 1))
        out.append({"kind": ev.kind, "state": ev.state, "t": ev.t_start,
                    "magnitude": ev.magnitude, "max_deviation_m": dev,
                    "recovery_time_s": (t[rec] - ev.t_start) if rec is not None else np.nan,
                    "recovered": rec is not None})
    return out


def episode_row(ep: Episode, result: dict[str, Any]) -> dict[str, Any]:
    pm = perturbation_metrics(ep)
    row = {"seed": ep.seed, "mode": ep.mode, "perception": ep.perception_mode,
           "perturb": "+".join(ep.kinds) or "none", **result,
           "num_perturbations": len(pm),
           "num_regrasps": ep.fsm.stats.num_regrasps,
           "num_realigns": ep.fsm.stats.num_realigns}
    row['protocol'] = ep.perturber.protocol
    row['plan_id'] = ep.perturber.plan_id()
    row['planned_events'] = len(ep.perturber.schedule)
    row['delivered_events'] = sum(e.plan_index is not None for e in ep.perturber.events)
    row['all_events_delivered'] = all(p.fired for p in ep.perturber.schedule)
    for kind in ("arm_push", "slot_move", "object_nudge"):
        evs = [p for p in pm if p["kind"] == kind]
        row[f"{kind}_magnitude"] = evs[0]["magnitude"] if evs else np.nan
        row[f"{kind}_recovery_s"] = evs[0]["recovery_time_s"] if evs else np.nan
        row[f"{kind}_max_dev_m"] = evs[0]["max_deviation_m"] if evs else np.nan
    est = np.array(ep.log.est_err) if ep.log.est_err else np.zeros((1, 2))
    row["mean_est_err_capsule_m"] = float(est[:, 0].mean())
    row["mean_est_err_slot_m"] = float(est[:, 1].mean())
    row['visual_guard_ticks']=ep.fsm.stats.visual_guard_ticks
    row['learned_sample_count']=ep.perception._learned.sample_index if ep.perception._learned is not None else 0
    return row


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (float(max(0, c - h)), float(min(1, c + h)))


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    groups: dict[tuple, list] = {}
    for r in rows:
        groups.setdefault((r['perception'], r.get('protocol', 'legacy_state'), r["mode"], r["perturb"]), []).append(r)
    for (perception, protocol, mode, pert), rs in groups.items():
        n, k = len(rs), sum(r["success"] for r in rs)
        g: dict[str, Any] = {"episodes": n, "successes": k, "success_rate": k / n,
                             "success_rate_ci95": wilson_ci(k, n)}
        g['failure_counts'] = {reason: sum(reason in r.get('failure_reason', '').split(';') for r in rs)
                               for reason in ('contained', 'supported', 'released', 'slow', 'stable', 'task_done', 'within_timeout')}
        g['legacy_center_successes'] = sum(r.get('legacy_center_success', False) for r in rs)
        g['episodes_with_all_events_delivered'] = sum(r.get('all_events_delivered', False) for r in rs)
        keys = ["time_to_complete_s", "final_place_error_m", "num_regrasps", "num_realigns",
                "num_perturbations", "mean_est_err_capsule_m", "mean_est_err_slot_m"]
        for kind in ("arm_push", "slot_move", "object_nudge"):
            keys += [f"{kind}_recovery_s", f"{kind}_max_dev_m"]
        for key in keys:
            v = np.array([r[key] for r in rs], float)
            v = v[~np.isnan(v)]
            if len(v):
                g[key] = {"mean": float(v.mean()), "std": float(v.std())}
        succ_t = [r["time_to_complete_s"] for r in rs if r["success"]]
        if succ_t:
            g["time_to_complete_s_successful"] = {"mean": float(np.mean(succ_t)),
                                                  "std": float(np.std(succ_t))}
        out[f"{perception}/{protocol}/{mode}/{pert}"] = g
    return out
