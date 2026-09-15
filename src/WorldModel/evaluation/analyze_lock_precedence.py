"""Multi-seed TEMPORAL-PRECEDENCE test: does station-lock onset precede the
throughput collapse?

READ-ONLY.  Imports nothing from the model / simulator (numpy + scipy only).
Run by FILE PATH (not ``-m``) so ``WorldModel/__init__`` (torch) is not imported::

    python WorldModel/evaluation/analyze_lock_precedence.py

Why this is not circular (the upgrade over seed-556)
----------------------------------------------------
``analyze_lock_onset.py`` could only compare lock facets against each other on a
single seed -- occupancy / committed / rejections are all downstream faces of
the lock, so "lock precedes collapse" was not cleanly separable.

The trace re-collection (``run_phase_c_physical_only_pp_trace.py``) now records,
per sampled tick, TWO INDEPENDENT measurements:

  * LOCK signal  L(t): demand-concentration = max-station share of
    per-interval ``admission_attempts`` (the ``share_attempts`` metric that the
    aggregate analysis validated), and
  * THROUGHPUT signal T(t): ``order_state.total_completed`` (completions) plus
    pending / in-progress backlog -- measured on the order side, NOT on the
    station admission side.

Precedence is only meaningful on runs that actually collapse.  A run is
labelled COLLAPSED by the paper's run-level definition -- it finishes with
fewer than 300 completed orders AND a mean deadlock ratio of at least 0.40.
Within such a run, the collapse-onset TICK is then located from the completion
RATE (first sustained stall while backlog stays high), giving a time for the
precedence lead.

Pre-registered parameters (do not tune to outcome)
--------------------------------------------------
See the constants block.  Reported per (load, arm):
  1. counts: seeds with data / with lock onset / with collapse / with both;
  2. lead = collapse_tick - lock_onset_tick over both-defined seeds: mean/median,
     bootstrap CI, sign test (#lead>0), Wilcoxon vs 0;
  3. cross-correlation lag between L and completion-rate (expect lock LEADS, i.e.
     negative correlation at positive lag);
  4. bivariate Granger F-test both directions (lock->rate vs rate->lock),
     per-seed p, fraction significant, Fisher-combined p.
Honest caveats are written into the markdown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TRACE_ROOT = (
    REPO / "WorldModel" / "checkpoints" / "phaseC_wm_onpolicy_round1_v1"
    / "phasec_physical_only_pp_trace_v1"
)
DEFAULT_OUT_DIR = (
    REPO / "WorldModel" / "checkpoints" / "phaseC_wm_onpolicy_round1_v1"
    / "phasec_physical_only_pp_factorial_900_950_v1"
    / "station_lock_causal_analysis"
)

# ---- pre-registered detection parameters -------------------------------------
SHARE_GRID = [0.6, 0.7, 0.8, 0.9]   # lock-onset: max-station attempt-share crossings
PRIMARY_SHARE = 0.8                 # headline lock-onset threshold
SUSTAIN = 3                         # consecutive samples a crossing must persist
SMOOTH_WINDOW = 5                   # centered rolling-mean samples (~50 ticks @stride10)
# Run-level COLLAPSE LABEL (matches the paper's definition): a collapsed run
# finishes with < COLLAPSE_MAX_ORDERS completed orders AND deadlock_ratio_mean
# >= COLLAPSE_MIN_DEADLOCK.  This decides WHICH runs collapsed (the precedence
# gate); the per-tick detector below only LOCATES the collapse-onset TICK within
# an already-labelled-collapsed run (for the lead), it no longer decides collapse.
COLLAPSE_MAX_ORDERS = 300
COLLAPSE_MIN_DEADLOCK = 0.40
COLLAPSE_RATE_FRAC = 0.25           # onset tick: completion rate < frac * early rate ...
COLLAPSE_BACKLOG_MULT = 1.5         # ... WHILE backlog >= mult * early backlog median
EARLY_FRACTION = 1.0 / 3.0          # "early/healthy" window = first third of samples
GRANGER_LAG = 3
GRANGER_ALPHA = 0.05
BOOTSTRAP_ITERS = 10000
BOOTSTRAP_SEED = 20260909


# ---- small helpers -----------------------------------------------------------
def rolling_mean(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return values
    half = window // 2
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = values[max(0, i - half): min(len(values), i + half + 1)].mean()
    return out


def first_sustained_crossing(ticks, series, thresh, sustain=SUSTAIN):
    run = 0
    for i, v in enumerate(series):
        run = run + 1 if v >= thresh else 0
        if run >= sustain:
            return float(ticks[i - sustain + 1])
    return None


def bootstrap_ci(values, iters=BOOTSTRAP_ITERS, seed=BOOTSTRAP_SEED):
    values = [float(v) for v in values]
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0]]
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    means = np.sort(rng.choice(arr, size=(iters, len(arr)), replace=True).mean(axis=1))
    lo = means[int(np.floor(0.025 * (iters - 1)))]
    hi = means[int(np.ceil(0.975 * (iters - 1)))]
    return [round(float(lo), 4), round(float(hi), 4)]


def granger_f_pvalue(y, x, lag=GRANGER_LAG):
    """H0: x does not Granger-cause y.  OLS restricted vs full, F-test.

    y, x are 1-D arrays (should be stationary; we pass completion-rate and the
    attempt-share level).  Returns (p_value, f_stat) or (None, None) if too short.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    if n <= 2 * lag + 2 or len(x) != n:
        return None, None
    rows = n - lag
    yt = y[lag:]
    # restricted design: intercept + lagged y
    Xr = [np.ones(rows)]
    for i in range(1, lag + 1):
        Xr.append(y[lag - i: n - i])
    Xr = np.column_stack(Xr)
    Xf = [Xr]
    for i in range(1, lag + 1):
        Xf.append((x[lag - i: n - i]).reshape(-1, 1))
    Xf = np.column_stack(Xf)

    def rss(design, target):
        beta, _, _, _ = np.linalg.lstsq(design, target, rcond=None)
        resid = target - design @ beta
        return float(resid @ resid)

    rss_r = rss(Xr, yt)
    rss_f = rss(Xf, yt)
    df_f = rows - Xf.shape[1]
    if df_f <= 0 or rss_f <= 0:
        return None, None
    f_stat = ((rss_r - rss_f) / lag) / (rss_f / df_f)
    if not np.isfinite(f_stat) or f_stat < 0:
        return None, None
    p = float(stats.f.sf(f_stat, lag, df_f))
    return p, float(f_stat)


def fisher_combine(pvals):
    pvals = [p for p in pvals if p is not None and 0 < p <= 1]
    if not pvals:
        return None
    stat = -2.0 * np.sum(np.log(pvals))
    return float(stats.chi2.sf(stat, 2 * len(pvals)))


def cross_corr_best_lag(lock, rate, max_lag):
    """Correlation of lock(t) with rate(t+tau) for tau in [-max_lag, max_lag].

    Lock LEADS a throughput drop => most-negative correlation at tau > 0.
    Returns (best_lag_samples, best_corr) where negative corr is the signal.
    """
    lock = np.asarray(lock, dtype=float)
    rate = np.asarray(rate, dtype=float)
    n = min(len(lock), len(rate))
    lock, rate = lock[:n], rate[:n]
    best_lag, best_corr = 0, 0.0
    for tau in range(-max_lag, max_lag + 1):
        if tau >= 0:
            a, b = lock[: n - tau], rate[tau:]
        else:
            a, b = lock[-tau:], rate[: n + tau]
        if len(a) < 5 or np.std(a) == 0 or np.std(b) == 0:
            continue
        c = float(np.corrcoef(a, b)[0, 1])
        if c < best_corr:  # track most negative (lock high -> rate low)
            best_corr, best_lag = c, tau
    return best_lag, round(best_corr, 4)


# ---- trace extraction --------------------------------------------------------
def extract_series(arm_block):
    """From one arm's trace block -> aligned per-tick series, or None."""
    st = arm_block.get("station_trace", {}).get("trace") or []
    tp = arm_block.get("throughput_trace", {}).get("trace") or []
    if len(st) < 6 or len(tp) < 6:
        return None

    # station side: cumulative admission_attempts per station, by tick
    st_by_tick = {}
    for row in st:
        tick = int(row["tick"])
        att = {int(s["station_id"]): float(s["admission_attempts"]) for s in row["stations"]}
        st_by_tick[tick] = att
    tp_by_tick = {int(r["tick"]): r for r in tp}

    ticks = sorted(set(st_by_tick) & set(tp_by_tick))
    if len(ticks) < 6:
        return None
    station_ids = sorted({sid for att in st_by_tick.values() for sid in att})

    attempts_cum = np.array(
        [[st_by_tick[t].get(sid, 0.0) for sid in station_ids] for t in ticks]
    )
    completed = np.array([float(tp_by_tick[t]["completed"]) for t in ticks])
    backlog = np.array(
        [float(tp_by_tick[t].get("pending", 0)) + float(tp_by_tick[t].get("in_progress", 0))
         for t in ticks]
    )
    ticks = np.array(ticks, dtype=float)

    # LOCK signal: per-interval attempt concentration (instantaneous, responsive)
    interval_att = np.diff(attempts_cum, axis=0)
    interval_att = np.clip(interval_att, 0, None)
    tot = interval_att.sum(axis=1)
    share_interval = np.divide(
        interval_att.max(axis=1), tot, out=np.zeros_like(tot), where=tot > 0
    )
    share_interval = rolling_mean(share_interval)
    # cumulative share (robustness)
    tot_c = attempts_cum.sum(axis=1)
    share_cum = np.divide(
        attempts_cum.max(axis=1), tot_c, out=np.zeros_like(tot_c), where=tot_c > 0
    )
    share_cum = rolling_mean(share_cum)

    # THROUGHPUT signal: completion rate per interval (independent of station side)
    dt = np.diff(ticks)
    rate = np.divide(np.diff(completed), dt, out=np.zeros_like(dt), where=dt > 0)
    rate = rolling_mean(rate)

    return {
        "ticks": ticks,                 # length T
        "ticks_mid": ticks[1:],         # length T-1 (aligned with interval series)
        "share_interval": share_interval,   # length T-1
        "share_cum": share_cum,             # length T
        "rate": rate,                       # length T-1
        "backlog": backlog,                 # length T
        "completed_final": float(completed[-1]),
    }


def detect_lock_onset(ser, thresh):
    return first_sustained_crossing(ser["ticks_mid"], ser["share_interval"], thresh)


def detect_collapse(ser):
    """Collapse tick: completion rate stalls WHILE backlog stays high.

    Returns tick or None.  Guards against the 'finished all orders' case by
    requiring elevated backlog at the stall point.
    """
    rate = ser["rate"]
    ticks_mid = ser["ticks_mid"]
    backlog = ser["backlog"][1:]  # align with interval series
    n = len(rate)
    if n < 6:
        return None
    early_n = max(2, int(n * EARLY_FRACTION))
    healthy_rate = float(np.median(rate[:early_n]))
    early_backlog = float(np.median(backlog[:early_n]))
    if healthy_rate <= 0:
        return None
    rate_thr = COLLAPSE_RATE_FRAC * healthy_rate
    backlog_thr = max(1.0, COLLAPSE_BACKLOG_MULT * early_backlog)
    run = 0
    for i in range(early_n, n):
        if rate[i] < rate_thr and backlog[i] >= backlog_thr:
            run += 1
        else:
            run = 0
        if run >= SUSTAIN:
            return float(ticks_mid[i - SUSTAIN + 1])
    return None


# ---- per (load, arm) aggregation --------------------------------------------
def analyze_group(records, load, arm):
    lock_onsets = {f"share_ge_{t}": [] for t in SHARE_GRID}
    leads = []                  # collapse_tick - lock_tick, collapsed & both-defined
    lead_pairs = []
    xcorr_lags, xcorr_corrs = [], []
    granger_lock_to_rate, granger_rate_to_lock = [], []
    n_data = 0
    n_collapsed = 0             # run-level LABEL (< COLLAPSE_MAX_ORDERS & >= deadlock)
    n_collapsed_onset_time = 0  # labelled collapsed AND collapse-onset tick found
    n_lock = 0                  # lock onset (share>=PRIMARY) over all runs
    n_lock_in_collapsed = 0     # lock onset among collapsed runs

    for rec in records:
        ser = rec["series"]
        seed = rec["seed"]
        n_data += 1
        completed = rec.get("completed_orders")
        deadlock = rec.get("deadlock_ratio_mean")
        is_collapsed = (
            completed is not None and deadlock is not None
            and completed < COLLAPSE_MAX_ORDERS and deadlock >= COLLAPSE_MIN_DEADLOCK
        )

        onset_primary = None
        for t in SHARE_GRID:
            o = detect_lock_onset(ser, t)
            lock_onsets[f"share_ge_{t}"].append(o)
            if abs(t - PRIMARY_SHARE) < 1e-9:
                onset_primary = o
        if onset_primary is not None:
            n_lock += 1

        if not is_collapsed:
            continue
        # ---- collapsed runs only: onset timing + precedence tests ------------
        n_collapsed += 1
        if onset_primary is not None:
            n_lock_in_collapsed += 1
        collapse_tick = detect_collapse(ser)
        if collapse_tick is not None:
            n_collapsed_onset_time += 1

        lag, corr = cross_corr_best_lag(ser["share_interval"], ser["rate"], GRANGER_LAG + 2)
        if corr < 0:
            xcorr_lags.append(lag)
            xcorr_corrs.append(corr)
        p_lr, _ = granger_f_pvalue(ser["rate"], ser["share_interval"], GRANGER_LAG)
        p_rl, _ = granger_f_pvalue(ser["share_interval"], ser["rate"], GRANGER_LAG)
        if p_lr is not None:
            granger_lock_to_rate.append(p_lr)
        if p_rl is not None:
            granger_rate_to_lock.append(p_rl)

        if onset_primary is not None and collapse_tick is not None:
            lead = collapse_tick - onset_primary
            leads.append(lead)
            lead_pairs.append({
                "seed": seed, "lock_tick": onset_primary, "collapse_tick": collapse_tick,
                "lead": lead, "completed_orders": completed,
                "deadlock_ratio_mean": round(deadlock, 4),
            })

    def onset_summary(vals):
        got = [v for v in vals if v is not None]
        return {
            "n_crossed": len(got),
            "median_tick": round(float(np.median(got)), 1) if got else None,
        }

    sign_pos = sum(1 for x in leads if x > 0)
    wilcoxon_p = None
    if len(leads) >= 6 and any(abs(x) > 1e-9 for x in leads):
        try:
            wilcoxon_p = float(stats.wilcoxon(leads, alternative="greater").pvalue)
        except ValueError:
            wilcoxon_p = None

    def sig_frac(pvals):
        got = [p for p in pvals if p is not None]
        if not got:
            return None
        return round(sum(1 for p in got if p < GRANGER_ALPHA) / len(got), 3)

    return {
        "load": load,
        "arm": arm,
        "n_seeds_with_data": n_data,
        "collapse_definition": f"completed_orders < {COLLAPSE_MAX_ORDERS} AND "
                               f"deadlock_ratio_mean >= {COLLAPSE_MIN_DEADLOCK}",
        "n_collapsed_runs": n_collapsed,
        "n_collapsed_with_onset_time": n_collapsed_onset_time,
        "n_with_lock_onset(share>=%.1f)" % PRIMARY_SHARE: n_lock,
        "n_lock_onset_in_collapsed": n_lock_in_collapsed,
        "n_with_both": len(leads),
        "lock_onset_by_threshold": {k: onset_summary(v) for k, v in lock_onsets.items()},
        "lead_collapse_minus_lock": {
            "n": len(leads),
            "mean": round(float(np.mean(leads)), 1) if leads else None,
            "median": round(float(np.median(leads)), 1) if leads else None,
            "ci95_mean": bootstrap_ci(leads),
            "sign_positive": f"{sign_pos}/{len(leads)}" if leads else "0/0",
            "wilcoxon_greater_p": wilcoxon_p,
        },
        "cross_correlation": {
            "n_negative_peak": len(xcorr_corrs),
            "median_lead_lag_samples": (round(float(np.median(xcorr_lags)), 1)
                                        if xcorr_lags else None),
            "median_peak_corr": (round(float(np.median(xcorr_corrs)), 3)
                                 if xcorr_corrs else None),
            "note": "collapsed runs only; positive lag (stride-samples) => lock LEADS the rate drop",
        },
        "granger": {
            "scope": "collapsed runs only",
            "lock_to_rate_sig_frac": sig_frac(granger_lock_to_rate),
            "lock_to_rate_fisher_p": fisher_combine(granger_lock_to_rate),
            "rate_to_lock_sig_frac": sig_frac(granger_rate_to_lock),
            "rate_to_lock_fisher_p": fisher_combine(granger_rate_to_lock),
            "n_tested": len(granger_lock_to_rate),
            "lag": GRANGER_LAG,
        },
        "lead_pairs": sorted(lead_pairs, key=lambda d: d["seed"]),
    }


# ---- IO ----------------------------------------------------------------------
def load_records(trace_root: Path, loads, arms):
    per_seed = trace_root / "per_seed"
    if not per_seed.is_dir():
        raise SystemExit(f"trace per_seed dir not found: {per_seed}")
    groups = {}  # (load, arm) -> [ {seed, series} ]
    files = sorted(per_seed.glob("*.json"))
    n_files = 0
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        load = payload.get("load")
        if loads and load not in loads:
            continue
        seed = int(payload.get("seed"))
        n_files += 1
        for arm, block in payload.get("arms", {}).items():
            if arms and arm not in arms:
                continue
            ser = extract_series(block)
            if ser is None:
                continue
            metrics = block.get("metrics", {}) or {}
            completed = block.get("completed_orders", metrics.get("completed_orders"))
            deadlock = metrics.get("deadlock_ratio_mean")
            groups.setdefault((load, arm), []).append({
                "seed": seed,
                "series": ser,
                "completed_orders": float(completed) if completed is not None else None,
                "deadlock_ratio_mean": float(deadlock) if deadlock is not None else None,
            })
    return groups, n_files


def render_md(report) -> str:
    lines = ["# Station-lock temporal precedence (multi-seed, independent throughput)\n"]
    lines.append(f"- Trace root: `{report['trace_root']}`  ({report['n_seed_files']} per-seed files)\n")
    lines.append(f"- Lock signal: max-station share of per-interval `admission_attempts` "
                 f"(smoothed {SMOOTH_WINDOW} samples); onset = share>= {PRIMARY_SHARE} sustained {SUSTAIN}.\n")
    lines.append("- Throughput signal (INDEPENDENT): `total_completed` rate.\n")
    lines.append(f"- **Run-level COLLAPSE label:** completed_orders < {COLLAPSE_MAX_ORDERS} "
                 f"AND deadlock_ratio_mean >= {COLLAPSE_MIN_DEADLOCK}. Precedence tests "
                 "(lead / cross-corr / Granger) run on collapsed runs only.\n")
    lines.append(f"- Within a collapsed run, the collapse-onset TICK is located where the "
                 f"completion rate < {COLLAPSE_RATE_FRAC}x early rate WHILE backlog >= "
                 f"{COLLAPSE_BACKLOG_MULT}x early backlog.\n")
    lines.append("- Precedence lead = collapse_tick - lock_onset_tick (positive => lock first).\n")
    for g in report["groups"]:
        lead = g["lead_collapse_minus_lock"]
        lines.append(f"\n## {g['load']} / {g['arm']}\n")
        lines.append(f"- seeds with data {g['n_seeds_with_data']}; "
                     f"**collapsed runs {g['n_collapsed_runs']}** "
                     f"(with onset tick {g['n_collapsed_with_onset_time']}); "
                     f"lock onset in collapsed {g['n_lock_onset_in_collapsed']}; "
                     f"with both {g['n_with_both']}\n")
        lines.append(f"- **lead** n={lead['n']} mean={lead['mean']} median={lead['median']} "
                     f"CI95={lead['ci95_mean']} sign+={lead['sign_positive']} "
                     f"Wilcoxon(lead>0) p={lead['wilcoxon_greater_p']}\n")
        xc = g["cross_correlation"]
        lines.append(f"- cross-corr: negative-peak seeds={xc['n_negative_peak']} "
                     f"median lead-lag={xc['median_lead_lag_samples']} samples "
                     f"(x{report['stride_hint']} ticks) peak r={xc['median_peak_corr']}\n")
        gr = g["granger"]
        lines.append(f"- Granger lock->rate: sig_frac={gr['lock_to_rate_sig_frac']} "
                     f"Fisher p={gr['lock_to_rate_fisher_p']}; "
                     f"rate->lock: sig_frac={gr['rate_to_lock_sig_frac']} "
                     f"Fisher p={gr['rate_to_lock_fisher_p']} (n={gr['n_tested']}, lag={gr['lag']})\n")
    lines.append("\n---\n*Caveats:* Granger = predictive precedence at stride-10 resolution, "
                 "not structural proof; precedence is only interpretable on collapse seeds; "
                 "lock and throughput signals are measured on independent subsystems "
                 "(station admission vs order completion), which removes the seed-556 circularity "
                 "but does not by itself establish a mechanism.\n")
    return "".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace-root", default=str(DEFAULT_TRACE_ROOT))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--loads", default="", help="comma-separated subset; default all present")
    ap.add_argument("--arms", default="", help="comma-separated subset; default all present")
    args = ap.parse_args()

    loads = tuple(x for x in args.loads.split(",") if x) or None
    arms = tuple(x for x in args.arms.split(",") if x) or None
    trace_root = Path(args.trace_root)

    groups, n_files = load_records(trace_root, loads, arms)
    if not groups:
        raise SystemExit(f"no usable trace records under {trace_root} "
                         f"(loads={loads}, arms={arms}); is the collection still running?")

    results = [analyze_group(recs, load, arm)
               for (load, arm), recs in sorted(groups.items())]
    report = {
        "trace_root": trace_root.as_posix(),
        "n_seed_files": n_files,
        "stride_hint": 10,
        "params": {
            "share_grid": SHARE_GRID, "primary_share": PRIMARY_SHARE,
            "sustain": SUSTAIN, "smooth_window": SMOOTH_WINDOW,
            "collapse_max_orders": COLLAPSE_MAX_ORDERS,
            "collapse_min_deadlock": COLLAPSE_MIN_DEADLOCK,
            "collapse_rate_frac": COLLAPSE_RATE_FRAC,
            "collapse_backlog_mult": COLLAPSE_BACKLOG_MULT,
            "granger_lag": GRANGER_LAG, "bootstrap_iters": BOOTSTRAP_ITERS,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "groups": results,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lock_precedence_multiseed.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "lock_precedence_multiseed.md").write_text(
        render_md(report), encoding="utf-8")

    # console
    print(f"# Lock precedence (multi-seed)  trace_root={trace_root.name}  files={n_files}\n")
    for g in results:
        lead = g["lead_collapse_minus_lock"]
        gr = g["granger"]
        print(f"## {g['load']}/{g['arm']}: data={g['n_seeds_with_data']} "
              f"collapsed={g['n_collapsed_runs']} "
              f"(onset_tick={g['n_collapsed_with_onset_time']}) both={g['n_with_both']}")
        print(f"   lead mean={lead['mean']} median={lead['median']} CI={lead['ci95_mean']} "
              f"sign+={lead['sign_positive']} wilcoxon_p={lead['wilcoxon_greater_p']}")
        print(f"   granger lock->rate Fisher p={gr['lock_to_rate_fisher_p']} "
              f"(sig {gr['lock_to_rate_sig_frac']}) | rate->lock Fisher p={gr['rate_to_lock_fisher_p']} "
              f"(sig {gr['rate_to_lock_sig_frac']})\n")
    print(f"[info] wrote {out_dir / 'lock_precedence_multiseed.json'}")
    print(f"[info] wrote {out_dir / 'lock_precedence_multiseed.md'}")


if __name__ == "__main__":
    main()
