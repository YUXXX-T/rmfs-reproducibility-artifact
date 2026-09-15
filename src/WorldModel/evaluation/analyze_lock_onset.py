"""Station-lock ONSET timing analysis (temporal precedence probe).

READ-ONLY.  Imports nothing from the model / simulator.

Question
--------
Does station-lock *onset* precede throughput collapse?  Temporal precedence
is the causal ingredient not covered by the aggregate mediation analysis in
``analyze_station_lock_causal.py``.

Data reality (READ THIS)
------------------------
The 50-seed PP factorial did NOT record per-tick traces
(``trace_record_count == 0``).  The only per-tick station trace on disk is the
seed-556 replay in ``fig04_data/station_trace_seed556_high.json`` (2 arms,
300 samples over 1500 ticks): ``occupancy / committed / rejections`` per
station, plus a single terminal ``completed_orders``.

CRITICAL CAVEAT: there is NO per-tick completion / throughput series.  The
available station signals (committed pile-up, occupancy collapse to one
station, rejection wall) are all *facets of the lock itself* -> using them as
the "collapse time" to prove "lock precedes collapse" is circular.  This
script therefore reports what the trace CAN support honestly:

  * lock-onset timing: when max-station share crosses a threshold grid, and
    how gradually it develops (a leading cause should develop over many ticks,
    not instantaneously);
  * a contrast with the non-locking arm (which never crosses);
  * a *labelled-as-weak* lead vs a rejection-wall proxy, with the circularity
    stated in the output.

A defensible multi-seed precedence test needs a re-run that logs, per tick,
an INDEPENDENT throughput signal (cumulative completions / backlog) alongside
a lock metric -> then Granger / cross-correlation / change-point ordering.
This script is structured so that, once such traces exist, pointing
``--trace-dir`` at them reuses the same onset detectors.

Run:
    python WorldModel/evaluation/analyze_lock_onset.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SEED556_TRACE = (
    REPO
    / "references" / "writing" / "figures" / "forth_icra"
    / "fig04_data" / "station_trace_seed556_high.json"
)
OUT_DIR = (
    REPO / "WorldModel" / "checkpoints" / "phaseC_wm_onpolicy_round1_v1"
    / "phasec_physical_only_pp_factorial_900_950_v1"
    / "station_lock_causal_analysis"
)

# Pre-registered thresholds.
SHARE_GRID = [0.6, 0.7, 0.8, 0.9]      # lock-onset share crossings
SUSTAIN = 3                            # samples the crossing must persist
SMOOTH_WINDOW = 12                     # centered rolling mean (matches fig04)


def rolling_mean(values: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    half = window // 2
    out = np.empty_like(values)
    for i in range(len(values)):
        out[i] = values[max(0, i - half): min(len(values), i + half + 1)].mean()
    return out


def max_share_series(committed: np.ndarray) -> np.ndarray:
    tot = committed.sum(axis=1)
    return np.divide(committed.max(axis=1), tot, out=np.zeros_like(tot), where=tot > 0)


def first_sustained_crossing(
    ticks: np.ndarray, series: np.ndarray, thresh: float, sustain: int
) -> float | None:
    """First tick where series >= thresh for `sustain` consecutive samples."""
    run = 0
    for i, v in enumerate(series):
        run = run + 1 if v >= thresh else 0
        if run >= sustain:
            return float(ticks[i - sustain + 1])
    return None


def analyze_arm(name: str, arm: dict) -> dict:
    ticks = np.asarray(arm["ticks"], dtype=float)
    committed = np.asarray(arm["committed"], dtype=float)
    occupancy = np.asarray(arm["occupancy"], dtype=float)
    rejections = np.asarray(arm["rejections"], dtype=float)  # cumulative

    share = rolling_mean(max_share_series(committed))
    # rejection-wall proxy: when cumulative capacity rejections reach a
    # fraction of their final value. LABELLED WEAK (facet of the lock).
    rej_total = rejections.sum(axis=1)
    final = rej_total[-1]
    rej_onset = {
        f"reach_{int(p*100)}pct": (
            float(ticks[int(np.argmax(rej_total >= p * final))])
            if final > 0 and (rej_total >= p * final).any() else None
        )
        for p in (0.25, 0.5)
    }
    # "effective serving stations" over time (occupancy>0 count) -- how fast
    # service concentrates onto a single station.
    active = (occupancy > 0).sum(axis=1)

    lock_onset = {
        f"share_ge_{t}": first_sustained_crossing(ticks, share, t, SUSTAIN)
        for t in SHARE_GRID
    }
    return {
        "completed_orders": arm["completed_orders"],
        "deadlock_ratio_mean": arm["deadlock_ratio_mean"],
        "final_max_share_smoothed": float(share[-1]),
        "late_mean_share_t>=750": float(share[ticks >= 750].mean()),
        "lock_onset_tick": lock_onset,
        "rejection_wall_proxy_tick": rej_onset,
        "min_active_stations": int(active.min()),
        "tick_active_stations_first_hits_1": (
            float(ticks[int(np.argmax(active == 1))]) if (active == 1).any() else None
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trace", default=str(SEED556_TRACE))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        raise SystemExit(f"trace not found: {trace_path}")
    data = json.loads(trace_path.read_text(encoding="utf-8"))

    arm_map = {"greedy_physical_only": "Greedy(locks)",
               "s1_j1_physical_only": "S1J1(healthy)"}
    result = {"source": trace_path.name, "seed": 556, "load": "high",
              "share_grid": SHARE_GRID, "sustain_samples": SUSTAIN,
              "caveat": ("single seed; no per-tick completion series; station "
                         "signals are facets of the lock -> precedence vs "
                         "throughput-collapse is NOT cleanly separable here."),
              "arms": {}}
    for key, label in arm_map.items():
        if key in data:
            result["arms"][label] = analyze_arm(label, data[key])

    # console report
    print(f"# Lock-onset timing  (source={trace_path.name}, seed 556, high)\n")
    print(result["caveat"], "\n")
    for label, r in result["arms"].items():
        print(f"## {label}: {int(r['completed_orders'])} orders, "
              f"deadlock {r['deadlock_ratio_mean']:.3f}, "
              f"late-share {r['late_mean_share_t>=750']:.3f}")
        print("   lock-onset tick (share sustained>=thr):",
              {k: v for k, v in r["lock_onset_tick"].items()})
        print("   rejection-wall proxy (WEAK, facet of lock):",
              r["rejection_wall_proxy_tick"])
        print("   first tick collapsed to 1 active station:",
              r["tick_active_stations_first_hits_1"], "\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lock_onset_seed556.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8")
    print(f"[info] wrote {out_dir / 'lock_onset_seed556.json'}")


if __name__ == "__main__":
    main()
