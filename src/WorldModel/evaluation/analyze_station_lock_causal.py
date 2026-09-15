"""Multi-seed statistical / causal analysis of station-lock.

READ-ONLY.  This script does not import or modify any model / simulator code.
It only reads the frozen 50-seed PP factorial results

    WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/
        phasec_physical_only_pp_factorial_900_950_v1/per_seed/<load>_seed<NN>.json

and writes an analysis bundle (JSON + Markdown) next to them under
``station_lock_causal_analysis/``.

Goal
----
Support (or falsify) the paper claim that the throughput gain of the
world-model arms comes mainly from *avoiding station lock*.  We do this
with existing data only:

  1. Per-run station-lock metrics (several variants; we report which
     one actually discriminates rather than assuming ``granted``-share).
  2. Multi-seed distributions + paired comparisons (same seed across arms)
     with bootstrap CIs and Wilcoxon signed-rank tests.
  3. Collapse-threshold sensitivity: sweep a throughput cutoff, report
     fraction-collapsed per arm and lock metrics in collapsed vs
     non-collapsed subsets (means + bootstrap CI).
  4. A simple linear mediation analysis (Combo vs Greedy): does the
     station-lock metric mediate the arm -> throughput effect?

Everything below the ``PRE-REGISTRATION`` block is fixed before looking at
aggregates; the collapse cutoff is swept over a pre-declared grid so no
single threshold is cherry-picked.

Run:
    python -m WorldModel.evaluation.analyze_station_lock_causal
    python -m WorldModel.evaluation.analyze_station_lock_causal --loads high
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Callable

import numpy as np

try:  # scipy is optional; we fall back to numpy implementations.
    from scipy import stats as _scipy_stats  # type: ignore
except Exception:  # pragma: no cover - env without scipy
    _scipy_stats = None


# --------------------------------------------------------------------------- #
# PRE-REGISTRATION (fixed before inspecting aggregate outcomes)               #
# --------------------------------------------------------------------------- #
REPO = Path(__file__).resolve().parents[2]
FACTORIAL = (
    REPO
    / "WorldModel"
    / "checkpoints"
    / "phaseC_wm_onpolicy_round1_v1"
    / "phasec_physical_only_pp_factorial_900_950_v1"
)
PER_SEED_DIR = FACTORIAL / "per_seed"
OUT_DIR = FACTORIAL / "station_lock_causal_analysis"

ARMS = ["Greedy", "Hungarian", "JSQ", "PhaseC", "ComboS1J1"]
LOADS = ["low", "mid", "high"]
SEEDS = list(range(900, 950))

# Arms whose contrast we care about. JSQ is the balance-by-design natural
# comparator; ComboS1J1 is the proposed method; Greedy is the unbalanced ref.
PRIMARY_TREATMENT = "ComboS1J1"
BALANCED_COMPARATOR = "JSQ"
CONTROL = "Greedy"

# Collapse is a *low-throughput* run. We sweep an absolute completed-orders
# cutoff over this grid (informed only by the pooled quantile range, not by
# per-arm outcomes). A run is "collapsed" if completed_orders <= cutoff.
COLLAPSE_CUTOFF_GRID = [100, 125, 150, 175, 200, 250, 300]

BOOTSTRAP_ITERS = 10_000
BOOTSTRAP_SEED = 20260909
CI_ALPHA = 0.05  # 95% CI


# --------------------------------------------------------------------------- #
# Data extraction                                                             #
# --------------------------------------------------------------------------- #
@dataclass
class RunRecord:
    load: str
    seed: int
    arm: str
    completed_orders: float
    deadlock_ratio_mean: float
    deadlock_ratio_max: float
    station_pressure: float
    bottleneck_cvar: float
    # per-station cumulative counters
    granted: list[int] = field(default_factory=list)
    attempts: list[int] = field(default_factory=list)
    rejected: list[int] = field(default_factory=list)
    committed_load: list[int] = field(default_factory=list)
    entry_occupied: list[int] = field(default_factory=list)
    # derived lock metrics (filled in later)
    lock: dict[str, float] = field(default_factory=dict)


def _share(values: list[int]) -> float:
    v = np.asarray(values, dtype=float)
    tot = v.sum()
    return float(v.max() / tot) if tot > 0 else float("nan")


def _evenness(values: list[int]) -> float:
    """Normalized Shannon entropy (0 = one station, 1 = perfectly even)."""
    v = np.asarray(values, dtype=float)
    tot = v.sum()
    if tot <= 0:
        return float("nan")
    p = v[v > 0] / tot
    ent = float(-(p * np.log(p)).sum())
    n = len(values)
    return ent / math.log(n) if n > 1 else 0.0


def _hhi(values: list[int]) -> float:
    """Herfindahl-Hirschman concentration index in [1/n, 1]."""
    v = np.asarray(values, dtype=float)
    tot = v.sum()
    if tot <= 0:
        return float("nan")
    p = v / tot
    return float((p * p).sum())


def _lock_metrics(rec: RunRecord) -> dict[str, float]:
    return {
        # demand-concentration metrics (strong lock signal)
        "share_attempts": _share(rec.attempts),
        "share_rejected": _share(rec.rejected),
        "hhi_attempts": _hhi(rec.attempts),
        "evenness_attempts": _evenness(rec.attempts),
        # service-concentration metrics (capacity-bounded, weaker)
        "share_granted": _share(rec.granted),
        "evenness_granted": _evenness(rec.granted),
        # endpoint snapshot (degenerate; kept for comparison only)
        "share_committed_endpoint": _share(rec.committed_load),
        # absolute failed-entry volume
        "failed_entries": float(
            sum(rec.rejected) + sum(rec.entry_occupied)
        ),
    }


def load_runs(loads: list[str]) -> list[RunRecord]:
    runs: list[RunRecord] = []
    missing: list[str] = []
    for load in loads:
        for seed in SEEDS:
            path = PER_SEED_DIR / f"{load}_seed{seed}.json"
            if not path.exists():
                missing.append(path.name)
                continue
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            for arm in ARMS:
                a = payload["arms"][arm]
                m = a["metrics"]
                smf = a["station_audit"]["station_metrics_final"]
                rec = RunRecord(
                    load=load,
                    seed=seed,
                    arm=arm,
                    completed_orders=float(m["completed_orders"]),
                    deadlock_ratio_mean=float(m.get("deadlock_ratio_mean", float("nan"))),
                    deadlock_ratio_max=float(m.get("deadlock_ratio_max", float("nan"))),
                    station_pressure=float(m.get("station_pressure", float("nan"))),
                    bottleneck_cvar=float(m.get("bottleneck_CVaR", float("nan"))),
                    granted=[int(s["granted"]) for s in smf],
                    attempts=[int(s["attempts"]) for s in smf],
                    rejected=[int(s["rejected_capacity"]) for s in smf],
                    committed_load=[int(s["committed_load"]) for s in smf],
                    entry_occupied=[int(s["rollbacks"]["entry_occupied"]) for s in smf],
                )
                rec.lock = _lock_metrics(rec)
                runs.append(rec)
    if missing:
        print(f"[warn] {len(missing)} missing per-seed files: {missing[:5]} ...")
    return runs


# --------------------------------------------------------------------------- #
# Statistics helpers                                                          #
# --------------------------------------------------------------------------- #
def spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return float("nan"), float("nan")
    if _scipy_stats is not None:
        r = _scipy_stats.spearmanr(x, y)
        return float(r.statistic), float(r.pvalue)
    # numpy fallback: Pearson on ranks, no p-value.
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1]), float("nan")


def wilcoxon_signed_rank(delta: np.ndarray) -> float:
    d = delta[np.isfinite(delta)]
    d = d[d != 0]
    if len(d) < 1:
        return float("nan")
    if _scipy_stats is not None:
        try:
            return float(_scipy_stats.wilcoxon(d).pvalue)
        except ValueError:
            return float("nan")
    return float("nan")


def paired_bootstrap_ci(
    delta: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    d = delta[np.isfinite(delta)]
    if len(d) == 0:
        return float("nan"), float("nan"), float("nan")
    boot = np.array(
        [rng.choice(d, size=len(d), replace=True).mean() for _ in range(BOOTSTRAP_ITERS)]
    )
    lo, hi = np.quantile(boot, [CI_ALPHA / 2, 1 - CI_ALPHA / 2])
    return float(d.mean()), float(lo), float(hi)


def mean_bootstrap_ci(
    x: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float, int]:
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan"), 0
    boot = np.array(
        [rng.choice(x, size=len(x), replace=True).mean() for _ in range(BOOTSTRAP_ITERS)]
    )
    lo, hi = np.quantile(boot, [CI_ALPHA / 2, 1 - CI_ALPHA / 2])
    return float(x.mean()), float(lo), float(hi), int(len(x))


# --------------------------------------------------------------------------- #
# Analyses                                                                    #
# --------------------------------------------------------------------------- #
LOCK_METRIC_KEYS = [
    "share_attempts",
    "share_rejected",
    "hhi_attempts",
    "evenness_attempts",
    "share_granted",
    "evenness_granted",
    "share_committed_endpoint",
    "failed_entries",
]


def _by(runs: list[RunRecord], load: str, arm: str) -> dict[int, RunRecord]:
    return {r.seed: r for r in runs if r.load == load and r.arm == arm}


def analyze_correlations(runs: list[RunRecord], load: str) -> dict:
    """Which lock metric discriminates throughput? Spearman across all arms."""
    sub = [r for r in runs if r.load == load]
    tp = np.array([r.completed_orders for r in sub])
    out = {}
    for key in LOCK_METRIC_KEYS:
        lm = np.array([r.lock[key] for r in sub])
        rho, p = spearman(lm, tp)
        out[key] = {"spearman_rho_vs_throughput": rho, "p_value": p, "n": int(len(sub))}
    return out


def analyze_paired(runs: list[RunRecord], load: str, rng: np.random.Generator) -> dict:
    """Paired (same-seed) contrasts vs the control arm."""
    control = _by(runs, load, CONTROL)
    out = {}
    for treat in (PRIMARY_TREATMENT, BALANCED_COMPARATOR):
        tre = _by(runs, load, treat)
        seeds = sorted(set(control) & set(tre))
        # throughput delta
        d_tp = np.array([tre[s].completed_orders - control[s].completed_orders for s in seeds])
        mean_tp, lo_tp, hi_tp = paired_bootstrap_ci(d_tp, rng)
        p_tp = wilcoxon_signed_rank(d_tp)
        entry = {
            "n_pairs": len(seeds),
            "throughput_delta": {
                "mean": mean_tp, "ci95": [lo_tp, hi_tp], "wilcoxon_p": p_tp,
                "n_treatment_wins": int((d_tp > 0).sum()),
                "n_control_wins": int((d_tp < 0).sum()),
            },
            "lock_delta": {},
        }
        for key in ("share_attempts", "share_rejected", "evenness_attempts", "failed_entries"):
            d_lk = np.array([tre[s].lock[key] - control[s].lock[key] for s in seeds])
            m, lo, hi = paired_bootstrap_ci(d_lk, rng)
            entry["lock_delta"][key] = {
                "mean": m, "ci95": [lo, hi], "wilcoxon_p": wilcoxon_signed_rank(d_lk),
            }
        out[f"{treat}_minus_{CONTROL}"] = entry
    return out


def analyze_collapse_sensitivity(
    runs: list[RunRecord], load: str, rng: np.random.Generator
) -> dict:
    """Sweep the collapse cutoff; report fraction-collapsed per arm and lock
    metrics in collapsed vs non-collapsed subsets (means + bootstrap CI)."""
    sub = [r for r in runs if r.load == load]
    out = {"grid": COLLAPSE_CUTOFF_GRID, "per_cutoff": {}}
    for cutoff in COLLAPSE_CUTOFF_GRID:
        frac = {}
        for arm in ARMS:
            arm_runs = [r for r in sub if r.arm == arm]
            collapsed = [r for r in arm_runs if r.completed_orders <= cutoff]
            frac[arm] = len(collapsed) / len(arm_runs) if arm_runs else float("nan")
        # subset lock means pooled across arms
        collapsed = [r for r in sub if r.completed_orders <= cutoff]
        healthy = [r for r in sub if r.completed_orders > cutoff]
        subset_lock = {}
        for key in ("share_attempts", "share_rejected", "evenness_attempts", "failed_entries"):
            c_mean, c_lo, c_hi, c_n = mean_bootstrap_ci(
                np.array([r.lock[key] for r in collapsed]), rng
            )
            h_mean, h_lo, h_hi, h_n = mean_bootstrap_ci(
                np.array([r.lock[key] for r in healthy]), rng
            )
            subset_lock[key] = {
                "collapsed": {"mean": c_mean, "ci95": [c_lo, c_hi], "n": c_n},
                "non_collapsed": {"mean": h_mean, "ci95": [h_lo, h_hi], "n": h_n},
            }
        out["per_cutoff"][cutoff] = {
            "fraction_collapsed_by_arm": frac,
            "subset_lock_means": subset_lock,
        }
    return out


def analyze_mediation(runs: list[RunRecord], load: str, rng: np.random.Generator) -> dict:
    """Linear mediation for the Combo-vs-Greedy contrast, mediator = lock.

    total effect c   : throughput ~ T
    a-path           : lock      ~ T
    b-path & c'      : throughput ~ T + lock
    indirect = a*b ; proportion mediated = (c - c') / c
    Seeds are resampled (paired) to get a bootstrap CI on the indirect effect.
    """
    control = _by(runs, load, CONTROL)
    treat = _by(runs, load, PRIMARY_TREATMENT)
    seeds = sorted(set(control) & set(treat))
    out = {}
    for key in ("share_attempts", "share_rejected", "evenness_attempts"):
        # Stack the two arms into a regression frame.
        def build(seed_list):
            T, M, Y = [], [], []
            for s in seed_list:
                for arm_rec, t in ((control[s], 0.0), (treat[s], 1.0)):
                    lk = arm_rec.lock[key]
                    if not (math.isfinite(lk) and math.isfinite(arm_rec.completed_orders)):
                        continue
                    T.append(t); M.append(lk); Y.append(arm_rec.completed_orders)
            return np.array(T), np.array(M), np.array(Y)

        def fit(seed_list):
            T, M, Y = build(seed_list)
            if len(T) < 4 or np.ptp(T) == 0:
                return None
            # c: Y ~ T
            c = np.polyfit(T, Y, 1)[0]
            # a: M ~ T
            a = np.polyfit(T, M, 1)[0]
            # b, c': Y ~ [T, M]  via least squares
            X = np.column_stack([np.ones_like(T), T, M])
            coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
            cprime, b = coef[1], coef[2]
            indirect = a * b
            prop = indirect / c if c != 0 else float("nan")
            return {"c": c, "a": a, "b": b, "cprime": cprime,
                    "indirect": indirect, "prop_mediated": prop}

        point = fit(seeds)
        if point is None:
            out[key] = {"note": "insufficient variation"}
            continue
        boot_ind, boot_prop = [], []
        for _ in range(BOOTSTRAP_ITERS // 5):  # mediation bootstrap is heavier
            rs = rng.choice(seeds, size=len(seeds), replace=True)
            f = fit(list(rs))
            if f is not None:
                boot_ind.append(f["indirect"])
                boot_prop.append(f["prop_mediated"])
        ci = lambda arr: (
            [float(np.quantile(arr, CI_ALPHA / 2)), float(np.quantile(arr, 1 - CI_ALPHA / 2))]
            if arr else [float("nan"), float("nan")]
        )
        out[key] = {
            **{k: float(v) for k, v in point.items()},
            "indirect_ci95": ci(boot_ind),
            "prop_mediated_ci95": ci(boot_prop),
        }
    return out


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #
def build_report(results: dict) -> str:
    lines = ["# Station-lock causal analysis (offline, 50-seed PP factorial)\n"]
    lines.append(f"- Source: `{FACTORIAL.name}`  seeds {SEEDS[0]}-{SEEDS[-1]}\n")
    lines.append(f"- Bootstrap: {BOOTSTRAP_ITERS} iters, seed {BOOTSTRAP_SEED}, 95% CI\n")
    lines.append(
        "- Arms: control=`Greedy`, balanced comparator=`JSQ`, "
        f"treatment=`{PRIMARY_TREATMENT}`\n"
    )
    for load in results["loads"]:
        R = results["by_load"][load]
        lines.append(f"\n## Load: {load}\n")
        lines.append("### 1. Which lock metric discriminates throughput? (Spearman rho)\n")
        lines.append("| lock metric | rho vs throughput | p |")
        lines.append("|---|---|---|")
        for k, v in R["correlations"].items():
            lines.append(f"| {k} | {v['spearman_rho_vs_throughput']:+.3f} | {v['p_value']:.2e} |")
        lines.append("\n### 2. Paired contrasts vs Greedy (same seed)\n")
        for contrast, e in R["paired"].items():
            tp = e["throughput_delta"]
            lines.append(
                f"- **{contrast}** (n={e['n_pairs']}): throughput "
                f"{tp['mean']:+.1f} orders CI[{tp['ci95'][0]:+.1f},{tp['ci95'][1]:+.1f}] "
                f"(wins {tp['n_treatment_wins']}/{e['n_pairs']}, Wilcoxon p={tp['wilcoxon_p']:.2e})"
            )
            for k, lk in e["lock_delta"].items():
                lines.append(
                    f"    - {k}: {lk['mean']:+.3f} CI[{lk['ci95'][0]:+.3f},{lk['ci95'][1]:+.3f}]"
                )
        lines.append("\n### 3. Collapse-threshold sensitivity\n")
        lines.append("Fraction collapsed (completed_orders <= cutoff):\n")
        lines.append("| cutoff | " + " | ".join(ARMS) + " |")
        lines.append("|" + "---|" * (len(ARMS) + 1))
        for cutoff, cell in R["collapse"]["per_cutoff"].items():
            frac = cell["fraction_collapsed_by_arm"]
            lines.append(
                f"| {cutoff} | " + " | ".join(f"{frac[a]:.2f}" for a in ARMS) + " |"
            )
        mid = COLLAPSE_CUTOFF_GRID[len(COLLAPSE_CUTOFF_GRID) // 2]
        sl = R["collapse"]["per_cutoff"][mid]["subset_lock_means"]
        lines.append(f"\nLock means, collapsed vs non-collapsed (cutoff={mid}):\n")
        lines.append("| metric | collapsed (CI) | non-collapsed (CI) |")
        lines.append("|---|---|---|")
        for k, v in sl.items():
            c, h = v["collapsed"], v["non_collapsed"]
            lines.append(
                f"| {k} | {c['mean']:.3f} [{c['ci95'][0]:.3f},{c['ci95'][1]:.3f}] n={c['n']} "
                f"| {h['mean']:.3f} [{h['ci95'][0]:.3f},{h['ci95'][1]:.3f}] n={h['n']} |"
            )
        lines.append("\n### 4. Mediation (Combo - Greedy), mediator = lock\n")
        for k, v in R["mediation"].items():
            if "note" in v:
                lines.append(f"- {k}: {v['note']}")
                continue
            lines.append(
                f"- **{k}**: total c={v['c']:+.1f}, direct c'={v['cprime']:+.1f}, "
                f"indirect a*b={v['indirect']:+.1f} "
                f"CI[{v['indirect_ci95'][0]:+.1f},{v['indirect_ci95'][1]:+.1f}], "
                f"prop mediated={v['prop_mediated']:+.2f} "
                f"CI[{v['prop_mediated_ci95'][0]:+.2f},{v['prop_mediated_ci95'][1]:+.2f}]"
            )
    lines.append(
        "\n---\n*Interpretation guide:* the causal claim is supported if "
        "(i) a demand-concentration lock metric correlates strongly with "
        "throughput, (ii) the treatment reliably reduces lock and raises "
        "throughput on the same seeds, (iii) collapsed runs carry much higher "
        "lock than healthy runs across the whole cutoff grid, and (iv) a large, "
        "CI-excluding-zero fraction of the arm effect is mediated by lock.\n"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loads", nargs="+", default=LOADS, choices=LOADS)
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    if not PER_SEED_DIR.exists():
        raise SystemExit(f"per-seed dir not found: {PER_SEED_DIR}")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    runs = load_runs(args.loads)
    print(f"[info] loaded {len(runs)} runs across loads={args.loads}")

    results = {
        "source": FACTORIAL.name,
        "seeds": [SEEDS[0], SEEDS[-1]],
        "loads": args.loads,
        "config": {
            "bootstrap_iters": BOOTSTRAP_ITERS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "ci_alpha": CI_ALPHA,
            "collapse_cutoff_grid": COLLAPSE_CUTOFF_GRID,
            "control": CONTROL,
            "treatment": PRIMARY_TREATMENT,
            "balanced_comparator": BALANCED_COMPARATOR,
        },
        "by_load": {},
    }
    for load in args.loads:
        results["by_load"][load] = {
            "correlations": analyze_correlations(runs, load),
            "paired": analyze_paired(runs, load, rng),
            "collapse": analyze_collapse_sensitivity(runs, load, rng),
            "mediation": analyze_mediation(runs, load, rng),
        }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "station_lock_causal_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (out_dir / "station_lock_causal_report.md").write_text(
        build_report(results), encoding="utf-8"
    )
    print(f"[info] wrote {out_dir / 'station_lock_causal_results.json'}")
    print(f"[info] wrote {out_dir / 'station_lock_causal_report.md'}")


if __name__ == "__main__":
    main()
