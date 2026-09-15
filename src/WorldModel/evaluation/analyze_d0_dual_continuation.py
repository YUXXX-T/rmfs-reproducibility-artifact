"""
D0 Dual-Continuation Differential Analysis (6.2 pre-diagnostic)
================================================================
Quantifies the pi_ref gap: same decision-point snapshots + same candidates,
labels generated once with Greedy continuation (label_G, the production
pipeline) and once with the frozen A1 WM arm (label_WM).

Registered spec: td_bootstrap_6p2_plan.md section D0.

Outputs:
  1. Per-channel (peak/cvar/terminal + derived combo) distributions of
     delta = label_WM - label_G  (signed bias + |delta|), paired t + sign test,
     stratified by danger slice (r_at_decision quantile across groups).
  2. Intra-group rank flips: top-1 flip rate and pairwise disagreement rate
     between label_G-ranking and label_WM-ranking, per channel.
  3. WM-arm rollout stats aggregate (model-path fraction, conversion activity)
     to document the cold-start semantics of the WM continuation.

2026-07-08 metric revision (first formal D0 run):
  - risk_combo added as a derived channel (0.2*peak+0.3*cvar+0.5*terminal,
    evaluate_long_risk weights): per-channel flip rates are tie-censored
    (many exact within-group ties), the deployed combo weighting is where
    reordering actually shows.
  - top1_flip (positional argmin) is tie-fragile; top1_strict counts a flip
    only when every member of G's min-set is strictly worse than the WM
    optimum, with top1_wm_regret as the realized suboptimality magnitude.
  - pairwise_strict_flip_rate conditions on both-side strict pairs;
    flip_margin_g reports |G-margin| of inverted pairs (near-tie check).

Usage:
  python -m WorldModel.evaluation.analyze_d0_dual_continuation \
    --greedy-labels d0/labels_greedy.pt \
    --wm-labels d0/labels_wm.pt \
    [--danger-quantile 0.7] [--report-json d0/d0_report.json]
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from WorldModel.core.long_risk_schema import (
    LONG_RISK_QUANTILE_COMBO_WEIGHTS,
)

try:
    from scipy import stats as sps
except ImportError:
    sps = None

CHANNELS = ["risk_peak", "risk_cvar", "risk_terminal", "risk_combo"]
COMBO_W = {
    "risk_peak": LONG_RISK_QUANTILE_COMBO_WEIGHTS["peak_q95"],
    "risk_cvar": LONG_RISK_QUANTILE_COMBO_WEIGHTS["cvar_q90"],
    "risk_terminal": LONG_RISK_QUANTILE_COMBO_WEIGHTS["terminal_q90"],
}


def _channel_value(label, channel):
    if channel == "risk_combo":
        return float(sum(w * float(label[k]) for k, w in COMBO_W.items()))
    return float(label[channel])


def _load_labels(path):
    labels = torch.load(path, weights_only=False)
    return {l["candidate_key"]: l for l in labels}


def _sign_test(diffs):
    """Two-sided sign test on nonzero diffs -> (n_pos, n_neg, p)."""
    nz = [d for d in diffs if d != 0.0]
    n_pos = sum(1 for d in nz if d > 0)
    n_neg = len(nz) - n_pos
    if not nz or sps is None:
        return n_pos, n_neg, None
    p = sps.binomtest(n_pos, len(nz), 0.5).pvalue
    return n_pos, n_neg, float(p)


def _paired_t(diffs):
    if sps is None or len(diffs) < 2 or np.std(diffs) == 0:
        return None, None
    t, p = sps.ttest_rel(diffs, np.zeros_like(diffs))
    return float(t), float(p)


def _describe_diffs(pairs, channel):
    """pairs: list of (label_G, label_WM) dicts."""
    d = np.array([_channel_value(w, channel) - _channel_value(g, channel)
                  for g, w in pairs])
    ad = np.abs(d)
    t, tp = _paired_t(d)
    n_pos, n_neg, sp = _sign_test(d)
    return {
        "n": len(d),
        "delta_mean": float(d.mean()) if len(d) else None,
        "delta_std": float(d.std()) if len(d) else None,
        "abs_mean": float(ad.mean()) if len(d) else None,
        "abs_p50": float(np.median(ad)) if len(d) else None,
        "abs_p90": float(np.quantile(ad, 0.9)) if len(d) else None,
        "abs_max": float(ad.max()) if len(d) else None,
        "frac_nonzero": float((ad > 1e-12).mean()) if len(d) else None,
        "t": t, "t_p": tp,
        "sign_pos": n_pos, "sign_neg": n_neg, "sign_p": sp,
    }


def _rank_flips(groups, channel):
    """groups: {gid: [(label_G, label_WM), ...]} with >=2 candidates.

    Lower risk = better. top1_flip: argmin under G vs under WM differs
    (positional, tie-fragile — kept for backward comparability).
    top1_strict: WM's argmin lies outside G's min-value tie set AND G's
    argmin lies outside WM's min-value tie set — a genuine choice change.
    top1_wm_regret: G-label regret of WM's chosen top-1 vs G's optimum.
    top1_g_regret_wm: WM-label regret of G's chosen top-1 vs WM's optimum
    — the deployment-relevant direction (head is trained on G labels but
    deployed where the future is WM-continued).
    pairwise: strict order inversions; _strict variant conditions the
    denominator on pairs strictly ordered on BOTH sides. flip_margin_g:
    |G-side margin| of inverted pairs (near-tie diagnostics).
    """
    top1_flips, top1_strict, top1_total = 0, 0, 0
    top1_regret, top1_regret_wm = [], []
    pair_flips, pair_total = 0, 0
    strict_pairs = 0
    flip_margins = []
    for gid, pairs in groups.items():
        if len(pairs) < 2:
            continue
        vg = np.array([_channel_value(g, channel) for g, _ in pairs])
        vw = np.array([_channel_value(w, channel) for _, w in pairs])
        top1_total += 1
        ig, iw = int(np.argmin(vg)), int(np.argmin(vw))
        if ig != iw:
            top1_flips += 1
        g_min_set = set(np.flatnonzero(vg == vg.min()).tolist())
        w_min_set = set(np.flatnonzero(vw == vw.min()).tolist())
        if iw not in g_min_set and ig not in w_min_set:
            top1_strict += 1
        top1_regret.append(float(vg[iw] - vg.min()))
        top1_regret_wm.append(float(vw[ig] - vw.min()))
        for i in range(len(pairs)):
            for j in range(i + 1, len(pairs)):
                sg = np.sign(vg[i] - vg[j])
                sw = np.sign(vw[i] - vw[j])
                pair_total += 1
                if sg != 0 and sw != 0:
                    strict_pairs += 1
                    if sg != sw:
                        pair_flips += 1
                        flip_margins.append(float(abs(vg[i] - vg[j])))
    return {
        "groups": top1_total,
        "top1_flip_rate": top1_flips / top1_total if top1_total else None,
        "top1_flips": top1_flips,
        "top1_strict_flip_rate": (top1_strict / top1_total
                                  if top1_total else None),
        "top1_strict_flips": top1_strict,
        "top1_wm_regret_mean": (float(np.mean(top1_regret))
                                if top1_regret else None),
        "top1_wm_regret_max": (float(np.max(top1_regret))
                               if top1_regret else None),
        "top1_g_regret_wm_mean": (float(np.mean(top1_regret_wm))
                                  if top1_regret_wm else None),
        "top1_g_regret_wm_max": (float(np.max(top1_regret_wm))
                                 if top1_regret_wm else None),
        "pairs": pair_total,
        "strict_pairs": strict_pairs,
        "pairwise_flip_rate": pair_flips / pair_total if pair_total else None,
        "pairwise_strict_flip_rate": (pair_flips / strict_pairs
                                      if strict_pairs else None),
        "pairwise_flips": pair_flips,
        "flip_margin_g_mean": (float(np.mean(flip_margins))
                               if flip_margins else None),
        "flip_margin_g_max": (float(np.max(flip_margins))
                              if flip_margins else None),
    }


def _fmt(v, spec=".4f"):
    return "None" if v is None else format(v, spec)


def main():
    ap = argparse.ArgumentParser(
        description="D0 dual-continuation differential analysis (pi_ref gap)")
    ap.add_argument("--greedy-labels", required=True)
    ap.add_argument("--wm-labels", required=True)
    ap.add_argument("--danger-quantile", type=float, default=0.7,
                    help="Group r_at_decision quantile threshold for the "
                         "danger slice (default: 0.7)")
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    G = _load_labels(args.greedy_labels)
    W = _load_labels(args.wm_labels)

    common = sorted(set(G) & set(W))
    only_g, only_w = len(G) - len(common), len(W) - len(common)
    pairs_all = [(G[k], W[k]) for k in common]

    # Group index + danger slice via group-level r_at_decision
    groups_all = defaultdict(list)
    group_risk = {}
    for g, w in pairs_all:
        gid = g["candidate_group_id"]
        groups_all[gid].append((g, w))
        group_risk[gid] = g["r_at_decision"]

    thr = float(np.quantile(list(group_risk.values()), args.danger_quantile))
    danger_gids = {gid for gid, r in group_risk.items() if r >= thr}

    def _slice(pred):
        p = [(g, w) for g, w in pairs_all
             if pred(g["candidate_group_id"])]
        gr = {gid: v for gid, v in groups_all.items() if pred(gid)}
        return p, gr

    slices = {
        "all": _slice(lambda gid: True),
        "danger": _slice(lambda gid: gid in danger_gids),
        "calm": _slice(lambda gid: gid not in danger_gids),
    }

    print("D0 Dual-Continuation Differential Analysis")
    print(f"  greedy labels : {args.greedy_labels}  ({len(G)} labels)")
    print(f"  wm labels     : {args.wm_labels}  ({len(W)} labels)")
    print(f"  matched pairs : {len(common)}  "
          f"(greedy-only: {only_g}, wm-only: {only_w})")
    print(f"  groups        : {len(groups_all)}  |  danger slice: "
          f"r_at_decision >= q{args.danger_quantile:.2f} = {thr:.4f}  "
          f"({len(danger_gids)} groups)")

    report = {
        "matched_pairs": len(common),
        "groups": len(groups_all),
        "danger_quantile": args.danger_quantile,
        "danger_threshold": thr,
        "danger_groups": len(danger_gids),
        "channels": {},
    }

    # Event-rate side stat
    ev_g = np.mean([g["risk_event"] for g, _ in pairs_all]) if pairs_all else 0
    ev_w = np.mean([w["risk_event"] for _, w in pairs_all]) if pairs_all else 0
    ev_disagree = np.mean([g["risk_event"] != w["risk_event"]
                           for g, w in pairs_all]) if pairs_all else 0
    print(f"\n  risk_event rate: G={ev_g:.4f}  WM={ev_w:.4f}  "
          f"disagree={ev_disagree:.4f}")
    report["risk_event"] = {"rate_g": float(ev_g), "rate_wm": float(ev_w),
                            "disagree_rate": float(ev_disagree)}

    for ch in CHANNELS:
        print(f"\n=== {ch} ===")
        report["channels"][ch] = {}
        for sname, (spairs, sgroups) in slices.items():
            d = _describe_diffs(spairs, ch)
            f = _rank_flips(sgroups, ch)
            report["channels"][ch][sname] = {"diff": d, "flips": f}
            print(f"  [{sname:6s}] n={d['n']:4d}  "
                  f"delta(WM-G) mean={_fmt(d['delta_mean'])} "
                  f"std={_fmt(d['delta_std'])}  "
                  f"|d| mean={_fmt(d['abs_mean'])} "
                  f"p50={_fmt(d['abs_p50'])} p90={_fmt(d['abs_p90'])} "
                  f"max={_fmt(d['abs_max'])}  "
                  f"nonzero={_fmt(d['frac_nonzero'], '.2%')}")
            print(f"           paired t={_fmt(d['t'], '.3f')} "
                  f"(p={_fmt(d['t_p'], '.4f')})  "
                  f"sign +{d['sign_pos']}/-{d['sign_neg']} "
                  f"(p={_fmt(d['sign_p'], '.4f')})")
            print(f"           flips: top-1 {f['top1_flips']}/{f['groups']} "
                  f"({_fmt(f['top1_flip_rate'], '.2%')})  "
                  f"strict {f['top1_strict_flips']}/{f['groups']} "
                  f"({_fmt(f['top1_strict_flip_rate'], '.2%')})  "
                  f"wm-top1 G-regret mean={_fmt(f['top1_wm_regret_mean'])} "
                  f"max={_fmt(f['top1_wm_regret_max'])}  "
                  f"g-top1 WM-regret mean={_fmt(f['top1_g_regret_wm_mean'])} "
                  f"max={_fmt(f['top1_g_regret_wm_max'])}")
            print(f"           pairwise {f['pairwise_flips']}/{f['pairs']} "
                  f"({_fmt(f['pairwise_flip_rate'], '.2%')})  "
                  f"strict-pairs {f['pairwise_flips']}/{f['strict_pairs']} "
                  f"({_fmt(f['pairwise_strict_flip_rate'], '.2%')})  "
                  f"flip |G-margin| mean={_fmt(f['flip_margin_g_mean'])} "
                  f"max={_fmt(f['flip_margin_g_max'])}")

    # WM rollout stats aggregate (cold-start semantics documentation)
    wm_stats = [w.get("wm_rollout_stats") for _, w in pairs_all
                if w.get("wm_rollout_stats")]
    if wm_stats:
        agg = {k: float(np.mean([s[k] for s in wm_stats]))
               for k in wm_stats[0]}
        model_frac = (agg["model_assign_calls"] / agg["assign_calls"]
                      if agg["assign_calls"] else 0.0)
        print(f"\n=== WM continuation rollout stats (mean per rollout) ===")
        print(f"  assign_calls={agg['assign_calls']:.1f}  "
              f"model_path={agg['model_assign_calls']:.1f} "
              f"({model_frac:.1%})  "
              f"greedy_fallback={agg['fallback_greedy_calls']:.1f}")
        print(f"  conv contexts={agg['energy_conv_contexts']:.1f}  "
              f"warmup={agg['energy_conv_warmup_contexts']:.1f}  "
              f"active={agg['energy_conv_active_contexts']:.1f}  "
              f"modified={agg['energy_conv_modified_decisions']:.2f}")
        report["wm_rollout_stats_mean"] = agg
        report["wm_model_path_fraction"] = model_frac

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        print(f"\n  Report saved: {args.report_json}")


if __name__ == "__main__":
    main()
