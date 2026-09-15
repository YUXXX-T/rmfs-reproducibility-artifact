"""D0 three-batch merged analysis — paper-grade aggregation (pi_ref gap).

Merges the mid/high/low seed43 dual-continuation batches into the
registered paper readout (td_bootstrap_6p2_plan.md §4 D0, 2026-07-08
revision discipline):

  - PRIMARY evidence: paired sign test on delta = label_WM - label_G and
    strict top-1 / strict-pair flip rates (tie-robust);
  - regret magnitudes are reported MERGED ONLY, with leave-one-group-out
    (LOGO) sensitivity — mid43's standalone 9x asymmetry must never be
    cited alone (did not replicate in high43; single-outlier-driven);
  - danger slice uses the per-batch group r_at_decision q0.70 threshold
    (all three batches saturate at 1.0, so per-batch == pooled here; the
    saturation itself is the registered systematic proxy limitation).

Usage:
  python -m WorldModel.evaluation.analyze_d0_merged \
    --batch mid  DataGen/wm_data/d0/labels_greedy.pt        DataGen/wm_data/d0/labels_wm.pt \
    --batch high DataGen/wm_data/d0/labels_greedy_high43.pt DataGen/wm_data/d0/labels_wm_high43.pt \
    --batch low  DataGen/wm_data/d0/labels_greedy_low43.pt  DataGen/wm_data/d0/labels_wm_low43.pt \
    --report-json DataGen/wm_data/d0/d0_report_merged.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np

from WorldModel.evaluation.analyze_d0_dual_continuation import (
    CHANNELS, _channel_value, _describe_diffs, _load_labels, _rank_flips,
    _fmt,
)


def _group_regrets(pairs, channel):
    """Both regret directions for one group (list of (g,w) label pairs).

    deploy = WM-label regret of G's argmin (head trained on G labels,
    deployed where the future is WM-continued);
    reverse = G-label regret of WM's argmin."""
    vg = np.array([_channel_value(g, channel) for g, _ in pairs])
    vw = np.array([_channel_value(w, channel) for _, w in pairs])
    ig, iw = int(np.argmin(vg)), int(np.argmin(vw))
    return float(vw[ig] - vw.min()), float(vg[iw] - vg.min())


def _logo_ratio(deploy, reverse):
    """Leave-one-group-out range of ratio mean(deploy)/mean(reverse)."""
    deploy, reverse = np.asarray(deploy), np.asarray(reverse)
    n = len(deploy)

    def _ratio(d, r):
        mr = r.mean() if len(r) else 0.0
        return float(d.mean() / mr) if mr > 0 else None

    full = _ratio(deploy, reverse)
    ratios = []
    for i in range(n):
        m = np.ones(n, bool)
        m[i] = False
        ratios.append(_ratio(deploy[m], reverse[m]))
    valid = [r for r in ratios if r is not None]
    return {
        "ratio_full": full,
        "ratio_logo_min": min(valid) if valid else None,
        "ratio_logo_max": max(valid) if valid else None,
        "n_logo_undefined": sum(1 for r in ratios if r is None),
    }


def main():
    ap = argparse.ArgumentParser(
        description="D0 merged three-batch analysis (paper readout)")
    ap.add_argument("--batch", nargs=3, action="append", required=True,
                    metavar=("NAME", "GREEDY_PT", "WM_PT"))
    ap.add_argument("--danger-quantile", type=float, default=0.7)
    ap.add_argument("--report-json", default=None)
    args = ap.parse_args()

    # ---- load all batches, prefix group ids to avoid collisions ---------
    pairs_all = []          # (batch, gid, label_G, label_WM)
    batch_groups = {}       # batch -> {gid: [(g, w), ...]}
    batch_danger = {}       # batch -> set(gid), per-batch q threshold
    for name, gp, wp in args.batch:
        G, W = _load_labels(gp), _load_labels(wp)
        common = sorted(set(G) & set(W))
        groups = defaultdict(list)
        grisk = {}
        for k in common:
            g, w = G[k], W[k]
            gid = f"{name}:{g['candidate_group_id']}"
            groups[gid].append((g, w))
            grisk[gid] = g["r_at_decision"]
            pairs_all.append((name, gid, g, w))
        thr = float(np.quantile(list(grisk.values()), args.danger_quantile))
        batch_groups[name] = dict(groups)
        batch_danger[name] = {gid for gid, r in grisk.items() if r >= thr}
        print(f"  batch {name:5s}: {len(common)} pairs, {len(groups)} groups, "
              f"danger thr(q{args.danger_quantile:.2f})={thr:.4f} "
              f"({len(batch_danger[name])} groups)")

    all_groups = {}
    danger_gids = set()
    for name in batch_groups:
        all_groups.update(batch_groups[name])
        danger_gids |= batch_danger[name]
    n_groups = len(all_groups)
    print(f"  POOLED: {len(pairs_all)} pairs, {n_groups} groups, "
          f"{len(danger_gids)} danger groups")

    report = {
        "batches": {name: {"pairs": sum(len(v) for v in batch_groups[name].values()),
                           "groups": len(batch_groups[name]),
                           "danger_groups": len(batch_danger[name])}
                    for name in batch_groups},
        "pooled": {"pairs": len(pairs_all), "groups": n_groups,
                   "danger_groups": len(danger_gids)},
        "danger_quantile": args.danger_quantile,
        "channels": {},
        "per_batch_headline": {},
        "regret_sensitivity": {},
    }

    # ---- per-batch headline table (terminal + combo) ---------------------
    print("\n=== per-batch headline (PRIMARY evidence) ===")
    for name in batch_groups:
        bp = [(g, w) for gs in batch_groups[name].values() for g, w in gs]
        row = {}
        for ch in ("risk_terminal", "risk_combo"):
            d = _describe_diffs(bp, ch)
            f = _rank_flips(batch_groups[name], ch)
            row[ch] = {
                "delta_mean": d["delta_mean"],
                "sign": f"+{d['sign_pos']}/-{d['sign_neg']}",
                "sign_p": d["sign_p"],
                "top1_strict_flip_rate": f["top1_strict_flip_rate"],
                "pairwise_strict_flip_rate": f["pairwise_strict_flip_rate"],
            }
            print(f"  [{name:5s}] {ch:13s} d={_fmt(d['delta_mean'])} "
                  f"sign +{d['sign_pos']}/-{d['sign_neg']} "
                  f"(p={_fmt(d['sign_p'], '.2e')})  "
                  f"strict-top1={_fmt(f['top1_strict_flip_rate'], '.2%')}  "
                  f"strict-pair={_fmt(f['pairwise_strict_flip_rate'], '.2%')}")
        report["per_batch_headline"][name] = row

    # ---- pooled per-channel stats ----------------------------------------
    slices = {
        "all": (lambda gid: True),
        "danger": (lambda gid: gid in danger_gids),
        "calm": (lambda gid: gid not in danger_gids),
    }
    print("\n=== pooled channels ===")
    for ch in CHANNELS:
        report["channels"][ch] = {}
        for sname, pred in slices.items():
            sp = [(g, w) for _, gid, g, w in pairs_all if pred(gid)]
            sg = {gid: v for gid, v in all_groups.items() if pred(gid)}
            d = _describe_diffs(sp, ch)
            f = _rank_flips(sg, ch)
            report["channels"][ch][sname] = {"diff": d, "flips": f}
            if ch in ("risk_terminal", "risk_combo"):
                print(f"  {ch:13s} [{sname:6s}] n={d['n']:4d} "
                      f"d={_fmt(d['delta_mean'])} "
                      f"sign +{d['sign_pos']}/-{d['sign_neg']} "
                      f"(p={_fmt(d['sign_p'], '.2e')})  "
                      f"strict-top1={_fmt(f['top1_strict_flip_rate'], '.2%')}  "
                      f"strict-pair={_fmt(f['pairwise_strict_flip_rate'], '.2%')}")

    # ---- regret: merged-only with LOGO sensitivity (combo) ---------------
    print("\n=== regret (combo; MERGED-ONLY readout, LOGO sensitivity) ===")
    for sname, pred in (("all", slices["all"]), ("danger", slices["danger"])):
        gids, deploy, reverse = [], [], []
        for gid, pairs in all_groups.items():
            if len(pairs) < 2 or not pred(gid):
                continue
            dep, rev = _group_regrets(pairs, "risk_combo")
            gids.append(gid)
            deploy.append(dep)
            reverse.append(rev)
        deploy_a, reverse_a = np.array(deploy), np.array(reverse)
        sens = _logo_ratio(deploy_a, reverse_a)
        # per-group paired sign test on (deploy - reverse)
        dd = deploy_a - reverse_a
        nz = dd[np.abs(dd) > 1e-12]
        n_pos = int((nz > 0).sum())
        n_neg = int(len(nz) - n_pos)
        try:
            from scipy import stats as sps
            sp_p = (float(sps.binomtest(n_pos, len(nz), 0.5).pvalue)
                    if len(nz) else None)
        except ImportError:
            sp_p = None
        # top influential group by deployment-direction contribution
        top_i = int(np.argmax(deploy_a)) if len(deploy_a) else None
        top_share = (float(deploy_a[top_i] / deploy_a.sum())
                     if top_i is not None and deploy_a.sum() > 0 else None)
        blk = {
            "groups": len(gids),
            "deploy_regret_mean": float(deploy_a.mean()) if len(gids) else None,
            "reverse_regret_mean": float(reverse_a.mean()) if len(gids) else None,
            **sens,
            "group_sign_test": {"pos": n_pos, "neg": n_neg, "p": sp_p},
            "top_group": gids[top_i] if top_i is not None else None,
            "top_group_deploy_share": top_share,
        }
        report["regret_sensitivity"][sname] = blk
        print(f"  [{sname:6s}] groups={blk['groups']}  "
              f"deploy={_fmt(blk['deploy_regret_mean'])} "
              f"reverse={_fmt(blk['reverse_regret_mean'])}  "
              f"ratio={_fmt(blk['ratio_full'], '.2f')} "
              f"LOGO=[{_fmt(blk['ratio_logo_min'], '.2f')}, "
              f"{_fmt(blk['ratio_logo_max'], '.2f')}]")
        print(f"           group sign(dep-rev) +{n_pos}/-{n_neg} "
              f"(p={_fmt(sp_p, '.4f')})  top group {blk['top_group']} "
              f"share={_fmt(top_share, '.1%')}")

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        print(f"\n  Report saved: {args.report_json}")


if __name__ == "__main__":
    main()
