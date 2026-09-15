"""
V0: TD Residual Diagnostic (frozen model, pure computation)
============================================================
Quantifies how self-inconsistent the current frozen head is under the
Bellman operator — the baseline that justifies (or kills) TDValueHead
training. Registered spec: td_bootstrap_6p2_plan.md section 4 (V0).

Residual per candidate, per (gamma, K):

  value-scale "avg" (default, scale-matches the proxy readout):
    delta = (1-gamma) * sum_{k<K} gamma^k * risk_seq[k]
            + gamma^K * Vhat(s_{t+K}) - Vhat(s_t)
  value-scale "sum" (raw registered form):
    delta = sum_{k<K} gamma^k * risk_seq[k]
            + gamma^K * Vhat(s_{t+K}) - Vhat(s_t)

Vhat readout (proxy, registered as semantically mismatched — the residual
level itself is the evidence baseline):
  Vhat(s_t)     = long_risk_head terminal_q90 (dim 3) on the decision-point
                  sample with the candidate action rollout (Q-mode).
  Vhat(s_{t+K}) = terminal_q90 with the zero-rollout convention
                  long_risk_head(z', z'), z' = encode(frame at t+K)
                  (no candidate action exists at a bootstrap state).

Kill-check diagnostics (plan V0 deliverable 3): within-K risk variance and
corr(Vhat, realized discounted risk) — if risk is flat within K and Vhat is
already ~linear in the MC return, TD has no headroom.

Usage:
  python -m WorldModel.evaluation.analyze_td_residual_v0 \
    --tuples DataGen/wm_data/td_tuples/mid_seed43_W200.pt \
    --samples DataGen/wm_data/phaseB_fused_b3_w200_filtered/wm_train_data_phaseB_b3_w200_filtered.pt \
    --checkpoint WorldModel/checkpoints/phaseB_b3_bneckfix_w200_filtered_decoder/best_regret_world_model.pt \
    --run-filter mid_seed43 \
    --gammas 0.99,0.995 --Ks 25,50,100 --report-json v0_report.json
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import torch

from WorldModel.data.dump_td_tuples import (
    load_td_tuples, build_sample_index, match_tuples_to_samples,
    load_frozen_world_model,
)
from WorldModel.core.long_risk_schema import LONG_RISK_OUTPUT_INDEX

TERMINAL_DIM = LONG_RISK_OUTPUT_INDEX["terminal_q90"]


def _spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c)


def _q90_readout(model, node_history, edge_index, edge_features,
                 demand_context, action_node=None, action_global=None,
                 station_node_ids=None):
    """terminal_q90 readout. With action: Q-mode (encode + rollout).
    Without action: V-mode zero-rollout convention head(z', z')."""
    z, e, ea = model.encode_state(
        node_history, edge_index, edge_features, demand_context)
    if action_node is None:
        lr = model.long_risk_head(z, z)
    else:
        sn = station_node_ids.tolist() if station_node_ids is not None else None
        _, _, _, _, z_0, z_K = model.rollout(
            z, e, ea, action_node, action_global, edge_index, sn)
        lr = model.long_risk_head(z_K, z_0)
    return float(lr[TERMINAL_DIM].item())


def _describe(name, arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return {"n": 0}
    q = np.quantile
    return {
        "n": int(len(arr)),
        "mean": float(arr.mean()), "std": float(arr.std()),
        "p10": float(q(arr, 0.1)), "p50": float(q(arr, 0.5)),
        "p90": float(q(arr, 0.9)),
        "abs_mean": float(np.abs(arr).mean()),
        "abs_p50": float(q(np.abs(arr), 0.5)),
        "abs_p90": float(q(np.abs(arr), 0.9)),
    }


def main():
    ap = argparse.ArgumentParser(description="V0 TD residual diagnostic")
    ap.add_argument("--tuples", required=True)
    ap.add_argument("--samples", required=True,
                    help="Training samples .pt (fused or per-run) providing "
                         "the decision-point observation + action fields")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--gammas", type=str, default="0.99,0.995")
    ap.add_argument("--Ks", type=str, default="25,50,100",
                    help="Must be a subset of the dumped frame ticks")
    ap.add_argument("--value-scale", choices=["avg", "sum"], default="avg")
    ap.add_argument("--danger-quantile", type=float, default=0.7)
    ap.add_argument("--run-filter", type=str, default=None,
                    help="Substring of source_run_id (e.g. 'mid_seed43') to "
                         "disambiguate raw-key collisions in fused datasets")
    ap.add_argument("--max-candidates", type=int, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--report-json", type=str, default=None)
    args = ap.parse_args()

    gammas = [float(x) for x in args.gammas.split(",") if x]
    ks_req = [int(x) for x in args.Ks.split(",") if x]

    header, tuples = load_td_tuples(args.tuples)
    edge_index = header["edge_index"].to(args.device)
    avail = set(header["frame_ticks"])
    ks = [k for k in ks_req if k in avail]
    if set(ks_req) - avail:
        print(f"  WARNING: Ks {sorted(set(ks_req) - avail)} have no dumped "
              f"frames (available: {sorted(avail)}); skipped")
    if not ks:
        raise SystemExit("no usable K values")

    print("V0 TD Residual Diagnostic")
    print(f"  tuples     : {args.tuples}  ({len(tuples)} tuples, "
          f"policy={header['continuation_policy']}, W={header['W']})")
    print(f"  loading samples: {args.samples} ...")
    samples = torch.load(args.samples, weights_only=False)
    if isinstance(samples, dict) and "samples" in samples:
        samples = samples["samples"]
    index, dup = build_sample_index(samples, args.run_filter)
    matched, missing = match_tuples_to_samples(tuples, index)
    print(f"  samples    : {len(samples)} loaded, {len(index)} indexed "
          f"(dup keys skipped: {dup})")
    print(f"  matched    : {len(matched)}/{len(tuples)} tuples "
          f"(missing: {missing})")
    if not matched:
        raise SystemExit("no tuple<->sample matches — check --source-* filters")
    if args.max_candidates:
        matched = matched[:args.max_candidates]

    model, _ = load_frozen_world_model(
        args.checkpoint, matched[0][1], device=args.device)

    # Per-candidate readouts
    rows = []
    with torch.no_grad():
        for i, (t, s) in enumerate(matched):
            v_t = _q90_readout(
                model, s["node_history"], s["edge_index"],
                s["edge_features"], s["demand_context"],
                s["action_node"], s["action_global"], s["station_node_ids"])
            v_boot = {}
            for k in ks:
                fr = t["frames"][k]
                v_boot[k] = _q90_readout(
                    model, fr["node_history"], edge_index,
                    fr["edge_features"], fr["demand_context"])
            rows.append({
                "group": t["candidate_group_id"],
                "r_at_decision": t["r_at_decision"],
                "risk_seq": t["risk_seq"].numpy(),
                "v_t": v_t, "v_boot": v_boot,
            })
            if (i + 1) % 25 == 0:
                print(f"    readouts {i+1}/{len(matched)}")

    # Danger slice by group-level r_at_decision quantile
    group_risk = {}
    for r in rows:
        group_risk[r["group"]] = r["r_at_decision"]
    thr = float(np.quantile(list(group_risk.values()), args.danger_quantile))
    danger_gids = {g for g, v in group_risk.items() if v >= thr}
    print(f"\n  groups={len(group_risk)}  danger slice: r_at_decision >= "
          f"q{args.danger_quantile:.2f} = {thr:.4f} ({len(danger_gids)} groups)")

    report = {
        "n_matched": len(matched), "value_scale": args.value_scale,
        "danger_threshold": thr, "danger_groups": len(danger_gids),
        "grid": {},
    }

    for gamma in gammas:
        for K in ks:
            deltas, deltas_dgr, deltas_calm = [], [], []
            for r in rows:
                c = r["risk_seq"][:K]
                R = float(np.sum((gamma ** np.arange(K)) * c))
                if args.value_scale == "avg":
                    R = (1.0 - gamma) * R
                d = R + (gamma ** K) * r["v_boot"][K] - r["v_t"]
                deltas.append(d)
                (deltas_dgr if r["group"] in danger_gids
                 else deltas_calm).append(d)
            key = f"gamma={gamma}_K={K}"
            report["grid"][key] = {
                "all": _describe("all", deltas),
                "danger": _describe("danger", deltas_dgr),
                "calm": _describe("calm", deltas_calm),
            }
            a = report["grid"][key]["all"]
            print(f"\n=== gamma={gamma}  K={K}  (scale={args.value_scale}) ===")
            print(f"  delta  all   : mean={a['mean']:+.4f} std={a['std']:.4f} "
                  f"p10={a['p10']:+.4f} p50={a['p50']:+.4f} p90={a['p90']:+.4f}")
            print(f"  |delta| all  : mean={a['abs_mean']:.4f} "
                  f"p50={a['abs_p50']:.4f} p90={a['abs_p90']:.4f}")
            for nm in ("danger", "calm"):
                dd = report["grid"][key][nm]
                if dd["n"]:
                    print(f"  |delta| {nm:6s}: mean={dd['abs_mean']:.4f} "
                          f"p50={dd['abs_p50']:.4f} p90={dd['abs_p90']:.4f} "
                          f"(n={dd['n']})")

    # Kill-check diagnostics (gamma/K defaults: first of each grid)
    g0, K0 = gammas[-1], ks[min(1, len(ks) - 1)] if 50 in ks else ks[0]
    if 50 in ks:
        K0 = 50
    within_std = [float(np.std(r["risk_seq"][:K0])) for r in rows]
    disc_full = []
    for r in rows:
        w = len(r["risk_seq"])
        G = float(np.sum((g0 ** np.arange(w)) * r["risk_seq"]))
        disc_full.append((1.0 - g0) * G if args.value_scale == "avg" else G)
    v_ts = [r["v_t"] for r in rows]
    sp = _spearman(v_ts, disc_full)
    print(f"\n=== kill-check diagnostics (gamma={g0}, K={K0}) ===")
    print(f"  within-K risk std : mean={np.mean(within_std):.4f} "
          f"p50={np.median(within_std):.4f} "
          f"(TD headroom-less if ~0)")
    print(f"  Vhat(s_t)         : mean={np.mean(v_ts):.4f}  "
          f"realized disc ({args.value_scale}): mean={np.mean(disc_full):.4f}")
    print(f"  Spearman(Vhat, realized disc) = "
          f"{sp if sp is not None else 'NA'}")
    report["kill_check"] = {
        "gamma": g0, "K": K0,
        "within_K_risk_std_mean": float(np.mean(within_std)),
        "vhat_mean": float(np.mean(v_ts)),
        "realized_disc_mean": float(np.mean(disc_full)),
        "spearman_vhat_realized": sp,
    }

    if args.report_json:
        with open(args.report_json, "w", encoding="utf-8") as fp:
            json.dump(report, fp, indent=2, ensure_ascii=False)
        print(f"\n  Report saved: {args.report_json}")


if __name__ == "__main__":
    main()
