"""
V3: TDValueHead Training (6.2 TD bootstrap)
============================================
Trains ONLY the legacy additive scalar TDValueHead on snapshot TD tuples.
The base world model is frozen. This entry point is retained for reproducing
the registered 6.2 V3 artifacts; new Phase-C state-only, three-component
risk training lives in train_td_risk_v.py.

Target:
  avg: y = (1-gamma) * sum_{k<K} gamma^k c_k
           + gamma^K * stop_gradient(V_ema(g_boot))
  sum: y = sum_{k<K} gamma^k c_k
           + gamma^K * stop_gradient(V_ema(g_boot))

The historical input remains TDValueHead.pool(z_K, z_0) and therefore mixes
action-conditioned Q-mode with a zero-rollout V-mode bootstrap. Do not use
this class or script as the new pure V-head.
"""

import argparse
import copy
import json
import os
import random
from collections import defaultdict
from datetime import date

import numpy as np
import torch
import torch.nn as nn

from WorldModel.model import TDValueHead
from WorldModel.data.dump_td_tuples import (
    build_sample_index,
    load_frozen_world_model,
    load_td_tuples,
    match_tuples_to_samples,
)
from WorldModel.core.long_risk_schema import (
    LONG_RISK_OUTPUT_INDEX,
    decode_long_risk_predictions,
)


TERMINAL_DIM = LONG_RISK_OUTPUT_INDEX["terminal_q90"]


def _spearman(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def precompute_features(
    model,
    matched,
    K,
    gamma,
    value_scale,
    edge_index,
    device,
    terminal_window,
):
    """One pass through the frozen base model; downstream work is MLP-only."""
    feats = []
    with torch.no_grad():
        for i, (t, s) in enumerate(matched):
            if K not in t["frames"] or len(t["risk_seq"]) < K:
                continue

            z, e, ea = model.encode_state(
                s["node_history"],
                s["edge_index"],
                s["edge_features"],
                s["demand_context"],
            )
            sn = s["station_node_ids"].tolist()
            _, _, _, _, z_0, z_K = model.rollout(
                z,
                e,
                ea,
                s["action_node"],
                s["action_global"],
                s["edge_index"],
                sn,
            )
            g_t = TDValueHead.pool(z_K, z_0)
            lr_pred = model.long_risk_head(z_K, z_0)
            lr_components = decode_long_risk_predictions(
                lr_pred,
                scalar=lambda value: float(value.item()),
            )

            fr = t["frames"][K]
            zb, _, _ = model.encode_state(
                fr["node_history"],
                edge_index,
                fr["edge_features"],
                fr["demand_context"],
            )
            g_boot = TDValueHead.pool(zb, zb)
            lr_boot = model.long_risk_head(zb, zb)

            c = t["risk_seq"].numpy()
            R = float(np.sum((gamma ** np.arange(K)) * c[:K]))
            G_full = float(np.sum((gamma ** np.arange(len(c))) * c))
            if value_scale == "avg":
                R, G_full = (1 - gamma) * R, (1 - gamma) * G_full

            feats.append({
                "group": t["candidate_group_id"],
                "r_at_decision": t["r_at_decision"],
                "g_t": g_t,
                "g_boot": g_boot,
                "R": R,
                "G_full": G_full,
                "risk_peak": float(c.max()),
                "realized_terminal": float(
                    c[-min(terminal_window, len(c)):].mean()
                ),
                "v_proxy_t": float(lr_pred[TERMINAL_DIM].item()),
                "v_proxy_boot": float(lr_boot[TERMINAL_DIM].item()),
                "combo_pred": lr_components["long_risk_quantile_combo"],
                "terminal_pred": float(lr_pred[TERMINAL_DIM].item()),
            })
            if (i + 1) % 25 == 0:
                print(f"    features {i + 1}/{len(matched)}")
    return feats


def split_by_group(feats, val_ratio, seed):
    gids = sorted({f["group"] for f in feats})
    rng = random.Random(seed)
    rng.shuffle(gids)
    n_val = max(1, int(round(len(gids) * val_ratio))) if len(gids) > 1 else 0
    val_g = set(gids[:n_val])
    train = [f for f in feats if f["group"] not in val_g]
    val = [f for f in feats if f["group"] in val_g]
    return train, val


def pairwise_direction_acc(feats, pred_key, target_key, eps=1e-6):
    """Within-group pairwise ordering accuracy of prediction vs target."""
    groups = defaultdict(list)
    for f in feats:
        groups[f["group"]].append(f)
    correct = total = 0
    for members in groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                dt = members[i][target_key] - members[j][target_key]
                if abs(dt) <= eps:
                    continue
                dp = members[i][pred_key] - members[j][pred_key]
                total += 1
                if dp * dt > 0:
                    correct += 1
    return (correct / total if total else None), total


def main():
    ap = argparse.ArgumentParser(description="Train TDValueHead (6.2 V3)")
    ap.add_argument("--tuples", required=True)
    ap.add_argument("--samples", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument(
        "--output",
        required=True,
        help="Additive checkpoint path (*_tdv.pt); base untouched",
    )
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--K", type=int, default=50)
    ap.add_argument("--value-scale", choices=["avg", "sum"], default="avg")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--ema-tau", type=float, default=0.995)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--huber-delta", type=float, default=1.0)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--terminal-window", type=int, default=50)
    ap.add_argument("--danger-quantile", type=float, default=0.7)
    ap.add_argument(
        "--run-filter",
        type=str,
        default=None,
        help="Substring of source_run_id (e.g. 'mid_seed43')",
    )
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if os.path.abspath(args.output) == os.path.abspath(args.checkpoint):
        raise SystemExit(
            "--output must differ from --checkpoint (D1: 原 .pt 不动)"
        )

    header, tuples = load_td_tuples(args.tuples)
    if args.K not in set(header["frame_ticks"]):
        raise SystemExit(
            f"K={args.K} has no dumped frames "
            f"(available: {header['frame_ticks']})"
        )
    data_policy = header["continuation_policy"]
    arm = "V_td^{G}" if data_policy == "greedy" else "V_td^{WM}"
    print(f"TDValueHead Training (V3)  —  data policy: {data_policy} -> {arm}")
    print(
        f"  gamma={args.gamma}  K={args.K}  scale={args.value_scale}"
        f"  gamma^K={args.gamma ** args.K:.4f}"
    )

    print(f"  loading samples: {args.samples} ...")
    samples = torch.load(args.samples, weights_only=False)
    if isinstance(samples, dict) and "samples" in samples:
        samples = samples["samples"]
    index, _ = build_sample_index(samples, args.run_filter)
    matched, missing = match_tuples_to_samples(tuples, index)
    print(f"  matched: {len(matched)}/{len(tuples)} (missing {missing})")
    if len(matched) < 4:
        raise SystemExit("too few matches to train")

    model, model_config = load_frozen_world_model(
        args.checkpoint, matched[0][1], device=args.device
    )
    edge_index = header["edge_index"]

    print("  precomputing pooled features (frozen base model) ...")
    feats = precompute_features(
        model,
        matched,
        args.K,
        args.gamma,
        args.value_scale,
        edge_index,
        args.device,
        args.terminal_window,
    )
    print(f"  usable tuples: {len(feats)}")

    train, val = split_by_group(feats, args.val_ratio, args.split_seed)
    print(
        f"  split: train={len(train)} val={len(val)} "
        f"(groups {len({f['group'] for f in train})}/"
        f"{len({f['group'] for f in val})})"
    )

    def _stack(fs):
        return (
            torch.stack([f["g_t"] for f in fs]),
            torch.stack([f["g_boot"] for f in fs]),
            torch.tensor([f["R"] for f in fs], dtype=torch.float32),
        )

    g_tr, gb_tr, R_tr = _stack(train)
    g_va, gb_va, R_va = _stack(val) if val else (None, None, None)

    latent_dim = model_config.get("hidden_dim", 64)
    head = TDValueHead(latent_dim=latent_dim)
    target = copy.deepcopy(head)
    for p in target.parameters():
        p.requires_grad_(False)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    huber = nn.HuberLoss(delta=args.huber_delta)
    gk = args.gamma ** args.K

    def _val_metrics():
        if not val:
            return {}
        with torch.no_grad():
            v = head(g_va)
            y = R_va + gk * target(gb_va)
            delta = (y - v).numpy()
            sp = _spearman(v.numpy(), [f["G_full"] for f in val])
        return {
            "val_loss": float(huber(v, y).item()),
            "val_abs_delta": float(np.abs(delta).mean()),
            "val_spearman_realized": sp,
            "vhat_mean": float(v.mean()),
            "vhat_p95": float(np.quantile(v.numpy(), 0.95)),
            "vhat_max": float(v.max()),
        }

    history, best_val, best_state, patience = [], float("inf"), None, 0
    n = len(train)
    for epoch in range(1, args.epochs + 1):
        perm = torch.randperm(n)
        tot, nb = 0.0, 0
        for s0 in range(0, n, args.batch_size):
            idx = perm[s0:s0 + args.batch_size]
            with torch.no_grad():
                y = R_tr[idx] + gk * target(gb_tr[idx])
            v = head(g_tr[idx])
            loss = huber(v, y)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            opt.step()
            with torch.no_grad():
                for pt, po in zip(target.parameters(), head.parameters()):
                    pt.mul_(args.ema_tau).add_(
                        po, alpha=1 - args.ema_tau
                    )
            tot += float(loss.item())
            nb += 1

        m = {"epoch": epoch, "train_loss": tot / max(nb, 1), **_val_metrics()}
        history.append(m)
        print(
            f"  epoch {epoch:3d}  train={m['train_loss']:.5f}"
            + (
                f"  val={m['val_loss']:.5f}"
                f"  |delta|={m['val_abs_delta']:.5f}"
                f"  Vhat mean={m['vhat_mean']:.4f}"
                f" p95={m['vhat_p95']:.4f}"
                f"  spearman={m['val_spearman_realized']}"
                if val else ""
            )
        )
        if val and not np.isfinite(m["val_loss"]):
            print("  DIVERGED — stopping")
            break
        if val:
            if m["val_loss"] < best_val - 1e-6:
                best_val = m["val_loss"]
                best_state = copy.deepcopy(head.state_dict())
                patience = 0
            else:
                patience += 1
                if patience >= args.early_stop_patience:
                    print(f"  early stop at epoch {epoch}")
                    break
    if best_state is not None:
        head.load_state_dict(best_state)

    report = {
        "data_policy": data_policy,
        "arm": arm,
        "gamma": args.gamma,
        "K": args.K,
        "value_scale": args.value_scale,
        "n_train": len(train),
        "n_val": len(val),
    }
    if val:
        with torch.no_grad():
            v = head(g_va).numpy()
        for f, vi in zip(val, v):
            f["v_td"] = float(vi)

        y = R_va.numpy() + gk * np.array(
            [f["v_proxy_boot"] for f in val]
        )
        proxy_delta = np.abs(y - np.array([f["v_proxy_t"] for f in val]))
        with torch.no_grad():
            y_td = (R_va + gk * target(gb_va)).numpy()
        td_delta = np.abs(y_td - v)
        report["G1"] = {
            "abs_delta_td": float(td_delta.mean()),
            "abs_delta_proxy_terminal": float(proxy_delta.mean()),
        }

        gr = {f["group"]: f["r_at_decision"] for f in val}
        thr = float(np.quantile(list(gr.values()), args.danger_quantile))
        danger = [f for f in val if gr[f["group"]] >= thr]
        g2 = {}
        for nm, subset in (("all", val), ("danger", danger)):
            g2[nm] = {}
            for pk in ("v_td", "combo_pred", "terminal_pred"):
                for tk in ("G_full", "risk_peak"):
                    acc, npairs = pairwise_direction_acc(subset, pk, tk)
                    g2[nm][f"{pk}_vs_{tk}"] = {
                        "acc": acc, "pairs": npairs
                    }
        report["G2"] = g2
        report["G3"] = {
            "spearman_vtd_realized_disc": _spearman(
                v, [f["G_full"] for f in val]
            ),
            "spearman_terminal_pred_realized_terminal": _spearman(
                [f["terminal_pred"] for f in val],
                [f["realized_terminal"] for f in val],
            ),
        }

        with torch.no_grad():
            vb = head(gb_va).numpy()
        tail = np.array([f["G_full"] - f["R"] for f in val])
        report["boot_calibration"] = {
            "gk_vboot_mean": float((gk * vb).mean()),
            "realized_tail_mean": float(tail.mean()),
            "spearman_vboot_tail": _spearman(vb, tail),
        }

        print("\n=== V4 gate readouts (val) ===")
        print(
            f"  G1 |delta|: td={report['G1']['abs_delta_td']:.5f}"
            f"  proxy(terminal_q90)="
            f"{report['G1']['abs_delta_proxy_terminal']:.5f}"
        )
        for nm in ("all", "danger"):
            row = g2[nm]
            print(
                f"  G2 [{nm}] vs realized disc: "
                f"td={row['v_td_vs_G_full']['acc']}"
                f"  combo={row['combo_pred_vs_G_full']['acc']}"
                f"  terminal={row['terminal_pred_vs_G_full']['acc']}"
                f"  (pairs={row['v_td_vs_G_full']['pairs']})"
            )
        print(
            "  G3 Spearman: td-vs-disc="
            f"{report['G3']['spearman_vtd_realized_disc']}"
            "  terminal-vs-terminal="
            f"{report['G3']['spearman_terminal_pred_realized_terminal']}"
        )
        bc = report["boot_calibration"]
        print(
            f"  boot calib: gk*V(g_boot) mean={bc['gk_vboot_mean']:.4f}"
            f"  realized tail mean={bc['realized_tail_mean']:.4f}"
            f"  spearman={bc['spearman_vboot_tail']}"
        )

    out = {
        "schema_version": "tdv_v1",
        "td_value_head": head.state_dict(),
        "td_value_head_ema": target.state_dict(),
        "config": {
            "gamma": args.gamma,
            "K": args.K,
            "value_scale": args.value_scale,
            "latent_dim": latent_dim,
            "base_checkpoint": os.path.abspath(args.checkpoint),
            "tuples": os.path.abspath(args.tuples),
            "data_policy": data_policy,
            "drift_signal_version": f"td_value_{data_policy}_v1",
            "date": str(date.today()),
        },
        "history": history,
        "gate_report": report,
    }
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(out, args.output)
    print(f"\n  Saved additive checkpoint: {args.output}")
    rep_path = os.path.splitext(args.output)[0] + "_report.json"
    with open(rep_path, "w", encoding="utf-8") as fp:
        json.dump(
            {"history": history, "gate_report": report},
            fp,
            indent=2,
            ensure_ascii=False,
        )
    print(f"  Report: {rep_path}")


if __name__ == "__main__":
    main()
