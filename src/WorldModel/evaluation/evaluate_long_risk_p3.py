"""P3 offline group-rank validation for long-risk terminal potential.

P3 is the non-bandit direction check from
energy_s0_review_and_V_redesign.md.  It uses the existing counterfactual
candidate groups from val/test splits and checks whether predicted
long_risk terminal_q90 ranks candidates in the same direction as the offline
long-risk terminal labels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch

if __package__ is None or __package__ == "":
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

from WorldModel.dataset import WorldModelDataset
from WorldModel.core.long_risk_schema import (
    LONG_RISK_QUANTILE_COMBO_WEIGHTS,
    decode_long_risk_predictions,
    long_risk_quantile_combo,
)
from WorldModel.training.run_train_v6 import (
    _build_model_from_sample,
    _infer_num_stations,
    _load_splits,
)
from WorldModel.training.train import _sample_to_device


TARGETS = ("terminal", "combo", "cvar", "peak", "delta")


def _to_float(value) -> Optional[float]:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out) or math.isinf(out):
        return None
    return out


def _mean(values: Iterable[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mu = sum(values) / len(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def _rank(values: List[float]) -> List[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = rank
        i = j
    return ranks


def _pearson(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 1e-12 or vy <= 1e-12:
        return None
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def _spearman(xs: List[float], ys: List[float]) -> Optional[float]:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    return _pearson(_rank(xs), _rank(ys))


def _group_samples(samples: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for sample in samples:
        groups[str(sample.get("candidate_group_id", ""))].append(sample)
    return dict(groups)


def _load_model(checkpoint: str, sample: dict, horizon: int, device: torch.device):
    model = _build_model_from_sample(
        sample,
        horizon,
        _infer_num_stations(sample),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    state_dict = state.get("state_dict", state)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model, {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
    }


def _predict_sample(model, sample: dict, device: torch.device) -> Optional[dict]:
    if "long_risk_labels" not in sample:
        return None
    s = _sample_to_device(sample, device)
    station_node_ids = s.get("station_node_ids")
    if station_node_ids is not None:
        station_node_ids = station_node_ids.tolist()
    with torch.no_grad():
        z, e, edge_attr = model.encode_state(
            s["node_history"],
            s["edge_index"],
            s["edge_features"],
            s["demand_context"],
        )
        _, _, _, _, z_0, z_k = model.rollout(
            z,
            e,
            edge_attr,
            s["action_node"],
            s["action_global"],
            s["edge_index"],
            station_node_ids,
        )
        pred = model.long_risk_head(z_k, z_0).detach().cpu()

    labels = sample["long_risk_labels"]
    true_peak = _to_float(labels.get("risk_peak"))
    true_cvar = _to_float(labels.get("risk_cvar"))
    true_terminal = _to_float(labels.get("risk_terminal"))
    true_delta = _to_float(labels.get("risk_delta_group"))
    if (
        true_peak is None
        or true_cvar is None
        or true_terminal is None
        or true_delta is None
    ):
        return None
    decoded = decode_long_risk_predictions(pred, scalar=float)
    pred_peak = decoded["long_risk_peak_q95"]
    pred_cvar = decoded["long_risk_cvar_q90"]
    pred_terminal = decoded["long_risk_terminal_q90"]
    pred_delta = decoded["long_risk_delta_group_q90"]
    true_combo = long_risk_quantile_combo(
        true_peak,
        true_cvar,
        true_terminal,
    )
    pred_combo = decoded["long_risk_quantile_combo"]
    return {
        "sample_id": sample.get("sample_id"),
        "candidate_group_id": sample.get("candidate_group_id"),
        "source_load_level": sample.get("source_load_level", "unknown"),
        "source_run_id": sample.get("source_run_id", "unknown"),
        "pred_terminal": pred_terminal,
        "true_terminal": true_terminal,
        "pred_combo": pred_combo,
        "true_combo": true_combo,
        "pred_cvar": pred_cvar,
        "true_cvar": true_cvar,
        "pred_peak": pred_peak,
        "true_peak": true_peak,
        "pred_delta": pred_delta,
        "true_delta": true_delta,
    }


def _pairwise_acc(preds: List[float], trues: List[float]) -> Tuple[int, int]:
    correct = 0
    total = 0
    for i in range(len(preds)):
        for j in range(i + 1, len(preds)):
            if abs(trues[i] - trues[j]) < 1e-6:
                continue
            total += 1
            if (preds[i] - preds[j]) * (trues[i] - trues[j]) > 0:
                correct += 1
    return correct, total


def _group_metric_rows(
    pred_rows: List[dict],
    split_name: str,
) -> List[dict]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for row in pred_rows:
        groups[str(row["candidate_group_id"])].append(row)

    out = []
    for gid, members in groups.items():
        if len(members) < 2:
            continue
        load_values = [m.get("source_load_level", "unknown") for m in members]
        load = max(set(load_values), key=load_values.count)
        base = {
            "split": split_name,
            "source_load_level": load,
            "candidate_group_id": gid,
            "group_size": len(members),
        }
        for target in TARGETS:
            preds = [m[f"pred_{target}"] for m in members]
            trues = [m[f"true_{target}"] for m in members]
            correct, total = _pairwise_acc(preds, trues)
            model_pick = min(range(len(preds)), key=lambda i: preds[i])
            oracle_pick = min(range(len(trues)), key=lambda i: trues[i])
            row = dict(base)
            row.update({
                "target": target,
                "spearman": _spearman(preds, trues),
                "pairwise_correct": correct,
                "pairwise_total": total,
                "pairwise_acc": correct / total if total else None,
                "hit": 1 if model_pick == oracle_pick else 0,
                "top1_regret": trues[model_pick] - trues[oracle_pick],
                "pred_std": _std(preds),
                "true_std": _std(trues),
            })
            out.append(row)
    return out


def _summarize_group_rows(rows: List[dict]) -> List[dict]:
    summaries = []
    splits = sorted({r["split"] for r in rows})
    targets = sorted({r["target"] for r in rows})
    for split in splits:
        loads = sorted({r["source_load_level"] for r in rows if r["split"] == split})
        loads = ["all"] + loads
        for load in loads:
            for target in targets:
                subset = [
                    r for r in rows
                    if r["split"] == split
                    and r["target"] == target
                    and (load == "all" or r["source_load_level"] == load)
                ]
                if not subset:
                    continue
                pair_total = sum(r["pairwise_total"] for r in subset)
                pair_correct = sum(r["pairwise_correct"] for r in subset)
                spearmans = [
                    r["spearman"] for r in subset
                    if r["spearman"] is not None
                ]
                summaries.append({
                    "split": split,
                    "source_load_level": load,
                    "target": target,
                    "num_groups": len(subset),
                    "num_groups_spearman": len(spearmans),
                    "mean_group_size": round(
                        sum(r["group_size"] for r in subset) / len(subset),
                        4,
                    ),
                    "group_spearman_mean": (
                        round(sum(spearmans) / len(spearmans), 6)
                        if spearmans else None
                    ),
                    "group_spearman_p50": (
                        round(sorted(spearmans)[len(spearmans) // 2], 6)
                        if spearmans else None
                    ),
                    "pairwise_acc": (
                        round(pair_correct / pair_total, 6)
                        if pair_total else None
                    ),
                    "pairwise_total": pair_total,
                    "hit_rate": round(
                        sum(r["hit"] for r in subset) / len(subset),
                        6,
                    ),
                    "top1_regret_mean": round(
                        sum(r["top1_regret"] for r in subset) / len(subset),
                        6,
                    ),
                    "pred_std_mean": round(
                        sum(r["pred_std"] for r in subset) / len(subset),
                        6,
                    ),
                    "true_std_mean": round(
                        sum(r["true_std"] for r in subset) / len(subset),
                        6,
                    ),
                })
    return summaries


def _write_csv(path: str, rows: List[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _round_group_rows(rows: List[dict]) -> List[dict]:
    out = []
    for row in rows:
        r = dict(row)
        for key in ("spearman", "pairwise_acc", "top1_regret", "pred_std", "true_std"):
            if r.get(key) is not None:
                r[key] = round(float(r[key]), 6)
        out.append(r)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="P3 offline group-rank validation for long-risk terminal head"
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", nargs="+", default=["val", "test"],
                        choices=["train", "val", "test"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-json", required=True)
    parser.add_argument("--save-summary-csv", required=True)
    parser.add_argument("--save-group-csv", required=True)
    args = parser.parse_args()

    device = torch.device(
        args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    full = WorldModelDataset.from_file(args.data)
    train_ds, val_ds, test_ds = _load_splits(args.splits, full.samples)
    split_map = {"train": train_ds, "val": val_ds, "test": test_ds}

    first_sample = next(
        s for s in full.samples
        if "long_risk_labels" in s
    )
    horizon = int(first_sample["future_node_labels"].shape[0])
    model, load_info = _load_model(args.checkpoint, first_sample, horizon, device)

    all_group_rows = []
    prediction_counts = {}
    for split_name in args.split:
        dataset = split_map[split_name]
        pred_rows = []
        for sample in dataset.samples:
            pred = _predict_sample(model, sample, device)
            if pred is not None:
                pred["split"] = split_name
                pred_rows.append(pred)
        prediction_counts[split_name] = len(pred_rows)
        all_group_rows.extend(_group_metric_rows(pred_rows, split_name))

    all_group_rows = _round_group_rows(all_group_rows)
    summary_rows = _summarize_group_rows(all_group_rows)

    output = {
        "data": args.data,
        "splits": args.splits,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "checkpoint_load": load_info,
        "prediction_counts": prediction_counts,
        "long_risk_quantile_combo_weights": dict(
            LONG_RISK_QUANTILE_COMBO_WEIGHTS
        ),
        "summary": summary_rows,
        "gate_note": (
            "P3 supports the long-risk terminal head if terminal target has "
            "positive group-wise rank signal on val/test and by load. "
            "Use group_spearman_mean, pairwise_acc, and hit_rate together."
        ),
    }
    os.makedirs(os.path.dirname(args.save_json) or ".", exist_ok=True)
    with open(args.save_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    _write_csv(args.save_summary_csv, summary_rows)
    _write_csv(args.save_group_csv, all_group_rows)

    print(f"device={device}")
    print(f"prediction_counts={prediction_counts}")
    print(f"summary_json={args.save_json}")
    print(f"summary_csv={args.save_summary_csv}")
    print(f"group_csv={args.save_group_csv}")
    print("\nTerminal target summary:")
    for row in summary_rows:
        if row["target"] == "terminal":
            print(
                f"  {row['split']:>4s} {row['source_load_level']:>8s} "
                f"groups={row['num_groups']:4d} "
                f"rho={row['group_spearman_mean']} "
                f"pair_acc={row['pairwise_acc']} "
                f"hit={row['hit_rate']} "
                f"regret={row['top1_regret_mean']}"
            )


if __name__ == "__main__":
    main()
