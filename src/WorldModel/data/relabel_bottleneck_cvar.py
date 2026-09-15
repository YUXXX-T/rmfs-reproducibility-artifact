"""
Relabel bottleneck_CVaR from node-level congestion labels.

This updates system dim 4 without rerunning counterfactual rollouts.  The new
definition is the top-tail mean of node congestion_score over the static
bottleneck set:

  B = top 20% nodes by bottleneck_score
  bottleneck_CVaR[t] = mean(top 10% future_node_labels[t, B, congestion_score])

The script also recomputes realized_cost and, by default, B3 group baselines.
"""

import argparse
import os
import shutil
from collections import defaultdict
from typing import Dict, List

import torch

from WorldModel.core.costs import compute_realized_cost


BOTTLENECK_SCORE_DIM = 8
NODE_CONGESTION_SCORE_DIM = 5
SYSTEM_BOTTLENECK_CVAR_DIM = 4


def _quantile(x: torch.Tensor, q: float) -> float:
    if x.numel() == 0:
        return 0.0
    return float(torch.quantile(x.float(), q).item())


def _compute_group_baselines(samples: List[dict]) -> Dict[str, int]:
    groups = defaultdict(list)
    for sample in samples:
        gid = sample.get("candidate_group_id")
        if gid is not None:
            groups[gid].append(sample)

    added = 0
    mismatched = 0
    for members in groups.values():
        fsl_list = [
            s["future_system_labels"]
            for s in members
            if "future_system_labels" in s
        ]
        if not fsl_list:
            continue
        lengths = [int(f.shape[0]) for f in fsl_list]
        min_k = min(lengths)
        if any(k != min_k for k in lengths):
            mismatched += 1
        baseline = torch.stack([f[:min_k] for f in fsl_list]).mean(dim=0)
        for sample in members:
            sample["future_system_group_baseline"] = baseline.clone()
            added += 1

    return {
        "groups": len(groups),
        "baseline_added": added,
        "min_k_mismatch": mismatched,
    }


def relabel_samples(samples: List[dict], recompute_group_baseline: bool = True) -> Dict[str, float]:
    old_values = []
    new_values = []
    abs_deltas = []
    updated = 0
    skipped = 0

    for sample in samples:
        node_history = sample.get("node_history")
        future_node = sample.get("future_node_labels")
        future_system = sample.get("future_system_labels")
        if node_history is None or future_node is None or future_system is None:
            skipped += 1
            continue
        if node_history.ndim != 3 or future_node.ndim != 3 or future_system.ndim != 2:
            skipped += 1
            continue
        if node_history.shape[-1] <= BOTTLENECK_SCORE_DIM:
            skipped += 1
            continue
        if future_node.shape[-1] <= NODE_CONGESTION_SCORE_DIM:
            skipped += 1
            continue
        if future_system.shape[-1] <= SYSTEM_BOTTLENECK_CVAR_DIM:
            skipped += 1
            continue

        n_nodes = int(future_node.shape[1])
        if n_nodes <= 0:
            skipped += 1
            continue

        bottleneck_scores = node_history[-1, :, BOTTLENECK_SCORE_DIM].float()
        top_q = max(1, n_nodes // 5)
        bottleneck_idx = torch.topk(bottleneck_scores, k=top_q).indices

        pressure = future_node[:, bottleneck_idx, NODE_CONGESTION_SCORE_DIM].float()
        top_p = max(1, top_q // 10)
        new_cvar = torch.topk(pressure, k=top_p, dim=1).values.mean(dim=1)

        old_cvar = future_system[:, SYSTEM_BOTTLENECK_CVAR_DIM].detach().float().clone()
        updated_system = future_system.clone()
        updated_system[:, SYSTEM_BOTTLENECK_CVAR_DIM] = new_cvar.to(
            device=updated_system.device,
            dtype=updated_system.dtype,
        )
        sample["future_system_labels"] = updated_system
        if "realized_cost" in sample:
            sample["realized_cost"] = compute_realized_cost(updated_system.detach().cpu())

        old_values.append(old_cvar.cpu())
        new_values.append(new_cvar.detach().cpu())
        abs_deltas.append((new_cvar.detach().cpu() - old_cvar.cpu()).abs())
        updated += 1

    stats = {
        "samples": len(samples),
        "updated": updated,
        "skipped": skipped,
    }
    if old_values:
        old = torch.cat(old_values)
        new = torch.cat(new_values)
        delta = torch.cat(abs_deltas)
        stats.update({
            "old_mean": float(old.mean().item()),
            "new_mean": float(new.mean().item()),
            "old_q90": _quantile(old, 0.90),
            "new_q90": _quantile(new, 0.90),
            "old_q95": _quantile(old, 0.95),
            "new_q95": _quantile(new, 0.95),
            "mean_abs_delta": float(delta.mean().item()),
            "max_abs_delta": float(delta.max().item()),
        })

    if recompute_group_baseline:
        stats.update(_compute_group_baselines(samples))

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input wm_train_data*.pt")
    parser.add_argument("--output", required=True, help="Output relabeled .pt")
    parser.add_argument("--splits", default="", help="Optional splits.json to copy")
    parser.add_argument("--output-splits", default="", help="Output splits path")
    parser.add_argument(
        "--no-group-baseline",
        action="store_true",
        help="Do not recompute future_system_group_baseline",
    )
    args = parser.parse_args()

    samples = torch.load(args.input, weights_only=False)
    if not isinstance(samples, list):
        raise TypeError(f"Expected list samples in {args.input}, got {type(samples)!r}")

    stats = relabel_samples(samples, recompute_group_baseline=not args.no_group_baseline)

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    torch.save(samples, args.output)

    if args.splits:
        output_splits = args.output_splits
        if not output_splits:
            output_splits = os.path.join(out_dir or ".", "splits.json")
        split_dir = os.path.dirname(output_splits)
        if split_dir:
            os.makedirs(split_dir, exist_ok=True)
        shutil.copyfile(args.splits, output_splits)

    print("Relabeled bottleneck_CVaR")
    for key in sorted(stats):
        print(f"  {key}: {stats[key]}")
    print(f"  output: {args.output}")


if __name__ == "__main__":
    main()
