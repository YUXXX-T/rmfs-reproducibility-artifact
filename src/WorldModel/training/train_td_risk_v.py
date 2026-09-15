"""Train an immediate-risk residual TD V-head on Phase-C streams.

The trained object is policy-specific::

    J_risk^pi(s) = [J_stall(s), J_deadlock(s), J_handoff(s)]

It is intentionally not an action-value head.  Every input represents one
encoded state and includes the separately encoded demand context.  The
learned network is a zero-initialised correction around current raw risk,
not a from-scratch absolute-value predictor::

    J_hat(s) = clip(current_raw_risk(s) + residual_theta(s), 0, 1)

Targets use raw (unclipped) risk components from
``td_stream_tuples_v1``::

    y_t = (1-gamma) * sum_{k < K} gamma^k c_{t+k}
          + gamma^K * stop_gradient(J_ema(s_{t+K}))

A finite-trajectory discounted-average MC target is used as a calibration
anchor.  Complete seeds/runs, never adjacent frames, are held out.  Each risk
component is enabled only after a fail-closed multi-metric certificate:
held-out MC MAE improvement, RMSE non-degradation, rank preservation,
calibration bias, and non-collapsed prediction scale.  The same checks also
run on every held-out seed/run and load-specific arm by default.  Temporal
direction remains a reported diagnostic.  Failed components automatically
fall back to immediate risk.

Example (Linux)::

    python -m WorldModel.training.train_td_risk_v \
      --tuples phaseC_low_a1_td_tuples.pt phaseC_mid_a1_td_tuples.pt \
               phaseC_high_a1_td_tuples.pt \
      --checkpoint WorldModel/checkpoints/phaseC_wm1/best.pt \
      --expected-policy-family A1_LEGACY_EXACT \
      --val-seeds 209,210 --gamma 0.995 --epochs 40 \
      --output WorldModel/checkpoints/phaseC_wm1/td_risk_vhead_a1.pt
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
from datetime import date
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from WorldModel.model import RMFSWorldModel, TDRiskVHead
from WorldModel.core.lyapunov import PHYSICAL_SUMMARY_NAMES


SCHEMA_VERSION = "td_stream_tuples_v1"
HEAD_SCHEMA_VERSION = "td_risk_residual_vhead_v4"
COMPONENT_GATE_SCHEMA_VERSION = "td_component_gate_multimetric_v1"
RISK_NAMES = tuple(TDRiskVHead.RISK_COMPONENTS)
PHYSICAL_SUMMARY_DIM = len(PHYSICAL_SUMMARY_NAMES)


def _rankdata_average(values: Sequence[float]) -> np.ndarray:
    """Average ranks for ties (SciPy-free and deterministic)."""
    x = np.asarray(values, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and x[order[j]] == x[order[i]]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def spearman_tie_safe(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """Spearman correlation with correct average ranks for repeated values."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) < 3 or len(a) != len(b):
        return None
    ra, rb = _rankdata_average(a), _rankdata_average(b)
    if np.std(ra) == 0.0 or np.std(rb) == 0.0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


def _parse_int_set(text: Optional[str]) -> Optional[set[int]]:
    if not text:
        return None
    return {int(part.strip()) for part in text.split(",") if part.strip()}


def _to_device_frame(frame: dict, edge_index: torch.Tensor, device: str):
    return (
        frame["node_history"].to(device),
        edge_index.to(device),
        frame["edge_features"].to(device),
        frame["demand_context"].to(device),
    )


def _checkpoint_model_config(
    checkpoint: dict,
    example_frame: dict,
    num_stations: int,
) -> Tuple[dict, dict]:
    """Return constructor config and state_dict for a frozen base WM."""
    allowed = {
        "node_feat_dim", "edge_feat_dim", "demand_dim",
        "action_node_dim", "action_global_dim", "hidden_dim",
        "num_spatial_layers", "rollout_horizon", "num_stations",
    }
    config = {
        "node_feat_dim": int(example_frame["node_history"].shape[-1]),
        "edge_feat_dim": int(example_frame["edge_features"].shape[-1]),
        "demand_dim": int(example_frame["demand_context"].shape[-1]),
        "action_node_dim": 8,
        "action_global_dim": 6,
        "hidden_dim": 64,
        "num_spatial_layers": 3,
        "rollout_horizon": 10,
        "num_stations": int(num_stations),
    }
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        saved = checkpoint.get("model_config", {})
        config.update({k: v for k, v in saved.items() if k in allowed})
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    return config, state_dict


def load_frozen_world_model(
    checkpoint_path: str,
    example_frame: dict,
    num_stations: int,
    device: str,
) -> Tuple[RMFSWorldModel, dict]:
    checkpoint = torch.load(
        checkpoint_path, map_location="cpu", weights_only=False
    )
    config, state_dict = _checkpoint_model_config(
        checkpoint, example_frame, num_stations
    )
    model = RMFSWorldModel(**config)
    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, config


def _payload_runs(payload: dict) -> List[dict]:
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"expected {SCHEMA_VERSION}, got {payload.get('schema_version')!r}; "
            "rebuild Phase-C streams with WorldModel.data.build_td_stream_tuples"
        )
    if tuple(payload.get("risk_component_names", ())) != RISK_NAMES:
        raise ValueError(
            "tuple payload lacks the registered raw risk component manifest "
            f"{RISK_NAMES}; rebuild it with the current builder"
        )
    runs = payload.get("runs") or []
    if not runs:
        raise ValueError("tuple payload contains no runs")
    return runs


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_stream_checkpoint(
    runs: Sequence[dict],
    checkpoint_path: str,
    allow_mismatch: bool = False,
) -> dict:
    """Verify that state latents use the checkpoint recorded by the stream.

    Two different files in this repository share the basename
    ``best_regret_world_model.pt``.  Comparing names is therefore unsafe.
    When the recorded source exists locally, compare SHA256 digests; an
    unresolved recorded path or a digest mismatch requires an explicit
    override instead of silently changing the latent representation.
    """
    requested = Path(checkpoint_path).expanduser().resolve()
    if not requested.is_file():
        raise ValueError(f"base checkpoint does not exist: {requested}")
    requested_hash = _sha256_file(requested)

    recorded_values = sorted({
        str(run.get("header", {}).get("checkpoint_path"))
        for run in runs
        if run.get("header", {}).get("checkpoint_path")
    })
    verified = []
    unresolved = []
    mismatched = []
    for raw in recorded_values:
        source = Path(raw).expanduser()
        if not source.is_absolute():
            source = Path.cwd() / source
        source = source.resolve()
        if not source.is_file():
            unresolved.append({"recorded": raw, "resolved": str(source)})
            continue
        source_hash = _sha256_file(source)
        row = {
            "recorded": raw,
            "resolved": str(source),
            "sha256": source_hash,
        }
        if source_hash == requested_hash:
            verified.append(row)
        else:
            mismatched.append(row)

    if (unresolved or mismatched) and not allow_mismatch:
        details = []
        if mismatched:
            details.append(
                "digest mismatch: "
                + ", ".join(item["recorded"] for item in mismatched)
            )
        if unresolved:
            details.append(
                "unresolved source: "
                + ", ".join(item["recorded"] for item in unresolved)
            )
        raise ValueError(
            "TD stream/base-checkpoint provenance check failed ("
            + "; ".join(details)
            + "). Use the checkpoint that generated the stream, or pass "
              "--allow-checkpoint-mismatch only for a registered ablation."
        )

    return {
        "requested": str(requested),
        "requested_sha256": requested_hash,
        "recorded": recorded_values,
        "verified": verified,
        "unresolved": unresolved,
        "mismatched": mismatched,
        "override_used": bool(allow_mismatch and (unresolved or mismatched)),
        "metadata_missing": not bool(recorded_values),
    }


def _first_frame(payload: dict) -> Tuple[dict, int]:
    for run in _payload_runs(payload):
        frames = run.get("frames_by_tick") or {}
        if frames:
            station_ids = run.get("header", {}).get("station_node_ids") or []
            return frames[sorted(frames)[0]], len(station_ids)
    raise ValueError("tuple payload contains no observation frames")


def _bottleneck_scores(frame: dict) -> torch.Tensor:
    history = frame["node_history"]
    if history.ndim != 3 or history.shape[-1] <= 8:
        raise ValueError(
            "node_history must retain feature channel 8 (static bottleneck score)"
        )
    return history[-1, :, 8]


def _discounted_prefix(
    costs: torch.Tensor,
    gamma: float,
    steps: int,
) -> torch.Tensor:
    weights = gamma ** torch.arange(steps, dtype=costs.dtype)
    return (1.0 - gamma) * (weights[:, None] * costs[:steps]).sum(dim=0)


def _discounted_mc_average(costs: torch.Tensor, gamma: float) -> torch.Tensor:
    """Finite trajectory estimate of the infinite discounted average risk."""
    n = len(costs)
    weights = gamma ** torch.arange(n, dtype=costs.dtype)
    return (weights[:, None] * costs).sum(dim=0) / weights.sum().clamp_min(1e-12)


def _mc_anchor_eligible(record: dict) -> bool:
    """Only full W-length windows may act as finite-horizon MC anchors."""
    if "mc_anchor_eligible" in record:
        return bool(record["mc_anchor_eligible"])
    return not bool(record.get("truncated", False))


def _encode_run(
    model: RMFSWorldModel,
    run: dict,
    K: int,
    gamma: float,
    bottleneck_fraction: float,
    use_physical_summary: bool,
    device: str,
    max_segments: Optional[int],
) -> List[dict]:
    tuples = list(run.get("tuples") or [])
    if max_segments is not None:
        tuples = tuples[:max_segments]
    if not tuples:
        return []

    frames = run["frames_by_tick"]
    header = run.get("header", {})
    edge_index = header.get("edge_index")
    station_ids = header.get("station_node_ids") or []
    if edge_index is None:
        raise ValueError(f"{run.get('run_id')}: missing edge_index")
    if use_physical_summary:
        names = tuple(header.get("lyapunov_l0_summary_names") or ())
        if names != tuple(PHYSICAL_SUMMARY_NAMES):
            raise ValueError(
                f"{run.get('run_id')}: incompatible or missing L0 physical "
                "summary manifest"
            )

    needed_ticks = sorted({
        int(tick)
        for rec in tuples
        for tick in (rec["s_t_tick"], rec["boot_tick"])
    })
    feature_by_tick: Dict[int, torch.Tensor] = {}
    with torch.no_grad():
        for tick in needed_ticks:
            if tick not in frames:
                raise ValueError(f"{run.get('run_id')}: missing frame at tick {tick}")
            frame = frames[tick]
            node_history, ei, edge_features, demand = _to_device_frame(
                frame, edge_index, device
            )
            z, e_demand, _ = model.encode_state(
                node_history, ei, edge_features, demand
            )
            physical_summary = None
            if use_physical_summary:
                snapshot = frame.get("lyapunov_l0")
                values = (
                    snapshot.get("physical_summary")
                    if isinstance(snapshot, dict) else None
                )
                if values is None:
                    raise ValueError(
                        f"{run.get('run_id')}: frame {tick} lacks the L0 "
                        "physical summary; recollect with "
                        "--td-stream-lyapunov-l0"
                    )
                physical_summary = torch.as_tensor(
                    values, dtype=z.dtype, device=z.device
                ).flatten()
                if physical_summary.numel() != PHYSICAL_SUMMARY_DIM:
                    raise ValueError(
                        f"{run.get('run_id')}: frame {tick} L0 summary has "
                        f"{physical_summary.numel()} fields, expected "
                        f"{PHYSICAL_SUMMARY_DIM}"
                    )
            g = TDRiskVHead.build_state_features(
                z,
                e_demand,
                station_node_ids=station_ids,
                bottleneck_scores=_bottleneck_scores(frame).to(device),
                physical_summary=physical_summary,
                bottleneck_fraction=bottleneck_fraction,
            )
            feature_by_tick[tick] = g.detach().cpu()

    seed = header.get("seed")
    rows = []
    for rec in tuples:
        costs = rec.get("risk_components_seq")
        if costs is None:
            raise ValueError(
                f"{run.get('run_id')}: tuple is missing risk_components_seq; "
                "rebuild with the current stream tuple builder"
            )
        costs = costs.float()
        if costs.ndim != 2 or costs.shape[1] != len(RISK_NAMES):
            raise ValueError(
                f"{run.get('run_id')}: invalid risk component shape "
                f"{tuple(costs.shape)}"
            )
        if len(costs) < K:
            continue
        boot_current_risk = torch.as_tensor(
            rec.get("risk_components_at_boot", costs[K - 1]),
            dtype=torch.float32,
        ).flatten()
        if boot_current_risk.numel() != len(RISK_NAMES):
            raise ValueError(
                f"{run.get('run_id')}: invalid bootstrap current-risk "
                f"shape {tuple(boot_current_risk.shape)}"
            )
        if not torch.allclose(
                boot_current_risk, costs[K - 1], atol=1e-6, rtol=1e-6):
            raise ValueError(
                f"{run.get('run_id')}: bootstrap risk does not align with "
                "risk_components_seq[K-1]"
            )
        rows.append({
            "g_t": feature_by_tick[int(rec["s_t_tick"])],
            "g_boot": feature_by_tick[int(rec["boot_tick"])],
            "R": _discounted_prefix(costs, gamma, K),
            "G_mc": _discounted_mc_average(costs, gamma),
            "mc_mask": _mc_anchor_eligible(rec),
            "current_risk": torch.as_tensor(
                rec.get("risk_components_at_start", torch.zeros(3)),
                dtype=torch.float32,
            ),
            # risk_components_seq[k-1] is the post-step risk at tau+k.
            # The bootstrap frame is the post-step state at tau+K, hence this
            # is the matching immediate-risk baseline for J(s_{t+K}).
            "boot_current_risk": boot_current_risk,
            "run_id": run.get("run_id", "unknown"),
            "seed": int(seed) if seed is not None else None,
            "arm_label": run.get("arm_label", "unknown"),
            "policy_family": run.get("policy_family", "unknown"),
            "tick": int(rec["s_t_tick"]),
        })
    return rows


def _split_samples(
    rows: List[dict],
    val_ratio: float,
    split_seed: int,
    val_seeds: Optional[set[int]],
) -> Tuple[List[dict], List[dict], dict]:
    if val_seeds is not None:
        val = [row for row in rows if row["seed"] in val_seeds]
        train = [row for row in rows if row["seed"] not in val_seeds]
        split = {"unit": "seed", "val_units": sorted(val_seeds)}
    else:
        units = sorted({
            ("seed", row["seed"]) if row["seed"] is not None
            else ("run", row["run_id"])
            for row in rows
        }, key=str)
        rng = random.Random(split_seed)
        rng.shuffle(units)
        n_val = max(1, int(round(len(units) * val_ratio))) if len(units) > 1 else 0
        val_units = set(units[:n_val])

        def unit(row):
            return (("seed", row["seed"]) if row["seed"] is not None
                    else ("run", row["run_id"]))

        val = [row for row in rows if unit(row) in val_units]
        train = [row for row in rows if unit(row) not in val_units]
        split = {"unit": "seed_or_run", "val_units": [str(x) for x in units[:n_val]]}
    if not train or not val:
        raise ValueError(
            f"run/seed split produced train={len(train)} val={len(val)}; "
            "provide at least two seeds/runs or adjust --val-seeds"
        )
    return train, val, split


def _stack(rows: Sequence[dict]) -> Tuple[torch.Tensor, ...]:
    return (
        torch.stack([row["g_t"] for row in rows]),
        torch.stack([row["g_boot"] for row in rows]),
        torch.stack([row["R"] for row in rows]),
        torch.stack([row["G_mc"] for row in rows]),
        torch.stack([row["current_risk"] for row in rows]),
        torch.stack([row["boot_current_risk"] for row in rows]),
        torch.tensor([row["mc_mask"] for row in rows], dtype=torch.bool),
    )


def _component_metrics(pred: np.ndarray, target: np.ndarray) -> dict:
    result = {}
    for index, name in enumerate(RISK_NAMES):
        p, y = pred[:, index], target[:, index]
        pred_mean = float(np.mean(p))
        target_mean = float(np.mean(y))
        pred_std = float(np.std(p))
        target_std = float(np.std(y))
        target_abs_mean = float(np.mean(np.abs(y)))
        bias = float(np.mean(p - y))
        result[name] = {
            "n": int(len(y)),
            "mae": float(np.mean(np.abs(p - y))),
            "rmse": float(np.sqrt(np.mean((p - y) ** 2))),
            "bias": bias,
            "spearman": spearman_tie_safe(p, y),
            "pred_mean": pred_mean,
            "target_mean": target_mean,
            "pred_std": pred_std,
            "target_std": target_std,
            "target_abs_mean": target_abs_mean,
            "relative_abs_bias": (
                abs(bias) / target_abs_mean
                if target_abs_mean > 1e-12 else None
            ),
            "mean_recovery_ratio": (
                pred_mean / target_mean if target_mean > 1e-12 else None
            ),
            "std_recovery_ratio": (
                pred_std / target_std if target_std > 1e-12 else None
            ),
        }
    return result


def _series_gate_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    immediate: np.ndarray,
    *,
    min_anchors: int,
    min_absolute_improvement: float,
    min_relative_improvement: float,
    min_spearman: float,
    max_spearman_drop: float,
    max_relative_bias: float,
    min_std_ratio: float,
    max_rmse_relative_degrade: float,
    variation_epsilon: float,
) -> dict:
    """Return the static part of a fail-closed component certificate.

    MAE alone rewards an all-zero predictor when a risk channel is sparse.
    The additional RMSE, rank, calibration-bias, and prediction-scale checks
    make that degenerate solution ineligible even when its aggregate absolute
    error is numerically smaller.  Ranking and scale are explicitly marked
    not-applicable when the target itself is constant.
    """
    p = np.asarray(pred, dtype=float)
    y = np.asarray(target, dtype=float)
    base = np.asarray(immediate, dtype=float)
    if not (len(p) == len(y) == len(base)):
        raise ValueError("gate series must have equal lengths")

    n = int(len(y))
    candidate_mae = (
        float(np.mean(np.abs(p - y))) if n else math.inf
    )
    immediate_mae = (
        float(np.mean(np.abs(base - y))) if n else math.inf
    )
    candidate_rmse = (
        float(np.sqrt(np.mean((p - y) ** 2))) if n else math.inf
    )
    immediate_rmse = (
        float(np.sqrt(np.mean((base - y) ** 2))) if n else math.inf
    )
    improvement = immediate_mae - candidate_mae
    required = max(
        float(min_absolute_improvement),
        float(min_relative_improvement) * immediate_mae,
    )
    candidate_spearman = spearman_tie_safe(p, y)
    immediate_spearman = spearman_tie_safe(base, y)
    pred_mean = float(np.mean(p)) if n else 0.0
    target_mean = float(np.mean(y)) if n else 0.0
    pred_std = float(np.std(p)) if n else 0.0
    target_std = float(np.std(y)) if n else 0.0
    candidate_bias = float(np.mean(p - y)) if n else math.inf
    target_abs_mean = float(np.mean(np.abs(y))) if n else 0.0
    bias_scale = max(target_abs_mean, float(variation_epsilon))
    relative_abs_bias = abs(candidate_bias) / bias_scale
    target_has_variation = target_std > float(variation_epsilon)
    std_ratio = pred_std / target_std if target_has_variation else None
    rmse_limit = (
        immediate_rmse * (1.0 + float(max_rmse_relative_degrade))
        + float(variation_epsilon)
    )

    applicable = {
        "rank_floor": target_has_variation,
        "rank_preservation": target_has_variation,
        "std_ratio": target_has_variation,
    }

    checks = {
        "enough_anchors": n >= int(min_anchors),
        "finite": bool(
            n
            and np.isfinite(candidate_mae)
            and np.isfinite(immediate_mae)
            and np.isfinite(candidate_rmse)
            and np.isfinite(immediate_rmse)
            and np.isfinite(candidate_bias)
            and np.isfinite(pred_mean)
            and np.isfinite(target_mean)
            and np.isfinite(pred_std)
            and np.isfinite(target_std)
        ),
        "mae_improvement": bool(
            np.isfinite(improvement) and improvement > required
        ),
        "rmse_non_degradation": bool(
            np.isfinite(candidate_rmse)
            and np.isfinite(rmse_limit)
            and candidate_rmse <= rmse_limit
        ),
        "rank_floor": bool(
            not target_has_variation
            or (
                candidate_spearman is not None
                and np.isfinite(candidate_spearman)
                and candidate_spearman >= float(min_spearman)
            )
        ),
        "rank_preservation": bool(
            not target_has_variation
            or (
                candidate_spearman is not None
                and np.isfinite(candidate_spearman)
                and (
                    immediate_spearman is None
                    or not np.isfinite(immediate_spearman)
                    or candidate_spearman
                    >= immediate_spearman - float(max_spearman_drop)
                )
            )
        ),
        "relative_bias": bool(
            np.isfinite(relative_abs_bias)
            and relative_abs_bias <= float(max_relative_bias)
        ),
        "std_ratio": bool(
            not target_has_variation
            or (
                std_ratio is not None
                and np.isfinite(std_ratio)
                and std_ratio >= float(min_std_ratio)
            )
        ),
    }
    failed_checks = [name for name, passed in checks.items() if not passed]
    return {
        "n": n,
        "candidate_mae": candidate_mae,
        "immediate_risk_mae": immediate_mae,
        "candidate_rmse": candidate_rmse,
        "immediate_risk_rmse": immediate_rmse,
        "rmse_limit": rmse_limit,
        "absolute_improvement": improvement,
        "required_improvement": required,
        "candidate_spearman": candidate_spearman,
        "immediate_risk_spearman": immediate_spearman,
        "candidate_mean": pred_mean,
        "target_mean": target_mean,
        "candidate_std": pred_std,
        "target_std": target_std,
        "candidate_bias": candidate_bias,
        "target_abs_mean": target_abs_mean,
        "relative_abs_bias": relative_abs_bias,
        "std_ratio": std_ratio,
        "target_has_variation": target_has_variation,
        "applicable": applicable,
        "checks": checks,
        "failed_checks": failed_checks,
        "passed": not failed_checks,
    }


def _validation_breakdowns(
    pred: np.ndarray,
    target: np.ndarray,
    immediate: np.ndarray,
    rows: Sequence[dict],
) -> dict:
    """Report candidate-vs-immediate MC metrics by seed and load arm."""
    if len(pred) != len(target) or len(pred) != len(immediate):
        raise ValueError("breakdown arrays must have equal lengths")
    if len(rows) != len(target):
        raise ValueError("breakdown rows must align with MC-anchor arrays")

    def grouped(key_fn) -> dict:
        groups: Dict[str, List[int]] = {}
        for index, row in enumerate(rows):
            groups.setdefault(key_fn(row), []).append(index)

        result = {}
        for key, indices in sorted(groups.items()):
            candidate = _component_metrics(pred[indices], target[indices])
            baseline = _component_metrics(immediate[indices], target[indices])
            result[key] = {
                "n": len(indices),
                "candidate": candidate,
                "immediate_risk": baseline,
                "absolute_mae_improvement": {
                    name: baseline[name]["mae"] - candidate[name]["mae"]
                    for name in RISK_NAMES
                },
            }
        return result

    return {
        "by_seed": grouped(
            lambda row: (
                str(row["seed"])
                if row.get("seed") is not None
                else f"run:{row.get('run_id', 'unknown')}"
            )
        ),
        "by_arm": grouped(
            lambda row: str(row.get("arm_label") or "unknown")
        ),
    }


def _component_gate_decisions(
    pred: np.ndarray,
    target: np.ndarray,
    immediate: np.ndarray,
    rows: Sequence[dict],
    min_absolute_improvement: float = 0.001,
    min_relative_improvement: float = 0.02,
    require_all_units: bool = True,
    min_unit_anchors: int = 20,
    danger_threshold: float = 0.8,
    min_spearman: float = 0.10,
    max_spearman_drop: float = 0.05,
    max_relative_bias: float = 0.50,
    min_std_ratio: float = 0.10,
    max_rmse_relative_degrade: float = 0.0,
    variation_epsilon: float = 1e-8,
) -> dict:
    """Issue per-component multi-metric deployment certificates.

    The aggregate must improve MC MAE without degrading RMSE, retain useful
    rank and prediction scale, and remain calibrated in normalized bias.  By
    default the same checks must pass in every held-out seed/run and load arm.
    Temporal direction (including the danger slice) remains a reported
    diagnostic; it is not a hard certificate because sparse adjacent pairs can
    otherwise disable a well-calibrated state-value estimate nondeterministically.
    """
    if len(pred) != len(target) or len(pred) != len(immediate):
        raise ValueError("gate arrays must have equal lengths")
    if len(rows) != len(target):
        raise ValueError("gate rows must align with MC-anchor arrays")
    if min_absolute_improvement < 0.0 or min_relative_improvement < 0.0:
        raise ValueError("gate improvement thresholds must be non-negative")
    if min_unit_anchors <= 0:
        raise ValueError("min_unit_anchors must be positive")
    if not -1.0 <= min_spearman <= 1.0:
        raise ValueError("min_spearman must lie in [-1, 1]")
    if max_spearman_drop < 0.0:
        raise ValueError("max_spearman_drop must be non-negative")
    if max_relative_bias < 0.0 or min_std_ratio < 0.0:
        raise ValueError("gate bias/std thresholds must be non-negative")
    if max_rmse_relative_degrade < 0.0:
        raise ValueError("max_rmse_relative_degrade must be non-negative")
    if variation_epsilon <= 0.0:
        raise ValueError("variation_epsilon must be positive")

    if len(target) == 0:
        return {
            name: {
                "enabled": False,
                "fallback": "current_raw_risk",
                "gate_schema_version": COMPONENT_GATE_SCHEMA_VERSION,
                "aggregate": {
                    "n": 0,
                    "candidate_mae": None,
                    "immediate_risk_mae": None,
                    "absolute_improvement": None,
                    "required_improvement": None,
                    "checks": {},
                    "failed_checks": ["no_complete_held_out_mc_anchors"],
                    "passed": False,
                },
                "require_all_units": bool(require_all_units),
                "min_unit_anchors": int(min_unit_anchors),
                "units": {},
                "reasons": ["no_complete_held_out_mc_anchors"],
            }
            for name in RISK_NAMES
        }

    by_unit: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        seed = row.get("seed")
        seed_or_run = (
            f"seed:{seed}"
            if seed is not None
            else f"run:{row.get('run_id', 'unknown')}"
        )
        arm = f"arm:{str(row.get('arm_label') or 'unknown')}"
        for unit in (seed_or_run, arm):
            by_unit.setdefault(unit, []).append(index)

    candidate_direction = _temporal_direction_metrics(
        pred, target, rows, danger_threshold
    )
    immediate_direction = _temporal_direction_metrics(
        immediate, target, rows, danger_threshold
    )

    result = {}
    for component, name in enumerate(RISK_NAMES):
        aggregate = _series_gate_metrics(
            pred[:, component],
            target[:, component],
            immediate[:, component],
            min_anchors=min_unit_anchors,
            min_absolute_improvement=min_absolute_improvement,
            min_relative_improvement=min_relative_improvement,
            min_spearman=min_spearman,
            max_spearman_drop=max_spearman_drop,
            max_relative_bias=max_relative_bias,
            min_std_ratio=min_std_ratio,
            max_rmse_relative_degrade=max_rmse_relative_degrade,
            variation_epsilon=variation_epsilon,
        )
        direction = candidate_direction[name]
        base_direction = immediate_direction[name]
        aggregate["temporal_direction"] = direction
        aggregate["immediate_risk_temporal_direction"] = base_direction
        aggregate["reported_diagnostics"] = [
            "within_run_temporal_direction",
            "danger_slice_temporal_direction",
        ]

        unit_rows = {}
        all_units_pass = True
        if require_all_units:
            for unit, indices in sorted(by_unit.items()):
                unit_metric = _series_gate_metrics(
                    pred[indices, component],
                    target[indices, component],
                    immediate[indices, component],
                    min_anchors=min_unit_anchors,
                    min_absolute_improvement=min_absolute_improvement,
                    min_relative_improvement=min_relative_improvement,
                    min_spearman=min_spearman,
                    max_spearman_drop=max_spearman_drop,
                    max_relative_bias=max_relative_bias,
                    min_std_ratio=min_std_ratio,
                    max_rmse_relative_degrade=max_rmse_relative_degrade,
                    variation_epsilon=variation_epsilon,
                )
                # Preserve the historical convenience field while the full
                # check map makes the new multi-metric reason auditable.
                unit_metric["enough_anchors"] = unit_metric["checks"][
                    "enough_anchors"
                ]
                unit_rows[unit] = unit_metric
                all_units_pass = all_units_pass and unit_metric["passed"]

        enabled = aggregate["passed"] and (
            all_units_pass if require_all_units else True
        )
        reasons = [
            f"aggregate_{check}"
            for check in aggregate["failed_checks"]
        ]
        if require_all_units and not all_units_pass:
            reasons.append("not_all_validation_units_passed")
        result[name] = {
            "enabled": bool(enabled),
            "fallback": "current_raw_risk" if not enabled else None,
            "gate_schema_version": COMPONENT_GATE_SCHEMA_VERSION,
            "aggregate": aggregate,
            "require_all_units": bool(require_all_units),
            "min_unit_anchors": int(min_unit_anchors),
            "units": unit_rows,
            "reasons": reasons,
        }
    return result


def _danger_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> dict:
    score = target.max(axis=1)
    mask = score >= threshold
    if not mask.any():
        return {"n": 0, "threshold": threshold}
    return {
        "n": int(mask.sum()),
        "threshold": float(threshold),
        "mae": float(np.mean(np.abs(pred[mask] - target[mask]))),
        "underestimate_rate": float(np.mean(pred[mask] < target[mask])),
    }


def _temporal_direction_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    rows: Sequence[dict],
    danger_threshold: float,
) -> dict:
    """Consecutive-state risk direction accuracy within each held-out run."""
    by_run: Dict[str, List[int]] = {}
    for index, row in enumerate(rows):
        by_run.setdefault(str(row.get("run_id", "unknown")), []).append(index)
    for indices in by_run.values():
        indices.sort(key=lambda index: int(rows[index].get("tick", index)))

    result = {}
    for component, name in enumerate(RISK_NAMES):
        correct = pairs = danger_correct = danger_pairs = 0
        for indices in by_run.values():
            for left, right in zip(indices, indices[1:]):
                target_delta = target[right, component] - target[left, component]
                if abs(float(target_delta)) <= 1e-8:
                    continue
                pred_delta = pred[right, component] - pred[left, component]
                is_correct = float(pred_delta) * float(target_delta) > 0.0
                pairs += 1
                correct += int(is_correct)
                is_danger = max(
                    float(target[left].max()),
                    float(target[right].max()),
                ) >= danger_threshold
                if is_danger:
                    danger_pairs += 1
                    danger_correct += int(is_correct)
        result[name] = {
            "accuracy": correct / pairs if pairs else None,
            "pairs": pairs,
            "danger_accuracy": (
                danger_correct / danger_pairs if danger_pairs else None
            ),
            "danger_pairs": danger_pairs,
        }
    return result


def _evaluate(
    head: TDRiskVHead,
    target_head: TDRiskVHead,
    tensors: Tuple[torch.Tensor, ...],
    gamma_k: float,
    mc_weight: float,
    device: str,
    danger_threshold: float,
    train_mc_mean: np.ndarray,
    rows: Optional[Sequence[dict]] = None,
) -> dict:
    g, gb, R, G_mc, current, boot_current, mc_mask = tensors
    with torch.no_grad():
        g_device = g.to(device)
        current_device = current.to(device)
        pred = head(
            g_device, current_device, apply_component_gates=False
        )
        gated_pred = head(
            g_device, current_device, apply_component_gates=True
        )
        boot = target_head(
            gb.to(device), boot_current.to(device),
            apply_component_gates=False,
        )
        td_target = R.to(device) + gamma_k * boot
        mc_target = G_mc.to(device)
        td_loss = F.smooth_l1_loss(pred, td_target)
        mask_device = mc_mask.to(device)
        if bool(mask_device.any()):
            mc_loss = F.smooth_l1_loss(
                pred[mask_device], mc_target[mask_device]
            )
        else:
            mc_loss = pred.new_zeros(())

    p = pred.cpu().numpy()
    p_gated = gated_pred.cpu().numpy()
    y_td = td_target.cpu().numpy()
    mask_np = mc_mask.numpy().astype(bool)
    y_mc = G_mc.numpy()[mask_np]
    p_mc = p[mask_np]
    p_gated_mc = p_gated[mask_np]
    immediate = current.numpy()[mask_np]
    constant = np.broadcast_to(train_mc_mean, y_mc.shape)
    if len(y_mc):
        eligible_rows = (
            [row for row, keep in zip(rows, mask_np) if keep]
            if rows is not None else []
        )
        mc_components = _component_metrics(p_mc, y_mc)
        gated_mc_components = _component_metrics(p_gated_mc, y_mc)
        component_baselines = {
            "immediate_risk": _component_metrics(immediate, y_mc),
            "constant_train_mean": _component_metrics(constant, y_mc),
        }
        baselines = {
            "constant_train_mean_mc_mae": float(
                np.mean(np.abs(constant - y_mc))
            ),
            "immediate_risk_mc_mae": float(
                np.mean(np.abs(immediate - y_mc))
            ),
            "vhead_mc_mae": float(np.mean(np.abs(p_mc - y_mc))),
            "gated_vhead_mc_mae": float(
                np.mean(np.abs(p_gated_mc - y_mc))
            ),
        }
        danger = _danger_metrics(p_mc, y_mc, danger_threshold)
        direction = (
            _temporal_direction_metrics(
                p_mc, y_mc, eligible_rows, danger_threshold
            )
            if eligible_rows else {}
        )
        immediate_direction = (
            _temporal_direction_metrics(
                immediate, y_mc, eligible_rows, danger_threshold
            )
            if eligible_rows else {}
        )
        gated_direction = (
            _temporal_direction_metrics(
                p_gated_mc, y_mc, eligible_rows, danger_threshold
            )
            if eligible_rows else {}
        )
        validation_breakdowns = (
            _validation_breakdowns(
                p_mc, y_mc, immediate, eligible_rows
            )
            if eligible_rows else {"by_seed": {}, "by_arm": {}}
        )
    else:
        mc_components = {
            name: {"n": 0} for name in RISK_NAMES
        }
        gated_mc_components = {
            name: {"n": 0} for name in RISK_NAMES
        }
        component_baselines = {
            baseline: {name: {"n": 0} for name in RISK_NAMES}
            for baseline in ("immediate_risk", "constant_train_mean")
        }
        baselines = {
            "constant_train_mean_mc_mae": 0.0,
            "immediate_risk_mc_mae": 0.0,
            "vhead_mc_mae": 0.0,
            "gated_vhead_mc_mae": 0.0,
        }
        danger = {"n": 0, "threshold": float(danger_threshold)}
        direction = {}
        immediate_direction = {}
        gated_direction = {}
        validation_breakdowns = {"by_seed": {}, "by_arm": {}}
    return {
        "loss": float(td_loss + mc_weight * mc_loss),
        "td_huber": float(td_loss),
        "mc_huber": float(mc_loss),
        "mc_anchor_count": int(mask_np.sum()),
        "td": _component_metrics(p, y_td),
        "mc": mc_components,
        "gated_mc": gated_mc_components,
        "component_baselines": component_baselines,
        "baselines": baselines,
        "danger": danger,
        "mc_temporal_direction": direction,
        "immediate_risk_mc_temporal_direction": immediate_direction,
        "gated_mc_temporal_direction": gated_direction,
        "validation_breakdowns": validation_breakdowns,
        "component_gates": {
            name: bool(head.component_gates[index].item())
            for index, name in enumerate(RISK_NAMES)
        },
    }


def _expand_tuple_paths(values: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for value in values:
        # Shells normally expand globs; Path.glob fallback keeps quoted globs
        # convenient on Windows without introducing recursive surprises.
        if any(ch in value for ch in "*?"):
            parent = Path(value).parent
            matches = sorted(parent.glob(Path(value).name))
            paths.extend(str(path) for path in matches)
        else:
            paths.append(value)
    unique = list(dict.fromkeys(paths))
    if not unique:
        raise ValueError("--tuples matched no files")
    return unique


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tuples", nargs="+", required=True,
                        help="One or more td_stream_tuples_v1 files")
    parser.add_argument("--checkpoint", required=True,
                        help="Frozen Phase-C world-model checkpoint")
    parser.add_argument(
        "--allow-checkpoint-mismatch",
        action="store_true",
        help=("Allow a TD stream whose recorded source checkpoint is missing "
              "or has a different SHA256. Use only for an explicit latent-"
              "representation ablation."),
    )
    parser.add_argument("--output", required=True,
                        help="Additive TD risk V-head checkpoint")
    parser.add_argument("--expected-policy-family", default=None,
                        help="Fail unless all tuples carry this target policy")
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--residual-scale", type=float, default=1.0,
        help="Maximum absolute learned correction before clipping to [0,1]",
    )
    parser.add_argument("--mc-weight", type=float, default=0.5)
    parser.add_argument("--ema-tau", type=float, default=0.995)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--val-seeds", default=None,
                        help="Comma-separated held-out seeds, e.g. 209,210")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--early-stop-patience", type=int, default=7)
    parser.add_argument("--danger-quantile", type=float, default=0.8)
    parser.add_argument("--danger-weight", type=float, default=1.0,
                        help="Optional train loss multiplier for dangerous MC rows")
    parser.add_argument(
        "--gate-min-absolute-improvement", type=float, default=0.001,
        help=("Minimum held-out MC MAE improvement over immediate risk for "
              "each component"),
    )
    parser.add_argument(
        "--gate-min-relative-improvement", type=float, default=0.02,
        help=("Minimum relative held-out MC MAE improvement over immediate "
              "risk for each component"),
    )
    parser.add_argument(
        "--gate-min-unit-anchors", type=int, default=20,
        help=("Minimum complete MC anchors required in every held-out seed/run "
              "and every load-specific arm for a component gate to pass"),
    )
    parser.add_argument(
        "--gate-aggregate-only", action="store_true",
        help=("Do not require improvement in every held-out seed/run and "
              "load-specific arm. The default requires aggregate and "
              "per-unit improvement."),
    )
    parser.add_argument(
        "--gate-min-spearman", type=float, default=0.10,
        help="Minimum held-out MC rank correlation for every certified component",
    )
    parser.add_argument(
        "--gate-max-spearman-drop", type=float, default=0.05,
        help=("Maximum allowed Spearman drop relative to immediate risk on "
              "aggregate and validation units"),
    )
    parser.add_argument(
        "--gate-max-relative-bias", type=float, default=0.50,
        help=("Maximum absolute candidate bias divided by held-out target "
              "absolute mean"),
    )
    parser.add_argument(
        "--gate-min-std-ratio", type=float, default=0.10,
        help=("Minimum predicted/target MC standard-deviation ratio; rejects "
              "constant risk solutions"),
    )
    parser.add_argument(
        "--gate-max-rmse-relative-degrade", type=float, default=0.0,
        help=("Maximum relative RMSE degradation versus immediate risk; zero "
              "requires non-degradation"),
    )
    parser.add_argument(
        "--gate-variation-epsilon", type=float, default=1e-8,
        help=("Target standard deviations at or below this value are treated "
              "as constant, making rank/std checks not-applicable"),
    )
    parser.add_argument("--bottleneck-fraction", type=float, default=0.20)
    parser.add_argument(
        "--use-physical-summary",
        action="store_true",
        help=("Append the fixed 12-field analytic L0 summary to V inputs; "
              "requires streams collected with --td-stream-lyapunov-l0"),
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-runs", type=int, default=None,
                        help="Smoke-test limit across all tuple files")
    parser.add_argument("--max-segments-per-run", type=int, default=None,
                        help="Smoke-test limit; do not use for final training")
    args = parser.parse_args()

    if not 0.0 < args.gamma < 1.0:
        raise SystemExit("--gamma must lie in (0, 1)")
    if args.mc_weight < 0.0 or args.danger_weight <= 0.0:
        raise SystemExit("--mc-weight must be >= 0 and --danger-weight > 0")
    if args.residual_scale <= 0.0:
        raise SystemExit("--residual-scale must be positive")
    if (args.gate_min_absolute_improvement < 0.0
            or args.gate_min_relative_improvement < 0.0):
        raise SystemExit("component-gate improvement thresholds must be >= 0")
    if args.gate_min_unit_anchors <= 0:
        raise SystemExit("--gate-min-unit-anchors must be positive")
    if not -1.0 <= args.gate_min_spearman <= 1.0:
        raise SystemExit("--gate-min-spearman must lie in [-1, 1]")
    if args.gate_max_spearman_drop < 0.0:
        raise SystemExit("--gate-max-spearman-drop must be non-negative")
    if args.gate_max_relative_bias < 0.0 or args.gate_min_std_ratio < 0.0:
        raise SystemExit("component-gate bias/std thresholds must be non-negative")
    if args.gate_max_rmse_relative_degrade < 0.0:
        raise SystemExit("--gate-max-rmse-relative-degrade must be non-negative")
    if args.gate_variation_epsilon <= 0.0:
        raise SystemExit("--gate-variation-epsilon must be positive")
    if os.path.abspath(args.output) == os.path.abspath(args.checkpoint):
        raise SystemExit("--output must not overwrite the frozen base checkpoint")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tuple_paths = _expand_tuple_paths(args.tuples)
    first_payload = torch.load(tuple_paths[0], map_location="cpu", weights_only=False)
    example_frame, num_stations = _first_frame(first_payload)
    model, model_config = load_frozen_world_model(
        args.checkpoint, example_frame, num_stations, args.device
    )

    all_rows: List[dict] = []
    policy_families: set[str] = set()
    arm_labels: set[str] = set()
    K_values: set[int] = set()
    W_values: set[int] = set()
    run_count = 0
    checkpoint_provenance = []

    for path_index, path in enumerate(tuple_paths):
        payload = (first_payload if path_index == 0 else torch.load(
            path, map_location="cpu", weights_only=False
        ))
        runs = _payload_runs(payload)
        provenance = validate_stream_checkpoint(
            runs,
            args.checkpoint,
            allow_mismatch=args.allow_checkpoint_mismatch,
        )
        provenance["tuple_file"] = os.path.abspath(path)
        checkpoint_provenance.append(provenance)
        if provenance["metadata_missing"]:
            print(
                f"WARNING: {path} has no recorded source checkpoint; "
                "latent provenance could not be verified"
            )
        elif provenance["override_used"]:
            print(
                f"WARNING: checkpoint provenance override active for {path}"
            )
        K = int(payload["K"])
        K_values.add(K)
        W_values.add(int(payload["W"]))
        print(f"loading {path}: {len(runs)} runs, K={K}, "
              f"policy={payload.get('policy_family')}")
        for run in runs:
            if args.max_runs is not None and run_count >= args.max_runs:
                break
            family = str(run.get("policy_family", payload.get("policy_family", "unknown")))
            policy_families.add(family)
            arm_labels.add(str(run.get("arm_label", "unknown")))
            rows = _encode_run(
                model, run, K, args.gamma, args.bottleneck_fraction,
                args.use_physical_summary,
                args.device, args.max_segments_per_run,
            )
            all_rows.extend(rows)
            run_count += 1
            print(f"  encoded {run.get('run_id')}: {len(rows)} segments")
        del payload
        if args.max_runs is not None and run_count >= args.max_runs:
            break
    del first_payload

    if len(K_values) != 1:
        raise SystemExit(f"all tuple files must use one K, got {sorted(K_values)}")
    if len(W_values) != 1:
        raise SystemExit(f"all tuple files must use one W, got {sorted(W_values)}")
    if len(policy_families) != 1:
        raise SystemExit(
            "refusing mixed continuation-policy TD targets: "
            f"{sorted(policy_families)}"
        )
    policy_family = next(iter(policy_families))
    if (args.expected_policy_family is not None
            and policy_family != args.expected_policy_family):
        raise SystemExit(
            f"policy family {policy_family!r} != expected "
            f"{args.expected_policy_family!r}"
        )
    if len(all_rows) < 4:
        raise SystemExit("too few usable on-policy TD segments")

    K = next(iter(K_values))
    W = next(iter(W_values))
    val_seeds = _parse_int_set(args.val_seeds)
    train_rows, val_rows, split_manifest = _split_samples(
        all_rows, args.val_ratio, args.split_seed, val_seeds
    )
    print(f"dataset: train={len(train_rows)} val={len(val_rows)} "
          f"runs={run_count} policy={policy_family} K={K}")
    print(f"held out: {split_manifest['val_units']}")

    train_tensors = _stack(train_rows)
    val_tensors = _stack(val_rows)
    input_dim = int(train_tensors[0].shape[-1])
    latent_dim = int(model_config.get("hidden_dim", 64))
    demand_dim = latent_dim
    physical_summary_dim = (
        PHYSICAL_SUMMARY_DIM if args.use_physical_summary else 0
    )
    expected_input_dim = 7 * latent_dim + physical_summary_dim
    if input_dim != expected_input_dim:
        raise SystemExit(
            f"state signature dimension {input_dim} != expected {expected_input_dim}; "
            "check station/bottleneck/demand pooling"
        )

    head = TDRiskVHead(
        latent_dim=latent_dim,
        demand_dim=demand_dim,
        hidden_dim=args.hidden_dim,
        physical_summary_dim=physical_summary_dim,
        bottleneck_fraction=args.bottleneck_fraction,
        residual_scale=args.residual_scale,
    ).to(args.device)
    target_head = copy.deepcopy(head).to(args.device).eval()
    for parameter in target_head.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr)
    gamma_k = args.gamma ** K
    train_mc = train_tensors[3]
    train_mc_mask = train_tensors[6]
    eligible_train_mc = train_mc[train_mc_mask]
    if len(eligible_train_mc):
        train_mc_mean = eligible_train_mc.mean(dim=0).numpy()
        danger_threshold = float(torch.quantile(
            eligible_train_mc.max(dim=1).values, args.danger_quantile
        ))
    else:
        train_mc_mean = np.zeros(len(RISK_NAMES), dtype=np.float32)
        danger_threshold = 1.0
        if args.mc_weight > 0.0:
            print(
                "WARNING: no complete W-length MC anchors in training split; "
                "training the TD objective only"
            )

    # Epoch 0 is the exact immediate-risk baseline because both residual
    # heads are zero-initialised.  Keep it as a real checkpoint candidate so
    # an optimisation run that only makes held-out MC error worse fails
    # closed instead of saving the least-bad trained residual.
    baseline_metrics = _evaluate(
        head, target_head, val_tensors, gamma_k, args.mc_weight,
        args.device, danger_threshold, train_mc_mean, val_rows,
    )
    baseline_selection_metric = (
        "val_mc_mae"
        if baseline_metrics["mc_anchor_count"] > 0 else "val_loss"
    )
    baseline_row = {
        "epoch": 0,
        "train_loss": None,
        "train_td": None,
        "train_mc": None,
        "train_mc_anchor_batches": 0,
        "val_loss": baseline_metrics["loss"],
        "val_td_huber": baseline_metrics["td_huber"],
        "val_mc_huber": baseline_metrics["mc_huber"],
        "val_mc_mae": baseline_metrics["baselines"]["vhead_mc_mae"],
        "selection_metric": baseline_selection_metric,
    }
    baseline_row["selection_score"] = baseline_row[
        baseline_selection_metric
    ]
    best_score = baseline_row["selection_score"]
    best_head = copy.deepcopy(head.state_dict())
    best_target = copy.deepcopy(target_head.state_dict())
    patience = 0
    history = [baseline_row]
    n = len(train_rows)
    g_tr, gb_tr, R_tr, G_tr, current_tr, boot_current_tr, mc_tr = train_tensors
    danger_mask = (
        mc_tr & (G_tr.max(dim=1).values >= danger_threshold)
    )
    sample_weights = torch.ones(n, dtype=torch.float32)
    sample_weights[danger_mask] = args.danger_weight

    for epoch in range(1, args.epochs + 1):
        head.train()
        permutation = torch.randperm(n)
        total_loss = total_td = total_mc = 0.0
        batches = mc_batches = 0
        for start in range(0, n, args.batch_size):
            idx = permutation[start:start + args.batch_size]
            g = g_tr[idx].to(args.device)
            gb = gb_tr[idx].to(args.device)
            R = R_tr[idx].to(args.device)
            G_mc = G_tr[idx].to(args.device)
            current = current_tr[idx].to(args.device)
            boot_current = boot_current_tr[idx].to(args.device)
            mc_mask = mc_tr[idx].to(args.device)
            weights = sample_weights[idx].to(args.device)

            with torch.no_grad():
                y_td = R + gamma_k * target_head(
                    gb, boot_current, apply_component_gates=False
                )
            pred = head(g, current, apply_component_gates=False)
            td_each = F.smooth_l1_loss(pred, y_td, reduction="none").mean(dim=1)
            mc_each = F.smooth_l1_loss(pred, G_mc, reduction="none").mean(dim=1)
            mc_active = mc_mask.to(dtype=mc_each.dtype)
            loss = (
                (
                    td_each
                    + args.mc_weight * mc_each * mc_active
                ) * weights
            ).sum() / weights.sum()

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
            optimizer.step()
            with torch.no_grad():
                for target_param, online_param in zip(
                        target_head.parameters(), head.parameters()):
                    target_param.mul_(args.ema_tau).add_(
                        online_param, alpha=1.0 - args.ema_tau
                    )

            total_loss += float(loss)
            total_td += float(td_each.mean())
            if bool(mc_mask.any()):
                total_mc += float(mc_each[mc_mask].mean())
                mc_batches += 1
            batches += 1

        head.eval()
        metrics = _evaluate(
            head, target_head, val_tensors, gamma_k, args.mc_weight,
            args.device, danger_threshold, train_mc_mean, val_rows,
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(batches, 1),
            "train_td": total_td / max(batches, 1),
            "train_mc": total_mc / max(mc_batches, 1),
            "train_mc_anchor_batches": mc_batches,
            "val_loss": metrics["loss"],
            "val_td_huber": metrics["td_huber"],
            "val_mc_huber": metrics["mc_huber"],
            "val_mc_mae": metrics["baselines"]["vhead_mc_mae"],
        }
        row["selection_metric"] = (
            "val_mc_mae" if metrics["mc_anchor_count"] > 0 else "val_loss"
        )
        row["selection_score"] = row[row["selection_metric"]]
        history.append(row)
        print(
            f"epoch {epoch:03d} train={row['train_loss']:.6f} "
            f"val={row['val_loss']:.6f} td={row['val_td_huber']:.6f} "
            f"mc_mae={row['val_mc_mae']:.6f}"
        )

        if not np.isfinite(row["val_loss"]):
            raise SystemExit("validation loss diverged")
        if row["selection_score"] < best_score - 1e-7:
            best_score = row["selection_score"]
            best_head = copy.deepcopy(head.state_dict())
            best_target = copy.deepcopy(target_head.state_dict())
            patience = 0
        else:
            patience += 1
            if patience >= args.early_stop_patience:
                print(f"early stop at epoch {epoch}")
                break

    if best_head is None or best_target is None:
        raise SystemExit("training produced no finite checkpoint")
    head.load_state_dict(best_head)
    target_head.load_state_dict(best_target)
    head.eval()

    # Freeze deployment gates only after selecting the best residual model.
    # They are based exclusively on complete held-out MC anchors, never on TD
    # self-consistency alone.
    val_g, _, _, val_mc, val_current, _, val_mc_mask = val_tensors
    with torch.no_grad():
        val_candidate = head(
            val_g.to(args.device), val_current.to(args.device),
            apply_component_gates=False,
        ).cpu().numpy()
    val_mask_np = val_mc_mask.numpy().astype(bool)
    eligible_val_rows = [
        row for row, keep in zip(val_rows, val_mask_np) if keep
    ]
    gate_decisions = _component_gate_decisions(
        val_candidate[val_mask_np],
        val_mc.numpy()[val_mask_np],
        val_current.numpy()[val_mask_np],
        eligible_val_rows,
        min_absolute_improvement=args.gate_min_absolute_improvement,
        min_relative_improvement=args.gate_min_relative_improvement,
        require_all_units=not args.gate_aggregate_only,
        min_unit_anchors=args.gate_min_unit_anchors,
        danger_threshold=danger_threshold,
        min_spearman=args.gate_min_spearman,
        max_spearman_drop=args.gate_max_spearman_drop,
        max_relative_bias=args.gate_max_relative_bias,
        min_std_ratio=args.gate_min_std_ratio,
        max_rmse_relative_degrade=args.gate_max_rmse_relative_degrade,
        variation_epsilon=args.gate_variation_epsilon,
    )
    component_gates = [
        gate_decisions[name]["enabled"] for name in RISK_NAMES
    ]
    head.set_component_gates(component_gates)
    target_head.set_component_gates(component_gates)
    final_metrics = _evaluate(
        head, target_head, val_tensors, gamma_k, args.mc_weight,
        args.device, danger_threshold, train_mc_mean, val_rows,
    )
    final_metrics["gate_decisions"] = gate_decisions

    checkpoint = {
        "schema_version": HEAD_SCHEMA_VERSION,
        "gate_schema_version": COMPONENT_GATE_SCHEMA_VERSION,
        "online_ready": False,
        "online_blocker": "rollout_endpoint_demand_not_predicted",
        "td_risk_vhead": head.cpu().state_dict(),
        "td_risk_vhead_ema": target_head.cpu().state_dict(),
        "config": {
            "latent_dim": latent_dim,
            "demand_dim": demand_dim,
            "hidden_dim": args.hidden_dim,
            "residual_scale": args.residual_scale,
            "physical_summary_dim": physical_summary_dim,
            "physical_summary_names": (
                list(PHYSICAL_SUMMARY_NAMES)
                if physical_summary_dim else []
            ),
            "bottleneck_fraction": args.bottleneck_fraction,
            "input_dim": input_dim,
            "risk_components": list(RISK_NAMES),
            "value_semantics": (
                "current_raw_risk_plus_bounded_td_residual_with_"
                "component_fallback"
            ),
            "gamma": args.gamma,
            "K": K,
            "finite_mc_window_W": W,
            "mc_weight": args.mc_weight,
            "checkpoint_selection": (
                "held_out_complete_mc_mae_if_available_else_val_loss"
            ),
            "mc_anchor_excludes_right_censored": True,
            "train_mc_anchor_count": int(train_mc_mask.sum()),
            "val_mc_anchor_count": int(val_tensors[6].sum()),
            "component_gates": {
                name: bool(component_gates[index])
                for index, name in enumerate(RISK_NAMES)
            },
            "component_gate_rule": {
                "schema_version": COMPONENT_GATE_SCHEMA_VERSION,
                "baseline": "current_raw_risk",
                "metric": "held_out_complete_mc_multi_metric",
                "required_metrics": [
                    "mae_improvement",
                    "rmse_non_degradation",
                    "spearman_floor",
                    "spearman_preservation",
                    "relative_bias",
                    "prediction_std_ratio",
                ],
                "reported_diagnostics": [
                    "within_run_temporal_direction",
                    "danger_slice_temporal_direction",
                ],
                "min_absolute_improvement": (
                    args.gate_min_absolute_improvement
                ),
                "min_relative_improvement": (
                    args.gate_min_relative_improvement
                ),
                "require_all_validation_units": (
                    not args.gate_aggregate_only
                ),
                "validation_unit_kinds": ["seed_or_run", "arm_label"],
                "min_unit_anchors": args.gate_min_unit_anchors,
                "min_spearman": args.gate_min_spearman,
                "max_spearman_drop": args.gate_max_spearman_drop,
                "max_relative_bias": args.gate_max_relative_bias,
                "min_std_ratio": args.gate_min_std_ratio,
                "max_rmse_relative_degrade": (
                    args.gate_max_rmse_relative_degrade
                ),
                "variation_epsilon": args.gate_variation_epsilon,
                "danger_threshold": danger_threshold,
                "failure_fallback": "current_raw_risk",
            },
            "policy_family": policy_family,
            "arm_labels": sorted(arm_labels),
            "base_checkpoint": os.path.abspath(args.checkpoint),
            "checkpoint_provenance": checkpoint_provenance,
            "checkpoint_mismatch_override": args.allow_checkpoint_mismatch,
            "tuple_files": [os.path.abspath(path) for path in tuple_paths],
            "date": str(date.today()),
            "online_ready": False,
            "online_blocker": "rollout_endpoint_demand_not_predicted",
            "intended_use": "diagnostic_state_value_only",
        },
        "split_manifest": split_manifest,
        "history": history,
        "validation": final_metrics,
        "component_gate_decisions": gate_decisions,
    }
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(checkpoint, args.output)

    report_path = os.path.splitext(args.output)[0] + "_report.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump({
            "schema_version": HEAD_SCHEMA_VERSION,
            "gate_schema_version": COMPONENT_GATE_SCHEMA_VERSION,
            "config": checkpoint["config"],
            "split_manifest": split_manifest,
            "history": history,
            "validation": final_metrics,
            "component_gate_decisions": gate_decisions,
        }, handle, indent=2, ensure_ascii=False)

    print(f"saved V-head: {args.output}")
    print(f"report: {report_path}")
    print("validation MC component metrics:")
    for name in RISK_NAMES:
        metric = final_metrics["mc"][name]
        if metric.get("n") == 0:
            print(f"  {name:8s} no complete MC anchors")
        else:
            print(f"  {name:8s} mae={metric['mae']:.6f} "
                  f"spearman={metric['spearman']} "
                  f"bias={metric['bias']:+.6f}")
    print("component fallback gates:")
    for name in RISK_NAMES:
        decision = gate_decisions[name]
        aggregate = decision["aggregate"]
        if aggregate["n"]:
            print(
                f"  {name:8s} enabled={decision['enabled']} "
                f"candidate_mae={aggregate['candidate_mae']:.6f} "
                f"immediate_mae={aggregate['immediate_risk_mae']:.6f}"
            )
        else:
            print(f"  {name:8s} enabled=False no complete MC anchors")


if __name__ == "__main__":
    main()
