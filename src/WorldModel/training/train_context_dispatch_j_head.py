"""Train a frozen-World-Model, context-level dispatch ranking head.

This script is deliberately isolated from every deployed/validated assigner.
It does not update the Phase-C World Model, the station congestion ``phi``
head, or the certified S1 robot scorer.  It only trains a small head for the
cross-context question:

    which dispatchable (order, pod, station) context should S1 score next?

Training labels must come from complete behavior-continuation counterfactual
rollouts.  Robot candidates belonging to one fixed context are collapsed to
the lowest true continuation composite cost (global system cost plus the
target station's queue/load trajectory), and contexts from the same run/tick
form one equal-weight ranking group.  The current analytic J expression is
reported as a baseline only; it is never used as a target.

The learned score keeps the requested semantics explicit:

    J(c) = w_service * service_phi
         + w_traffic * traffic_phi
         + w_work    * marginal_work
         - w_debt    * service_debt
         + bounded latent residual,

where all four semantic weights are constrained non-negative.  Lower J is
better.  ``service_debt`` here is station-local unserved-work pressure from
the frozen demand observation; no new e_demand/encoder schema is introduced.

Outputs are development artifacts only.  Online integration remains disabled
until a held-out closed-loop ablation validates the trained checkpoint.
"""

from __future__ import annotations

import argparse
import copy
import glob
import hashlib
import json
import math
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Config.config_loader import load_config
from WorldModel.core.costs import DEFAULT_LAMBDAS, compute_realized_cost
from WorldModel.core.station_congestion_head import CHANNEL_NAMES
from WorldModel.data.build_station_congestion_head_dataset import (
    build_station_representations,
)
from WorldModel.evaluation.station_congestion_endpoint import (
    build_station_layout,
    load_frozen_station_head,
)
from WorldModel.graph.graph_builder import build_static_graph
from WorldModel.training.train_td_risk_v import load_frozen_world_model
from WorldState.world import WorldState


HEAD_SCHEMA_VERSION = "context_dispatch_j_head_v1"
TRAINING_SCHEMA_VERSION = "context_dispatch_j_training_v1"
VALIDATION_SCHEMA_VERSION = "context_dispatch_j_validation_v1"
SEMANTIC_CHANNELS = (
    "service_pressure",
    "traffic_pressure",
    "marginal_work",
    "service_debt",
)
EXPLICIT_FEATURES = (
    "best_robot_to_pod_distance",
    "pod_to_station_distance",
    "station_to_return_distance",
    "station_queue_pressure",
    "best_route_pressure",
    "order_size",
)
TARGET_GAMMA = 0.95
TARGET_LOCAL_STATION_LAMBDAS = (1.0, 1.0)

DEFAULT_BASE = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1"
)
DEFAULT_DATA_ROOT = DEFAULT_BASE / "behavior_h10_531_540_v1"
DEFAULT_WORLD_MODEL = (
    DEFAULT_BASE / "model_round1_v1/best_regret_world_model.pt"
)
DEFAULT_PHI_HEAD = (
    DEFAULT_BASE
    / "station_congestion_head_region_dev_511_520_v1"
    / "linear_head_v1/best_station_congestion_head.pt"
)
DEFAULT_LAYOUT_CONFIG = Path("Config/world_model_config_PP_48_mid.json")

# These blocks already have a frozen evaluation role in the current project.
# Refuse accidental fitting on them instead of relying on a comment/filename.
PROTECTED_SEEDS = frozenset(range(501, 511)) | frozenset(range(551, 571))


def _inverse_softplus(value: float) -> float:
    value = float(value)
    if value <= 0.0:
        raise ValueError("softplus initial value must be positive")
    return math.log(math.expm1(value))


class ContextDispatchJHead(nn.Module):
    """Semantically constrained context cost with a bounded latent residual."""

    schema_version = HEAD_SCHEMA_VERSION
    semantic_channels = SEMANTIC_CHANNELS

    def __init__(
        self,
        feature_dim: int,
        *,
        hidden_dim: int = 128,
        dropout: float = 0.05,
        residual_limit: float = 0.25,
    ) -> None:
        super().__init__()
        if int(feature_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("feature_dim and hidden_dim must be positive")
        if not 0.0 <= float(dropout) < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if float(residual_limit) < 0.0:
            raise ValueError("residual_limit must be non-negative")
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.residual_limit = float(residual_limit)

        self.residual_net = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, 1),
        )
        # Initial values preserve the analytic sign contract while leaving all
        # magnitudes trainable.  service_debt is subtracted in forward().
        initial = [0.5, 0.5, 0.10, 1.0]
        self.raw_semantic_weights = nn.Parameter(torch.tensor(
            [_inverse_softplus(value) for value in initial],
            dtype=torch.float32,
        ))

    def semantic_weights(self) -> torch.Tensor:
        return F.softplus(self.raw_semantic_weights) + 1e-6

    def forward(
        self,
        features: torch.Tensor,
        semantic: torch.Tensor,
    ) -> torch.Tensor:
        if features.ndim != 2 or features.size(1) != self.feature_dim:
            raise ValueError(
                f"features must be (B, {self.feature_dim}), "
                f"got {tuple(features.shape)}"
            )
        if semantic.ndim != 2 or semantic.size(1) != len(SEMANTIC_CHANNELS):
            raise ValueError(
                "semantic must have shape "
                f"(B, {len(SEMANTIC_CHANNELS)})"
            )
        weights = self.semantic_weights()
        base = (
            weights[0] * semantic[:, 0]
            + weights[1] * semantic[:, 1]
            + weights[2] * semantic[:, 2]
            - weights[3] * semantic[:, 3]
        )
        residual = self.residual_limit * torch.tanh(
            self.residual_net(features).squeeze(-1)
        )
        return base + residual

    def checkpoint_config(self) -> dict[str, Any]:
        return {
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "residual_limit": self.residual_limit,
            "semantic_channels": list(SEMANTIC_CHANNELS),
        }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(
        _json_safe(payload), indent=2, ensure_ascii=False
    ) + "\n"
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _atomic_torch_save(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    try:
        torch.save(dict(payload), tmp_name)
        os.replace(tmp_name, path)
    finally:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)


def _seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _resolve_device(raw: str) -> torch.device:
    if raw == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(raw)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def _expand_data_paths(values: Sequence[str]) -> list[Path]:
    result: list[Path] = []
    for raw in values:
        matches = [Path(value) for value in glob.glob(raw, recursive=True)]
        if not matches:
            path = Path(raw)
            matches = [path] if path.exists() else []
        for path in matches:
            if path.is_dir():
                found = sorted(path.rglob("behavior_h10_data.pt"))
                if not found:
                    found = sorted(path.rglob("data.pt"))
                result.extend(found)
            elif path.is_file() and path.suffix == ".pt":
                result.append(path)
    unique = sorted({path.resolve() for path in result})
    if not unique:
        raise FileNotFoundError("no .pt behavior-continuation datasets found")
    return unique


def _load_raw_samples(path: Path) -> list[dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    rows = payload.get("samples") if isinstance(payload, Mapping) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path}: expected a non-empty sample list")
    if not all(isinstance(row, Mapping) for row in rows):
        raise ValueError(f"{path}: sample payload is not a list of mappings")
    return [dict(row) for row in rows]


def _sample_seed(sample: Mapping[str, Any], path: Path) -> int:
    value = sample.get("simulation_seed")
    if value is None:
        raise ValueError(f"{path}: simulation_seed is required for leakage-safe split")
    return int(value)


def _load_name(sample: Mapping[str, Any], path: Path) -> str:
    value = sample.get("source_load_level") or sample.get("load_level")
    if value is None:
        for token in ("low", "mid", "high"):
            if token in path.parts or f"_{token}_" in str(sample.get("run_id", "")):
                return token
        raise ValueError(f"{path}: cannot determine load level")
    value = str(value).lower()
    if value not in {"low", "mid", "high"}:
        raise ValueError(f"{path}: unexpected load level {value!r}")
    return value


def _validate_candidate(
    sample: Mapping[str, Any],
    *,
    path: Path,
    horizon: int,
    station_ids: Sequence[int],
) -> dict[str, float]:
    required = {
        "run_id",
        "simulation_seed",
        "decision_tick",
        "candidate_group_id",
        "node_history",
        "edge_index",
        "edge_features",
        "demand_context",
        "action_global",
        "station_node_ids",
        "fixed_context",
        "future_system_labels",
        "future_station_labels",
        "future_mask",
        "realized_cost",
    }
    missing = sorted(required - set(sample))
    if missing:
        raise ValueError(f"{path}: sample missing required fields {missing}")
    if str(sample.get("rollout_continuation_mode")) != "behavior":
        raise ValueError(
            f"{path}: context-J training requires behavior continuation; "
            f"got {sample.get('rollout_continuation_mode')!r}"
        )
    if str(sample.get("action_type", "assign_robot")) == "no_assign":
        raise ValueError(f"{path}: NO_ASSIGN rows are not context-J robot candidates")
    labels = torch.as_tensor(sample["future_system_labels"], dtype=torch.float32)
    mask = torch.as_tensor(sample["future_mask"], dtype=torch.float32).flatten()
    if labels.ndim != 2 or labels.size(1) != len(DEFAULT_LAMBDAS):
        raise ValueError(f"{path}: future_system_labels must have shape (H, 7)")
    if labels.size(0) < int(horizon) or mask.numel() < int(horizon):
        raise ValueError(
            f"{path}: requested H={horizon} exceeds stored label length"
        )
    if not bool((mask[:horizon] > 0.5).all()):
        raise ValueError(f"{path}: right-censored H={horizon} label is forbidden")
    labels = labels[:horizon]
    if not bool(torch.isfinite(labels).all()):
        raise ValueError(f"{path}: future labels contain NaN/Inf")
    recomputed = compute_realized_cost(labels, lambdas=list(DEFAULT_LAMBDAS))
    stored = float(sample["realized_cost"])
    if not math.isfinite(stored):
        raise ValueError(f"{path}: realized_cost is not finite")
    if abs(recomputed - stored) > 1e-4 * max(1.0, abs(stored)):
        raise ValueError(
            f"{path}: realized_cost does not match its future labels "
            f"({stored} vs {recomputed})"
        )
    station_labels = torch.as_tensor(
        sample["future_station_labels"], dtype=torch.float32
    )
    if (
        station_labels.ndim != 3
        or station_labels.size(0) < int(horizon)
        or station_labels.size(1) != len(station_ids)
        or station_labels.size(2) != len(TARGET_LOCAL_STATION_LAMBDAS)
    ):
        raise ValueError(
            f"{path}: future_station_labels must have shape "
            f"(H, {len(station_ids)}, 2)"
        )
    station_labels = station_labels[:horizon]
    if not bool(torch.isfinite(station_labels).all()):
        raise ValueError(f"{path}: future station labels contain NaN/Inf")
    station_id = _fixed_context_key(sample)[2]
    try:
        station_offset = tuple(int(value) for value in station_ids).index(station_id)
    except ValueError as exc:
        raise ValueError(f"{path}: target station {station_id} is absent") from exc
    discounts = torch.tensor(
        [TARGET_GAMMA ** step for step in range(int(horizon))],
        dtype=torch.float32,
    )
    local_weights = torch.tensor(
        TARGET_LOCAL_STATION_LAMBDAS, dtype=torch.float32
    )
    local_station_cost = float((
        discounts
        * (station_labels[:, station_offset, :] * local_weights).sum(dim=-1)
    ).sum().item())
    # The global system cost alone is nearly tied across H=10 contexts.  The
    # station-local physical endpoint/trajectory term supplies the spatial
    # distinction J(c) is meant to learn, without inventing a pseudo-label.
    return {
        "system_cost": float(recomputed),
        "local_station_cost": local_station_cost,
        "target_cost": float(recomputed + local_station_cost),
    }


def _fixed_context_key(sample: Mapping[str, Any]) -> tuple[int, int, int]:
    context = sample.get("fixed_context")
    if not isinstance(context, Mapping):
        raise ValueError("fixed_context must be a mapping")
    return (
        int(context["order_id"]),
        int(context["pod_id"]),
        int(context["station_id"]),
    )


def _frame_key(sample: Mapping[str, Any]) -> tuple[str, int, int]:
    return (
        str(sample["run_id"]),
        int(sample["simulation_seed"]),
        int(sample["decision_tick"]),
    )


def _context_group_key(sample: Mapping[str, Any]) -> tuple[Any, ...]:
    return _frame_key(sample) + (
        str(sample["candidate_group_id"]),
    ) + _fixed_context_key(sample)


def _same_state(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    tensor_fields = (
        "node_history",
        "edge_index",
        "edge_features",
        "demand_context",
        "station_node_ids",
    )
    for name in tensor_fields:
        a = torch.as_tensor(left[name])
        b = torch.as_tensor(right[name])
        if a.shape != b.shape or not torch.equal(a, b):
            return False
    return True


def _aggregate_action_global(members: Sequence[Mapping[str, Any]]) -> torch.Tensor:
    values = torch.stack([
        torch.as_tensor(row["action_global"], dtype=torch.float32).flatten()
        for row in members
    ])
    if values.ndim != 2 or values.size(1) != 6:
        raise ValueError("action_global must have shape (6,)")
    # Robot-dependent terms use the best candidate proxy.  The remaining four
    # terms are context-invariant by construction and use a median audit-safe
    # reducer in case of harmless floating serialization noise.
    return torch.stack((
        values[:, 0].min(),
        values[:, 1].median(),
        values[:, 2].median(),
        values[:, 3].median(),
        values[:, 4].min(),
        values[:, 5].median(),
    ))


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _pearson(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 2 or right.size != left.size:
        return float("nan")
    left = left - left.mean()
    right = right - right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    if denominator <= 1e-15:
        return float("nan")
    return float((left @ right) / denominator)


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    return _pearson(_rankdata(left), _rankdata(right))


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    repeats: int = 2000,
) -> list[float]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return [float("nan"), float("nan")]
    if finite.size == 1:
        return [float(finite[0]), float(finite[0])]
    rng = np.random.default_rng(int(seed))
    draws = rng.choice(finite, size=(int(repeats), finite.size), replace=True)
    return [float(value) for value in np.quantile(draws.mean(axis=1), [0.025, 0.975])]


def _fit_feature_contract(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    values = torch.stack([row["features_raw"] for row in rows]).double()
    mean = values.mean(dim=0)
    std = values.std(dim=0, unbiased=False).clamp_min(1e-6)
    return {
        "kind": "train_only_standard_score",
        "mean": mean.float(),
        "std": std.float(),
    }


def _fit_semantic_contract(
    rows: Sequence[Mapping[str, Any]],
    *,
    service_debt_unit_range: bool = False,
) -> dict[str, Any]:
    values = torch.stack([row["semantic_raw"] for row in rows]).double()
    lower = torch.quantile(values, 0.05, dim=0)
    upper = torch.quantile(values, 0.95, dim=0)
    if service_debt_unit_range:
        debt_index = SEMANTIC_CHANNELS.index("service_debt")
        lower = lower.clone()
        upper = upper.clone()
        lower[debt_index] = 0.0
        upper[debt_index] = 1.0
    span = upper - lower
    constant = span <= 1e-9
    minimum = values.min(dim=0).values
    maximum = values.max(dim=0).values
    lower = torch.where(constant, minimum, lower)
    upper = torch.where(constant, maximum, upper)
    span = upper - lower
    constant = span <= 1e-9
    span = torch.where(constant, torch.ones_like(span), span)
    return {
        "kind": (
            "train_only_quantile_with_service_debt_unit_range"
            if service_debt_unit_range
            else "train_only_clipped_quantile_range"
        ),
        "lower_quantile": 0.05,
        "upper_quantile": 0.95,
        "channels": list(SEMANTIC_CHANNELS),
        "service_debt_unit_range": bool(service_debt_unit_range),
        "lower": lower.float(),
        "upper": upper.float(),
        "span": span.float(),
        "constant": constant.bool(),
    }


def _normalise_rows(
    rows: Sequence[dict[str, Any]],
    *,
    feature_contract: Mapping[str, Any],
    semantic_contract: Mapping[str, Any],
) -> None:
    feature_mean = torch.as_tensor(feature_contract["mean"], dtype=torch.float32)
    feature_std = torch.as_tensor(feature_contract["std"], dtype=torch.float32)
    lower = torch.as_tensor(semantic_contract["lower"], dtype=torch.float32)
    span = torch.as_tensor(semantic_contract["span"], dtype=torch.float32)
    constant = torch.as_tensor(semantic_contract["constant"], dtype=torch.bool)
    for row in rows:
        row["features"] = (row["features_raw"] - feature_mean) / feature_std
        value = ((row["semantic_raw"] - lower) / span).clamp(0.0, 1.0)
        row["semantic"] = torch.where(constant, torch.zeros_like(value), value)


def _split_rows(
    rows: Sequence[dict[str, Any]],
    *,
    train_seeds: set[int],
    val_seeds: set[int],
    test_seeds: set[int],
) -> dict[str, list[dict[str, Any]]]:
    result = {"train": [], "val": [], "test": []}
    for row in rows:
        seed = int(row["seed"])
        if seed in train_seeds:
            result["train"].append(row)
        elif seed in val_seeds:
            result["val"].append(row)
        elif seed in test_seeds:
            result["test"].append(row)
    for name, values in result.items():
        if not values:
            raise ValueError(f"{name} split contains zero context rows")
    return result


def _group_frames(rows: Sequence[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["frame_key"]].append(row)
    frames = []
    for key in sorted(grouped):
        members = grouped[key]
        if len(members) < 2:
            continue
        if len({row["context_key"] for row in members}) != len(members):
            raise ValueError(f"duplicate context row inside frame {key}")
        frames.append(members)
    if not frames:
        raise ValueError("split contains no multi-context ranking frame")
    return frames


def _pairwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float,
    temperature: float,
) -> tuple[torch.Tensor, int]:
    left, right = torch.triu_indices(
        target.numel(), target.numel(), offset=1, device=target.device
    )
    difference = target[right] - target[left]
    valid = difference.abs() > float(epsilon)
    if not bool(valid.any()):
        return prediction.sum() * 0.0, 0
    left = left[valid]
    right = right[valid]
    # target[right] > target[left] means left is better (lower cost), so the
    # desired prediction margin is prediction[right] - prediction[left] > 0.
    direction = torch.sign(target[right] - target[left])
    margin = direction * (prediction[right] - prediction[left])
    loss = F.softplus(-margin / float(temperature)).mean()
    return loss, int(valid.sum().item())


def _frame_loss(
    head: ContextDispatchJHead,
    frame: Sequence[Mapping[str, Any]],
    *,
    device: torch.device,
    epsilon: float,
    pairwise_temperature: float,
    listwise_temperature: float,
    pairwise_weight: float,
    listwise_weight: float,
    regression_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    features = torch.stack([row["features"] for row in frame]).to(device)
    semantic = torch.stack([row["semantic"] for row in frame]).to(device)
    target = torch.tensor(
        [float(row["target_cost"]) for row in frame],
        dtype=torch.float32,
        device=device,
    )
    prediction = head(features, semantic)
    pairwise, pairs = _pairwise_loss(
        prediction,
        target,
        epsilon=epsilon,
        temperature=pairwise_temperature,
    )
    target_distribution = torch.softmax(
        -target / float(listwise_temperature), dim=0
    )
    predicted_log_distribution = torch.log_softmax(
        -prediction / float(listwise_temperature), dim=0
    )
    listwise = -(target_distribution * predicted_log_distribution).sum()
    target_scale = (target.max() - target.min()).clamp_min(float(epsilon))
    target_norm = (target - target.mean()) / target_scale
    prediction_norm = (prediction - prediction.mean()) / target_scale
    regression = F.smooth_l1_loss(prediction_norm, target_norm)
    total = (
        float(pairwise_weight) * pairwise
        + float(listwise_weight) * listwise
        + float(regression_weight) * regression
    )
    return total, {
        "pairwise": float(pairwise.detach()),
        "listwise": float(listwise.detach()),
        "regression": float(regression.detach()),
        "pairs": float(pairs),
    }


def _pair_concordance(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    epsilon: float,
) -> tuple[int, int]:
    correct = 0
    total = 0
    for left in range(target.size):
        for right in range(left + 1, target.size):
            target_difference = float(target[right] - target[left])
            if abs(target_difference) <= float(epsilon):
                continue
            prediction_difference = float(prediction[right] - prediction[left])
            correct += int(target_difference * prediction_difference > 0.0)
            total += 1
    return correct, total


def _evaluate_score(
    head: ContextDispatchJHead,
    frames: Sequence[Sequence[Mapping[str, Any]]],
    *,
    device: torch.device,
    epsilon: float,
    bootstrap_seed: int,
) -> dict[str, Any]:
    head.eval()
    frame_rows = []
    per_load: dict[str, list[dict[str, float]]] = defaultdict(list)
    with torch.no_grad():
        for frame in frames:
            features = torch.stack([row["features"] for row in frame]).to(device)
            semantic = torch.stack([row["semantic"] for row in frame]).to(device)
            prediction = head(features, semantic).cpu().numpy()
            target = np.asarray(
                [float(row["target_cost"]) for row in frame], dtype=np.float64
            )
            # Fixed semantic-only comparison.  It is a baseline, not a label.
            semantic_np = semantic.cpu().numpy()
            baseline = (
                0.5 * semantic_np[:, 0]
                + 0.5 * semantic_np[:, 1]
                + 0.1 * semantic_np[:, 2]
                - semantic_np[:, 3]
            )
            learned_correct, pair_count = _pair_concordance(
                prediction, target, epsilon=epsilon
            )
            baseline_correct, baseline_pairs = _pair_concordance(
                baseline, target, epsilon=epsilon
            )
            target_best = float(np.min(target))
            learned_index = int(np.argmin(prediction))
            baseline_index = int(np.argmin(baseline))
            random_top1 = float(np.mean(target <= target_best + epsilon))
            row = {
                "contexts": float(len(frame)),
                "pairs": float(pair_count),
                "learned_pair_concordance": (
                    learned_correct / pair_count if pair_count else float("nan")
                ),
                "baseline_pair_concordance": (
                    baseline_correct / baseline_pairs
                    if baseline_pairs else float("nan")
                ),
                "learned_spearman": _spearman(prediction, target),
                "baseline_spearman": _spearman(baseline, target),
                "learned_top1": float(target[learned_index] <= target_best + epsilon),
                "baseline_top1": float(target[baseline_index] <= target_best + epsilon),
                "random_top1": random_top1,
            }
            frame_rows.append(row)
            per_load[str(frame[0]["load"])].append(row)

    def aggregate(values: Sequence[Mapping[str, float]], seed: int) -> dict[str, Any]:
        def metric(name: str) -> dict[str, Any]:
            rows = [
                float(row[name]) for row in values
                if math.isfinite(float(row[name]))
            ]
            return {
                "mean": float(np.mean(rows)) if rows else float("nan"),
                "median": float(np.median(rows)) if rows else float("nan"),
                "ci95": _bootstrap_mean_ci(rows, seed=seed),
            }

        return {
            "frames": len(values),
            "contexts": int(sum(row["contexts"] for row in values)),
            "non_tie_pairs": int(sum(row["pairs"] for row in values)),
            "learned_pair_concordance": metric("learned_pair_concordance"),
            "baseline_pair_concordance": metric("baseline_pair_concordance"),
            "learned_spearman": metric("learned_spearman"),
            "baseline_spearman": metric("baseline_spearman"),
            "learned_top1": metric("learned_top1"),
            "baseline_top1": metric("baseline_top1"),
            "random_top1": metric("random_top1"),
        }

    overall = aggregate(frame_rows, bootstrap_seed)
    by_load = {
        load: aggregate(values, bootstrap_seed + offset + 1)
        for offset, (load, values) in enumerate(sorted(per_load.items()))
    }
    pair_value = float(overall["learned_pair_concordance"]["mean"])
    top1_value = float(overall["learned_top1"]["mean"])
    # A small split can legitimately contain only target ties at epsilon.
    # Pairwise concordance is then undefined rather than failed; select on the
    # remaining finite metric instead of silently producing no checkpoint.
    selection_terms = []
    if math.isfinite(pair_value):
        selection_terms.append((0.7, pair_value))
    if math.isfinite(top1_value):
        selection_terms.append((0.3, top1_value))
    selection_score = (
        sum(weight * value for weight, value in selection_terms)
        / sum(weight for weight, _ in selection_terms)
        if selection_terms else float("-inf")
    )
    return {
        "overall": overall,
        "by_load": by_load,
        "checkpoint_selection_score": selection_score,
        "baseline_definition": (
            "0.5*service_phi + 0.5*traffic_phi + "
            "0.1*marginal_work - service_debt; comparison only"
        ),
    }


def _extract_compact_rows(
    *,
    data_paths: Sequence[Path],
    model,
    model_config: Mapping[str, Any],
    phi_head,
    phi_payload: Mapping[str, Any],
    layout,
    expected_edge_index: torch.Tensor,
    device: torch.device,
    horizon: int,
    allowed_seeds: set[int],
    max_files: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    source_manifest = []
    file_counts = []
    skipped_single_context_frames = 0
    selected_paths = list(data_paths[:max_files] if max_files else data_paths)
    station_to_offset = {
        int(station_id): offset
        for offset, station_id in enumerate(layout.station_ids)
    }
    representation = str(phi_payload["representation"]["name"])
    service_index = int(CHANNEL_NAMES.index("service"))
    traffic_index = int(CHANNEL_NAMES.index("traffic"))

    for file_index, path in enumerate(selected_paths, start=1):
        raw = _load_raw_samples(path)
        first_seed = _sample_seed(raw[0], path)
        if first_seed not in allowed_seeds:
            continue
        if any(_sample_seed(sample, path) != first_seed for sample in raw):
            raise ValueError(f"{path}: one data file contains multiple seeds")
        load = _load_name(raw[0], path)

        contexts: dict[
            tuple[Any, ...], list[tuple[dict[str, Any], dict[str, float]]]
        ] = defaultdict(list)
        frames: dict[tuple[str, int, int], list[tuple[Any, ...]]] = defaultdict(list)
        for sample in raw:
            cost = _validate_candidate(
                sample,
                path=path,
                horizon=horizon,
                station_ids=layout.station_ids,
            )
            group_key = _context_group_key(sample)
            contexts[group_key].append((sample, cost))
        for group_key in contexts:
            frames[group_key[:3]].append(group_key)

        file_contexts = 0
        file_frames = 0
        for frame_key in sorted(frames):
            group_keys = frames[frame_key]
            if len(group_keys) < 2:
                skipped_single_context_frames += 1
                continue
            representatives = [contexts[key][0][0] for key in group_keys]
            state_sample = representatives[0]
            if any(not _same_state(state_sample, sample) for sample in representatives[1:]):
                raise ValueError(f"{path}: contexts in frame {frame_key} do not share state")
            edge_index = torch.as_tensor(state_sample["edge_index"], dtype=torch.long)
            if not torch.equal(edge_index.cpu(), expected_edge_index.cpu()):
                raise ValueError(f"{path}: dataset graph differs from --layout-config")
            station_node_ids = torch.as_tensor(
                state_sample["station_node_ids"], dtype=torch.long
            ).flatten()
            if tuple(int(value) for value in station_node_ids.tolist()) != tuple(
                int(value) for value in layout.station_node_ids
            ):
                raise ValueError(f"{path}: station node ordering differs from layout")
            node_history = torch.as_tensor(
                state_sample["node_history"], dtype=torch.float32, device=device
            )
            edge_features = torch.as_tensor(
                state_sample["edge_features"], dtype=torch.float32, device=device
            )
            demand_context = torch.as_tensor(
                state_sample["demand_context"], dtype=torch.float32, device=device
            ).flatten()
            expected_demand_dim = 5 + len(layout.station_ids)
            if demand_context.numel() != expected_demand_dim:
                raise ValueError(
                    f"{path}: demand dimension {demand_context.numel()} != "
                    f"5 + station_count ({expected_demand_dim})"
                )
            with torch.no_grad():
                z, e_demand, _ = model.encode_state(
                    node_history,
                    edge_index.to(device),
                    edge_features,
                    demand_context,
                )
                station_representation = build_station_representations(
                    z,
                    layout.station_node_ids,
                    representation=representation,
                    station_region_node_ids=layout.station_region_node_ids,
                )
                phi = phi_head.forward_station_latents(station_representation)
                global_representation = torch.cat((
                    z.mean(dim=0), z.amax(dim=0), e_demand.flatten()
                ))

            for group_key in group_keys:
                members_with_cost = contexts[group_key]
                members = [item[0] for item in members_with_cost]
                candidate_costs = [
                    float(item[1]["target_cost"]) for item in members_with_cost
                ]
                if len(members) < 2:
                    raise ValueError(f"{path}: context group {group_key} has <2 robots")
                context_keys = {_fixed_context_key(sample) for sample in members}
                if len(context_keys) != 1:
                    raise ValueError(f"{path}: candidate group mixes fixed contexts")
                context_key = next(iter(context_keys))
                station_id = int(context_key[2])
                if station_id not in station_to_offset:
                    raise ValueError(f"{path}: unknown station id {station_id}")
                station_offset = station_to_offset[station_id]
                action = _aggregate_action_global(members)
                service = float(phi[station_offset, service_index].item())
                traffic = float(phi[station_offset, traffic_index].item())
                marginal_work = float(
                    0.5 * action[1] + 0.3 * action[2] + 0.2 * action[5]
                )
                service_debt = float(demand_context[5 + station_offset].item())
                explicit = action.detach().cpu()
                latent_feature = torch.cat((
                    station_representation[station_offset].detach().cpu(),
                    global_representation.detach().cpu(),
                    explicit,
                )).float()
                semantic = torch.tensor(
                    [service, traffic, marginal_work, service_debt],
                    dtype=torch.float32,
                )
                rows.append({
                    "frame_key": frame_key,
                    "context_key": context_key,
                    "run_id": frame_key[0],
                    "seed": int(frame_key[1]),
                    "tick": int(frame_key[2]),
                    "load": load,
                    "station_id": station_id,
                    "candidate_count": len(members),
                    "candidate_cost_min": float(min(candidate_costs)),
                    "candidate_cost_mean": float(np.mean(candidate_costs)),
                    "candidate_cost_std": float(np.std(candidate_costs)),
                    "target_cost": float(min(candidate_costs)),
                    "system_cost_min": float(min(
                        item[1]["system_cost"] for item in members_with_cost
                    )),
                    "local_station_cost_min": float(min(
                        item[1]["local_station_cost"] for item in members_with_cost
                    )),
                    "features_raw": latent_feature,
                    "semantic_raw": semantic,
                })
                file_contexts += 1
            file_frames += 1

        source_manifest.append({
            "path": path.as_posix(),
            "sha256": _sha256_file(path),
            "seed": first_seed,
            "load": load,
        })
        file_counts.append({
            "path": path.as_posix(),
            "samples": len(raw),
            "multi_context_frames": file_frames,
            "context_rows": file_contexts,
        })
        print(
            f"[encode {file_index}/{len(selected_paths)}] "
            f"{path.parent.name} seed={first_seed} load={load} "
            f"frames={file_frames} contexts={file_contexts}",
            flush=True,
        )
        del raw, contexts, frames

    if not rows:
        raise ValueError("no compact context rows were extracted")
    feature_dims = {int(row["features_raw"].numel()) for row in rows}
    if len(feature_dims) != 1:
        raise ValueError(f"context feature dimension changed: {feature_dims}")
    audit = {
        "source_files": source_manifest,
        "source_file_counts": file_counts,
        "source_file_count": len(source_manifest),
        "context_rows": len(rows),
        "feature_dim": next(iter(feature_dims)),
        "skipped_single_context_frames": skipped_single_context_frames,
        "model_hidden_dim": int(model_config.get("hidden_dim", 64)),
        "continuation_mode": "behavior",
        "horizon": int(horizon),
        "target": (
            "minimum true behavior-continuation composite cost among robot "
            "candidates: global DEFAULT_LAMBDAS system cost plus discounted "
            "target-station queue+assigned-load trajectory"
        ),
        "current_J_used_as_label": False,
    }
    return rows, audit


def _frame_counts(frames: Sequence[Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    seeds = sorted({int(frame[0]["seed"]) for frame in frames})
    by_load = defaultdict(lambda: {"frames": 0, "contexts": 0})
    for frame in frames:
        load = str(frame[0]["load"])
        by_load[load]["frames"] += 1
        by_load[load]["contexts"] += len(frame)
    return {
        "seeds": seeds,
        "frames": len(frames),
        "contexts": sum(len(frame) for frame in frames),
        "by_load": dict(sorted(by_load.items())),
    }


def train(
    *,
    data_paths: Sequence[Path],
    checkpoint: Path,
    phi_head_checkpoint: Path,
    layout_config: Path,
    output_root: Path,
    device: torch.device,
    torch_threads: int,
    train_seeds: set[int],
    val_seeds: set[int],
    test_seeds: set[int],
    horizon: int,
    hidden_dim: int,
    dropout: float,
    residual_limit: float,
    epochs: int,
    patience: int,
    group_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip: float,
    epsilon: float,
    pairwise_temperature: float,
    listwise_temperature: float,
    pairwise_weight: float,
    listwise_weight: float,
    regression_weight: float,
    seed: int,
    max_files: int | None,
    service_debt_unit_range: bool = False,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(
            f"output directory already exists; refusing overwrite: {output_root}"
        )
    for path in (checkpoint, phi_head_checkpoint, layout_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    split_sets = [train_seeds, val_seeds, test_seeds]
    if any(not values for values in split_sets):
        raise ValueError("train/val/test seed lists must all be non-empty")
    if any(split_sets[i] & split_sets[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("train/val/test seed lists must be disjoint")
    used_seeds = set().union(*split_sets)
    protected = sorted(used_seeds & set(PROTECTED_SEEDS))
    if protected:
        raise ValueError(
            "protected evaluation seeds cannot be used for fitting or checkpoint "
            f"selection: {protected}"
        )

    _seed_everything(seed)
    try:
        torch.set_num_threads(max(1, int(torch_threads)))
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    config = load_config(str(layout_config))
    world = WorldState(config)
    expected_edge_index, node_map, *_ = build_static_graph(world.map_state)
    layout = build_station_layout(world, node_map, expected_edge_index)

    first_samples = _load_raw_samples(data_paths[0])
    example = first_samples[0]
    model, model_config = load_frozen_world_model(
        str(checkpoint), example, len(layout.station_ids), str(device)
    )
    phi_head, phi_payload = load_frozen_station_head(
        phi_head_checkpoint,
        model_checkpoint_path=checkpoint,
        device=device,
    )
    del first_samples

    rows, extraction_audit = _extract_compact_rows(
        data_paths=data_paths,
        model=model,
        model_config=model_config,
        phi_head=phi_head,
        phi_payload=phi_payload,
        layout=layout,
        expected_edge_index=expected_edge_index,
        device=device,
        horizon=horizon,
        allowed_seeds=used_seeds,
        max_files=max_files,
    )
    split_rows = _split_rows(
        rows,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        test_seeds=test_seeds,
    )
    feature_contract = _fit_feature_contract(split_rows["train"])
    semantic_contract = _fit_semantic_contract(
        split_rows["train"],
        service_debt_unit_range=service_debt_unit_range,
    )
    _normalise_rows(
        rows,
        feature_contract=feature_contract,
        semantic_contract=semantic_contract,
    )
    split_frames = {
        name: _group_frames(values) for name, values in split_rows.items()
    }
    del model, phi_head
    if device.type == "cuda":
        torch.cuda.empty_cache()

    feature_dim = int(rows[0]["features"].numel())
    head = ContextDispatchJHead(
        feature_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        residual_limit=residual_limit,
    ).to(device)
    optimiser = torch.optim.AdamW(
        head.parameters(), lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    staging = output_root.parent / f".{output_root.name}.tmp.{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir(parents=True)
    log_path = staging / "train_log.jsonl"

    best_score = float("-inf")
    best_epoch = 0
    best_state = None
    stale = 0
    history = []
    rng = random.Random(seed)
    for epoch in range(1, int(epochs) + 1):
        train_frames = list(split_frames["train"])
        rng.shuffle(train_frames)
        head.train()
        total_losses = []
        components: dict[str, list[float]] = defaultdict(list)
        for offset in range(0, len(train_frames), int(group_batch_size)):
            batch = train_frames[offset:offset + int(group_batch_size)]
            losses = []
            stats = []
            for frame in batch:
                frame_loss, frame_stats = _frame_loss(
                    head,
                    frame,
                    device=device,
                    epsilon=epsilon,
                    pairwise_temperature=pairwise_temperature,
                    listwise_temperature=listwise_temperature,
                    pairwise_weight=pairwise_weight,
                    listwise_weight=listwise_weight,
                    regression_weight=regression_weight,
                )
                losses.append(frame_loss)
                stats.append(frame_stats)
            loss = torch.stack(losses).mean()
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), float(grad_clip))
            optimiser.step()
            total_losses.append(float(loss.detach()))
            for row in stats:
                for name, value in row.items():
                    components[name].append(float(value))

        validation = _evaluate_score(
            head,
            split_frames["val"],
            device=device,
            epsilon=epsilon,
            bootstrap_seed=seed + epoch,
        )
        score = float(validation["checkpoint_selection_score"])
        log_row = {
            "epoch": epoch,
            "train_loss": float(np.mean(total_losses)),
            "train_components": {
                name: float(np.mean(values)) if values else None
                for name, values in sorted(components.items())
            },
            "val_selection_score": score,
            "val_pair_concordance": validation["overall"][
                "learned_pair_concordance"
            ]["mean"],
            "val_top1": validation["overall"]["learned_top1"]["mean"],
            "semantic_weights": {
                name: float(value)
                for name, value in zip(
                    SEMANTIC_CHANNELS,
                    head.semantic_weights().detach().cpu().tolist(),
                )
            },
        }
        history.append(log_row)
        with log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(_json_safe(log_row), ensure_ascii=False) + "\n")
        print(
            f"[epoch {epoch:03d}] train={log_row['train_loss']:.6f} "
            f"val_pair={log_row['val_pair_concordance']:.4f} "
            f"val_top1={log_row['val_top1']:.4f}",
            flush=True,
        )
        if score > best_score + 1e-7:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy({
                name: value.detach().cpu() for name, value in head.state_dict().items()
            })
            stale = 0
        else:
            stale += 1
            if stale >= int(patience):
                break

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    head.load_state_dict(best_state, strict=True)
    head.to(device)
    evaluations = {
        name: _evaluate_score(
            head,
            frames,
            device=device,
            epsilon=epsilon,
            bootstrap_seed=seed + 1000 + offset,
        )
        for offset, (name, frames) in enumerate(split_frames.items())
    }
    semantic_weights = {
        name: float(value)
        for name, value in zip(
            SEMANTIC_CHANNELS,
            head.semantic_weights().detach().cpu().tolist(),
        )
    }
    signal_thresholds = {"train": 50, "val": 20, "test": 20}
    signal_audit = {}
    for split_name, report in evaluations.items():
        pair_count = int(report["overall"]["non_tie_pairs"])
        signal_audit[split_name] = {
            "non_tie_pairs": pair_count,
            "minimum_recommended_pairs": signal_thresholds[split_name],
            "sufficient_for_selection": pair_count >= signal_thresholds[split_name],
        }
    signal_audit["all_splits_sufficient"] = all(
        bool(row["sufficient_for_selection"])
        for name, row in signal_audit.items()
        if name != "all_splits_sufficient"
    )
    if not signal_audit["all_splits_sufficient"]:
        print(
            "[warning] context ranking signal is sparse; see "
            "validation.json signal_audit before using the checkpoint online",
            flush=True,
        )

    feature_contract_serialisable = {
        "kind": feature_contract["kind"],
        "mean": feature_contract["mean"],
        "std": feature_contract["std"],
        "composition": {
            "station_representation": phi_payload["representation"],
            "global_state": ["z_mean", "z_max", "e_demand_embedding"],
            "explicit_features": list(EXPLICIT_FEATURES),
        },
    }
    semantic_contract_serialisable = dict(semantic_contract)
    checkpoint_payload = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "head_schema_version": HEAD_SCHEMA_VERSION,
        "development_only": True,
        "online_ready": False,
        "online_blocker": "held-out closed-loop Dynamic-J validation not run",
        "head_config": head.checkpoint_config(),
        "state_dict": best_state,
        "semantic_weights": semantic_weights,
        "feature_contract": feature_contract_serialisable,
        "semantic_contract": semantic_contract_serialisable,
        "target_contract": {
            "rollout_continuation_mode": "behavior",
            "horizon": int(horizon),
            "system_label_lambdas": list(DEFAULT_LAMBDAS),
            "robot_aggregation": "minimum true composite cost within fixed context",
            "global_term": "DEFAULT_LAMBDAS discounted system labels",
            "local_term": {
                "name": "target_station_queue_plus_assigned_load",
                "gamma": TARGET_GAMMA,
                "weights": list(TARGET_LOCAL_STATION_LAMBDAS),
            },
            "ranking_group": "all distinct contexts from same run/tick",
            "lower_score_is_better": True,
            "current_analytic_j_is_not_a_label": True,
        },
        "base_world_model": {
            "path": checkpoint.as_posix(),
            "sha256": _sha256_file(checkpoint),
            "model_config": dict(model_config),
            "frozen": True,
        },
        "station_phi": {
            "path": phi_head_checkpoint.as_posix(),
            "sha256": _sha256_file(phi_head_checkpoint),
            "schema_version": phi_payload.get("schema_version"),
            "representation": phi_payload.get("representation"),
            "scale_contract_sha256": phi_payload.get("scale_contract_sha256"),
            "frozen": True,
        },
        "layout": {
            "config": layout_config.as_posix(),
            "config_sha256": _sha256_file(layout_config),
            **layout.to_dict(),
        },
        "split_seeds": {
            "train": sorted(train_seeds),
            "val": sorted(val_seeds),
            "test": sorted(test_seeds),
        },
        "training": {
            "seed": int(seed),
            "epochs_requested": int(epochs),
            "epochs_completed": len(history),
            "best_epoch": int(best_epoch),
            "best_val_selection_score": float(best_score),
            "optimizer": "AdamW",
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "group_batch_size": int(group_batch_size),
            "patience": int(patience),
            "loss_weights": {
                "pairwise": float(pairwise_weight),
                "listwise": float(listwise_weight),
                "regression": float(regression_weight),
            },
            "pairwise_temperature": float(pairwise_temperature),
            "listwise_temperature": float(listwise_temperature),
            "epsilon": float(epsilon),
            "grad_clip": float(grad_clip),
            "service_debt_unit_range": bool(service_debt_unit_range),
        },
        "signal_audit": signal_audit,
        "audit": {
            "encoder_frozen": True,
            "transition_frozen": True,
            "decoder_frozen": True,
            "station_phi_frozen": True,
            "s1_robot_scorer_changed": False,
            "e_demand_schema_changed": False,
            "no_assign_added": False,
            "protected_501_510_used": False,
            "protected_551_570_used": False,
            "behavior_continuation_labels_required": True,
            "group_balanced_training": True,
            "service_debt_physical_range_used": bool(service_debt_unit_range),
            "online_connection_allowed": False,
        },
    }
    checkpoint_path = staging / "best_context_j_head.pt"
    _atomic_torch_save(checkpoint_path, checkpoint_payload)

    validation = {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "development_only": True,
        "checkpoint": checkpoint_path.name,
        "checkpoint_selection": (
            "max validation 0.7*mean_frame_pair_concordance + "
            "0.3*mean_frame_top1"
        ),
        "best_epoch": int(best_epoch),
        "semantic_weights": semantic_weights,
        "signal_audit": signal_audit,
        "splits": evaluations,
        "interpretation": {
            "learned_quantity": "cross-context dispatch cost J(c)",
            "within_context_robot_ranking": "unchanged frozen S1",
            "service_and_traffic": "frozen station phi outputs",
            "service_debt": "station-local unserved-work pressure",
            "service_debt_normalisation": (
                "fixed physical [0,1]"
                if service_debt_unit_range
                else "train-only 5%-95% quantile"
            ),
            "marginal_work": "robot-aggregated route/work proxy",
            "bounded_residual": float(residual_limit),
            "long_horizon_claim": False,
            "online_policy_changed": False,
        },
    }
    validation_path = staging / "validation.json"
    _atomic_json(validation_path, validation)

    summary = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "development_only": True,
        "device": str(device),
        "data": extraction_audit,
        "split_counts": {
            name: _frame_counts(frames) for name, frames in split_frames.items()
        },
        "best_epoch": int(best_epoch),
        "best_val_selection_score": float(best_score),
        "semantic_weights": semantic_weights,
        "signal_audit": signal_audit,
        "outputs": {
            "checkpoint": checkpoint_path.name,
            "validation": validation_path.name,
            "train_log": log_path.name,
        },
        "audit": checkpoint_payload["audit"],
    }
    summary_path = staging / "train_summary.json"
    _atomic_json(summary_path, summary)

    manifest_lines = []
    for path in sorted(staging.iterdir()):
        if path.is_file() and path.name != "trained_outputs.sha256":
            manifest_lines.append(f"{_sha256_file(path)}  {path.name}")
    (staging / "trained_outputs.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="utf-8", newline="\n"
    )
    staging.rename(output_root)
    print(f"[complete] context-J head: {output_root}", flush=True)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        nargs="+",
        default=[str(DEFAULT_DATA_ROOT)],
        help=(
            "behavior-continuation .pt files, directories, or glob patterns; "
            "directories are searched recursively"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_WORLD_MODEL)
    parser.add_argument(
        "--phi-head-checkpoint", type=Path, default=DEFAULT_PHI_HEAD
    )
    parser.add_argument("--layout-config", type=Path, default=DEFAULT_LAYOUT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--torch-threads", type=int, default=8)
    parser.add_argument("--train-seeds", nargs="+", type=int, default=list(range(531, 537)))
    parser.add_argument("--val-seeds", nargs="+", type=int, default=[537, 538])
    parser.add_argument("--test-seeds", nargs="+", type=int, default=[539, 540])
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--residual-limit", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--group-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--pairwise-temperature", type=float, default=0.25)
    parser.add_argument("--listwise-temperature", type=float, default=0.50)
    parser.add_argument("--pairwise-weight", type=float, default=1.0)
    parser.add_argument("--listwise-weight", type=float, default=0.50)
    parser.add_argument("--regression-weight", type=float, default=0.25)
    parser.add_argument(
        "--service-debt-unit-range",
        action="store_true",
        help=(
            "normalise the physically bounded service_debt channel with "
            "[0,1] instead of the train 5%%-95%% quantile; other semantic "
            "channels retain the train-only quantile contract"
        ),
    )
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="debug-only prefix limit; omit for real training",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    positive_ints = (
        args.torch_threads,
        args.horizon,
        args.hidden_dim,
        args.epochs,
        args.patience,
        args.group_batch_size,
    )
    if any(int(value) <= 0 for value in positive_ints):
        raise SystemExit("threads/horizon/dim/epochs/patience/batch must be positive")
    if args.max_files is not None and int(args.max_files) <= 0:
        raise SystemExit("--max-files must be positive")
    if not 0.0 <= float(args.dropout) < 1.0:
        raise SystemExit("--dropout must be in [0, 1)")
    strictly_positive = (
        args.learning_rate,
        args.grad_clip,
        args.epsilon,
        args.pairwise_temperature,
        args.listwise_temperature,
    )
    if any(float(value) <= 0.0 for value in strictly_positive):
        raise SystemExit("learning rate/clip/epsilon/temperatures must be positive")
    non_negative = (
        args.weight_decay,
        args.residual_limit,
        args.pairwise_weight,
        args.listwise_weight,
        args.regression_weight,
    )
    if any(float(value) < 0.0 for value in non_negative):
        raise SystemExit("weight decay/residual/loss weights must be non-negative")
    if (
        float(args.pairwise_weight)
        + float(args.listwise_weight)
        + float(args.regression_weight)
        <= 0.0
    ):
        raise SystemExit("at least one loss weight must be positive")

    device = _resolve_device(args.device)
    data_paths = _expand_data_paths(args.data)
    print(f"[device] {device}", flush=True)
    print(f"[data] files={len(data_paths)}", flush=True)
    train(
        data_paths=data_paths,
        checkpoint=args.checkpoint,
        phi_head_checkpoint=args.phi_head_checkpoint,
        layout_config=args.layout_config,
        output_root=args.output_root,
        device=device,
        torch_threads=args.torch_threads,
        train_seeds=set(args.train_seeds),
        val_seeds=set(args.val_seeds),
        test_seeds=set(args.test_seeds),
        horizon=args.horizon,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        residual_limit=args.residual_limit,
        epochs=args.epochs,
        patience=args.patience,
        group_batch_size=args.group_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        grad_clip=args.grad_clip,
        epsilon=args.epsilon,
        pairwise_temperature=args.pairwise_temperature,
        listwise_temperature=args.listwise_temperature,
        pairwise_weight=args.pairwise_weight,
        listwise_weight=args.listwise_weight,
        regression_weight=args.regression_weight,
        service_debt_unit_range=args.service_debt_unit_range,
        seed=args.seed,
        max_files=args.max_files,
    )


if __name__ == "__main__":
    main()
