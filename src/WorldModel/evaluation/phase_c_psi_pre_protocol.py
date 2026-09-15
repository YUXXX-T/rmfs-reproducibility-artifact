"""Frozen protocol for the behavior-aligned ``psi_pre`` kill test.

This protocol deliberately sits *after* the frozen Phase-C behavior H=10
collection and *before* any online score change.  It asks only whether a
station-conditioned readout can recover endpoint-local congestion from the
latent produced by the frozen H=10 rollout::

    z_H_pred = WM.rollout(z_t, a, H=10).z_endpoint
    psi_pre  = phi_roll(z_H_pred)

The protocol does not claim that the resulting head is a Lyapunov function,
does not compute ``DeltaPsi`` and never changes ``Q(c,r)``.  Source seeds are
531--540; the historical 501--510 block is excluded by construction.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence


BASE_ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1"
)
BEHAVIOR_ROOT = BASE_ROOT / "behavior_h10_531_540_v1"
OUTPUT_ROOT = BASE_ROOT / "psi_pre_h10_531_540_v1"
MODEL_CHECKPOINT = BASE_ROOT / "model_round1_v1" / "best_regret_world_model.pt"

SEEDS = tuple(range(531, 541))
LOADS = ("low", "mid", "high")
HORIZON = 10
REGION_HOPS = 3

TARGET_CHANNELS = (
    "station_queue",
    "assigned_load",
    "region_density_mean",
    "region_density_max",
    "region_wait_mean",
    "region_wait_max",
    "region_blocked_mean",
    "region_blocked_max",
    "region_congestion_mean",
    "region_congestion_max",
)

SERVICE_CHANNELS = TARGET_CHANNELS[:2]
TRAFFIC_CHANNELS = TARGET_CHANNELS[2:]

DATASET_SCHEMA_VERSION = "phase_c_psi_pre_latent_dataset_v1"
TRAINING_SCHEMA_VERSION = "phase_c_psi_pre_training_v1"
REPORT_SCHEMA_VERSION = "phase_c_psi_pre_validation_v1"
SHARD_SCHEMA_VERSION = "phase_c_psi_pre_run_shard_v1"
FROZEN_BUNDLE_SCHEMA_VERSION = "phase_c_psi_pre_bundle_v1"


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_to_repo(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def parse_run_name(name: str) -> tuple[str, int]:
    match = re.fullmatch(r"(?:behavior_h10_)?(low|mid|high)_seed(\d+)", name)
    if match is None:
        raise ValueError(f"cannot parse behavior run name: {name}")
    return match.group(1), int(match.group(2))


def expected_run_files(behavior_root: Path) -> list[dict[str, Any]]:
    runs_root = behavior_root / "runs"
    if not runs_root.is_dir():
        raise FileNotFoundError(runs_root)
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(runs_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("."):
            continue
        load, seed = parse_run_name(run_dir.name)
        if seed not in SEEDS:
            continue
        data_path = run_dir / "behavior_h10_data.pt"
        meta_path = run_dir / "behavior_h10_data_meta.json"
        if not data_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"incomplete behavior run {run_dir}: expected data and metadata"
            )
        rows.append({
            "run_id": run_dir.name,
            "load": load,
            "seed": seed,
            "data_path": data_path,
            "meta_path": meta_path,
            "data_sha256": sha256_file(data_path),
            "meta_sha256": sha256_file(meta_path),
        })
    expected = {(load, seed) for load in LOADS for seed in SEEDS}
    observed = {(row["load"], row["seed"]) for row in rows}
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing or extra:
        raise RuntimeError(f"behavior source coverage mismatch: missing={missing} extra={extra}")
    return rows


def _pair(seed_pair_index: int) -> tuple[int, int]:
    start = 2 * int(seed_pair_index)
    return (SEEDS[start], SEEDS[start + 1])


def fold_seed_splits() -> list[dict[str, Any]]:
    """Five deterministic whole-seed folds.

    Each seed is a test seed exactly once and a validation seed exactly once;
    no candidate from one seed can leak into another split.
    """

    folds = []
    pairs = [_pair(index) for index in range(5)]
    for index, test_pair in enumerate(pairs):
        val_pair = pairs[(index + 1) % len(pairs)]
        train = [seed for seed in SEEDS if seed not in test_pair + val_pair]
        folds.append({
            "fold": index,
            "train_seeds": train,
            "val_seeds": list(val_pair),
            "test_seeds": list(test_pair),
        })
    return folds


def formal_protocol(
    *,
    behavior_root: Path = BEHAVIOR_ROOT,
    model_checkpoint: Path = MODEL_CHECKPOINT,
) -> dict[str, Any]:
    return {
        "schema_version": "phase_c_psi_pre_protocol_v1",
        "purpose": (
            "offline kill test for a behavior-aligned station-conditioned "
            "phi_roll(z_H_pred); no online policy or Q score change"
        ),
        "source": {
            "behavior_root": behavior_root.as_posix(),
            "collection_schema": "wm_v4_counterfactual_behavior_continuation",
            "seeds": list(SEEDS),
            "loads": list(LOADS),
            "horizon": HORIZON,
            "fresh_encode_per_sample": True,
            "continuation_mode": "behavior",
            "future_orders_and_assignments": True,
        },
        "latent_contract": {
            "input": "frozen encoder state z_t plus frozen action-conditioned transition",
            "endpoint": "z_H_pred returned by RMFSWorldModel.rollout(...)[5]",
            "horizon": HORIZON,
            "step_embedding_contract": "uses frozen checkpoint behavior; no extrapolation",
            "station_representation": {
                "kind": "shared_station_region_mean_max",
                "features": ["station_node", "region_mean", "region_max"],
                "region_seed": "station_node_ids from each behavior sample",
                "undirected_graph_hops": REGION_HOPS,
                "station_count_variable": True,
                "checkpoint_scene_station_count_fixed": True,
                "station_id_embedding": False,
            },
        },
        "target_contract": {
            "endpoint_offset": HORIZON,
            "source": "future_station_labels and future_node_labels",
            "channels": list(TARGET_CHANNELS),
            "station_channels": list(SERVICE_CHANNELS),
            "traffic_channels": list(TRAFFIC_CHANNELS),
            "node_channel_mapping": {
                "local_density": 1,
                "local_wait_pressure": 2,
                "local_blocked_pressure": 3,
                "congestion_score": 5,
            },
            "region_aggregation": "mean_and_max over station-centered h3 node set",
            "target_scale": "frozen dataset labels; no fit on test folds",
            "cross_channel_cancellation": False,
        },
        "baseline_contract": {
            "decoder_state_persistence": "frozen station/node decoders at z_t reused at H=10",
            "decoder_h10": "frozen station/node decoders at z_H_pred",
            "decoder_h1_persistence": "frozen first-step decoder output reused at H=10",
            "constant": "weighted train-fold channel mean",
        },
        "split_contract": {
            "kind": "five whole-seed folds",
            "folds": fold_seed_splits(),
            "group_leakage": False,
            "run_weighting": "equal total weight per source run; candidate groups equal within run",
        },
        "kill_gate": {
            "status": "exploratory_pre_registered",
            "service_pooled_spearman_min": 0.50,
            "service_run_cluster_ci95_lower_min": 0.30,
            "service_within_frame_spearman_min": 0.50,
            "traffic_max_channel_count_min": 3,
            "traffic_pooled_spearman_min": 0.40,
            "traffic_run_cluster_ci95_lower_min": 0.20,
            "beats_frozen_decoder_on_channels_min": 6,
            "online_connection": False,
        },
        "forbidden": {
            "seeds_501_510": True,
            "checkpoint_reselection": True,
            "encoder_or_transition_update": True,
            "delta_psi_or_q_change": True,
            "random_sample_split": True,
            "post_test_threshold_tuning": True,
        },
        "model_checkpoint": model_checkpoint.as_posix(),
    }


__all__ = [
    "BASE_ROOT",
    "BEHAVIOR_ROOT",
    "DATASET_SCHEMA_VERSION",
    "FROZEN_BUNDLE_SCHEMA_VERSION",
    "HORIZON",
    "LOADS",
    "MODEL_CHECKPOINT",
    "OUTPUT_ROOT",
    "REGION_HOPS",
    "REPORT_SCHEMA_VERSION",
    "SEEDS",
    "SERVICE_CHANNELS",
    "SHARD_SCHEMA_VERSION",
    "TARGET_CHANNELS",
    "TRAFFIC_CHANNELS",
    "TRAINING_SCHEMA_VERSION",
    "canonical_sha256",
    "expected_run_files",
    "fold_seed_splits",
    "formal_protocol",
    "parse_run_name",
    "sha256_file",
]
