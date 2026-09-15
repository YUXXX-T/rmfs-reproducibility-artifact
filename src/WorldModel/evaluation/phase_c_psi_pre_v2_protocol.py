"""Frozen protocol for the corrected, traffic-deduplicated ``psi_pre`` v2.

This is an offline readout/evaluation protocol only.  It reuses the frozen
Phase-C behavior-continuation H=10 source block and checkpoint, but gives the
readout an explicit physical channel contract:

* service pressure: ``station_queue`` and ``assigned_load``;
* primary traffic pressure: station-region ``density_mean`` and ``density_max``;
* diagnostic traffic: ``blocked_max`` (reported, never part of the primary
  penalty or an online Q score).

The v1 ten-channel result is intentionally not overwritten.  v2 also freezes
the correction that applies sigmoid to NodeDecoder BCE-logit channels before
region aggregation; this affects only decoder baselines, not the frozen WM or
the behavior labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from WorldModel.evaluation.phase_c_psi_pre_protocol import (
    BASE_ROOT,
    BEHAVIOR_ROOT,
    FROZEN_BUNDLE_SCHEMA_VERSION,
    HORIZON,
    LOADS,
    MODEL_CHECKPOINT,
    REGION_HOPS,
    SEEDS,
    TARGET_CHANNELS as V1_TARGET_CHANNELS,
    canonical_sha256,
    expected_run_files,
    fold_seed_splits,
    relative_to_repo,
    sha256_file,
)


OUTPUT_ROOT = BASE_ROOT / "psi_pre_h10_531_540_v2"
FROZEN_BUNDLE_SCHEMA_VERSION_V2 = "phase_c_psi_pre_bundle_v2"
PROTOCOL_SCHEMA_VERSION = "phase_c_psi_pre_protocol_v2"
DATASET_SCHEMA_VERSION = "phase_c_psi_pre_latent_dataset_v2"
REPORT_SCHEMA_VERSION = "phase_c_psi_pre_validation_v2"

# The source v1 target order is retained in the materialised shards.  The v2
# projection uses only these columns, so no simulator recollection is needed.
TARGET_SOURCE_INDICES = (0, 1, 2, 3, 7)
TARGET_CHANNELS = (
    "station_queue",
    "assigned_load",
    "region_density_mean",
    "region_density_max",
    "region_blocked_max",
)
SERVICE_CHANNELS = TARGET_CHANNELS[:2]
TRAFFIC_PRIMARY_CHANNELS = TARGET_CHANNELS[2:4]
TRAFFIC_DIAGNOSTIC_CHANNELS = (TARGET_CHANNELS[4],)
PRIMARY_CHANNELS = SERVICE_CHANNELS + TRAFFIC_PRIMARY_CHANNELS

NODE_BCE_LOGIT_COLUMNS = (0, 2, 3)


def formal_protocol(
    *,
    behavior_root: Path = BEHAVIOR_ROOT,
    model_checkpoint: Path = MODEL_CHECKPOINT,
    source_v1_root: Path = BASE_ROOT / "psi_pre_h10_531_540_v1",
) -> dict[str, Any]:
    """Return the immutable v2 contract.

    ``blocked_max`` remains a supervised diagnostic output so that rare severe
    obstruction is visible, but it is explicitly excluded from ``PRIMARY``
    channels and from every online connection/gate in this protocol.
    """

    return {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "purpose": (
            "correctness rerun of behavior-aligned phi_roll(z_H_pred); "
            "deduplicate traffic channels and fix decoder-logit baseline"
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
            "station_representation": {
                "kind": "shared_station_region_mean_max",
                "features": ["station_node", "region_mean", "region_max"],
                "undirected_graph_hops": REGION_HOPS,
                "station_count_variable": True,
                "station_id_embedding": False,
            },
        },
        "channel_contract": {
            "source_v1_channels": list(V1_TARGET_CHANNELS),
            "source_indices": list(TARGET_SOURCE_INDICES),
            "channels": list(TARGET_CHANNELS),
            "service_channels": list(SERVICE_CHANNELS),
            "traffic_primary_channels": list(TRAFFIC_PRIMARY_CHANNELS),
            "traffic_diagnostic_channels": list(TRAFFIC_DIAGNOSTIC_CHANNELS),
            "primary_channels": list(PRIMARY_CHANNELS),
            "blocked_max_role": "diagnostic_only_no_primary_penalty_no_Q_connection",
            "blocked_max_readout": "trained_and_reported_for_diagnosis",
            "blocked_max_primary_penalty_weight": 0.0,
            "scalar_psi_aggregation": "not performed; retain named vector channels",
            "cross_channel_cancellation": False,
        },
        "baseline_contract": {
            "decoder_outputs": "NodeDecoder channels 0,2,3 are logits",
            "bce_logit_columns_sigmoid_before_region_aggregation": list(
                NODE_BCE_LOGIT_COLUMNS
            ),
            "station_decoder_channels": "unchanged continuous outputs",
            "historical_v1_raw_logit_baseline": "not used for v2 fairness metrics",
        },
        "split_contract": {
            "kind": "five whole-seed folds",
            "folds": fold_seed_splits(),
            "group_leakage": False,
            "run_weighting": "equal total weight per source run; candidate groups equal within run",
        },
        "diagnostic_bars": {
            "service_pooled_spearman_min": 0.50,
            "service_run_cluster_ci95_lower_min": 0.30,
            "service_within_frame_spearman_min": 0.50,
            "traffic_primary_pooled_spearman_min": 0.40,
            "traffic_primary_run_cluster_ci95_lower_min": 0.20,
            "traffic_primary_required_count": 2,
            "blocked_max_has_no_pass_bar": True,
            "online_connection_allowed": False,
        },
        "forbidden": {
            "seeds_501_510": True,
            "checkpoint_reselection": True,
            "encoder_or_transition_update": True,
            "delta_psi_or_q_change": True,
            "random_sample_split": True,
            "post_test_threshold_tuning": True,
        },
        "source_v1_root": source_v1_root.as_posix(),
        "model_checkpoint": model_checkpoint.as_posix(),
    }


__all__ = [
    "BASE_ROOT",
    "BEHAVIOR_ROOT",
    "DATASET_SCHEMA_VERSION",
    "FROZEN_BUNDLE_SCHEMA_VERSION",
    "FROZEN_BUNDLE_SCHEMA_VERSION_V2",
    "HORIZON",
    "LOADS",
    "MODEL_CHECKPOINT",
    "NODE_BCE_LOGIT_COLUMNS",
    "OUTPUT_ROOT",
    "PRIMARY_CHANNELS",
    "PROTOCOL_SCHEMA_VERSION",
    "REGION_HOPS",
    "REPORT_SCHEMA_VERSION",
    "SEEDS",
    "SERVICE_CHANNELS",
    "TARGET_CHANNELS",
    "TARGET_SOURCE_INDICES",
    "TRAFFIC_DIAGNOSTIC_CHANNELS",
    "TRAFFIC_PRIMARY_CHANNELS",
    "canonical_sha256",
    "expected_run_files",
    "fold_seed_splits",
    "formal_protocol",
    "relative_to_repo",
    "sha256_file",
]
