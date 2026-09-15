"""Offline paired control evaluation for the old and new context-J heads.

Both checkpoints are evaluated on exactly the same frozen 571--590 compact
context rows.  The World Model, station-phi head, labels, grouping, and
epsilon are identical; only the J-head checkpoint and its own train-only
normalization contract differ.  This is an offline control and never changes
the online dispatcher.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from Config.config_loader import load_config
from WorldState.world import WorldState
from WorldModel.evaluation.station_congestion_endpoint import (
    build_station_layout,
    load_frozen_station_head,
)
from WorldModel.graph.graph_builder import build_static_graph
from WorldModel.training.train_td_risk_v import load_frozen_world_model
from WorldModel.training import train_context_dispatch_j_head as train_mod


BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DATA_ROOT = BASE_ROOT / "context_j_behavior_h10_571_590_v1"
LABEL_AUDIT = DATA_ROOT / "validation/context_j_label_audit.json"
LABEL_AUDIT_SHA = DATA_ROOT / "validation/validated_outputs.sha256"
WORLD_MODEL = BASE_ROOT / "model_round1_v1/best_regret_world_model.pt"
DEFAULT_PHI_HEAD = (
    BASE_ROOT
    / "station_congestion_head_region_dev_511_520_v1"
    / "linear_head_v1/best_station_congestion_head.pt"
)
LAYOUT_CONFIG = Path("Config/world_model_config_PP_48_mid.json")
OLD_HEAD = (
    BASE_ROOT
    / "context_dispatch_j_head_train_531_540_v1/best_context_j_head.pt"
)
NEW_HEAD = (
    BASE_ROOT
    / "context_dispatch_j_head_train_571_590_v1/best_context_j_head.pt"
)
OUTPUT_ROOT = BASE_ROOT / "context_j_head_control_571_590_v1"

SEEDS = tuple(range(571, 591))
TRAIN_SEEDS = set(range(571, 583))
VAL_SEEDS = set(range(583, 587))
TEST_SEEDS = set(range(587, 591))
LOADS = ("low", "mid", "high")
HORIZON = 10
EPSILON = 0.01
PROTOCOL_SHA256 = "external-fingerprint-omitted"
EXPECTED_HEAD_SCHEMA = "context_dispatch_j_head_v1"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _safe(value.item())
    if isinstance(value, torch.Tensor):
        return _safe(value.detach().cpu().tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_label_audit(
    data_root: Path,
    audit_path: Path,
    audit_sha_path: Path,
) -> dict[str, Any]:
    if not audit_path.is_file() or not audit_sha_path.is_file():
        raise FileNotFoundError("label audit JSON/SHA256 is required")
    manifest = {}
    for line in audit_sha_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value, name = line.split(None, 1)
            manifest[name.strip()] = value
    relative = audit_path.relative_to(data_root).as_posix()
    if manifest.get(relative) != sha256_file(audit_path):
        raise ValueError("label audit SHA256 mismatch")
    audit = _load_json(audit_path)
    if audit.get("protocol_sha256") != PROTOCOL_SHA256:
        raise ValueError("unexpected context-J label protocol")
    if audit.get("expected_runs") != 60 or audit.get("completed_runs") != 60:
        raise ValueError("context-J source collection is incomplete")
    if not bool(audit.get("gates", {}).get("all_gates_passed")):
        raise ValueError("context-J label quality gate is not passed")
    for split, minimum in {"train": 150, "val": 40, "test": 40}.items():
        count = int(audit["signal_audit"][split]["non_tie_context_pairs"])
        if count < minimum:
            raise ValueError(f"{split} non-tie count {count} < {minimum}")
    return audit


def _load_head(
    path: Path,
    *,
    device: torch.device,
    world_model_hash: str,
    phi_hash: str,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("head_schema_version") != EXPECTED_HEAD_SCHEMA:
        raise ValueError(f"{path}: incompatible head schema")
    config = payload.get("head_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"{path}: missing head_config")
    expected_channels = list(train_mod.SEMANTIC_CHANNELS)
    if list(config.get("semantic_channels", ())) != expected_channels:
        raise ValueError(f"{path}: semantic channel contract changed")
    base = payload.get("base_world_model") or {}
    station = payload.get("station_phi") or {}
    if base.get("sha256") != world_model_hash:
        raise ValueError(f"{path}: trained against a different World Model")
    if station.get("sha256") != phi_hash:
        raise ValueError(f"{path}: trained against a different station phi head")
    target = payload.get("target_contract") or {}
    if int(target.get("horizon", -1)) != HORIZON:
        raise ValueError(f"{path}: target horizon is not H=10")
    if str(target.get("rollout_continuation_mode")) != "behavior":
        raise ValueError(f"{path}: target continuation is not behavior")
    head = train_mod.ContextDispatchJHead(
        int(config["feature_dim"]),
        hidden_dim=int(config["hidden_dim"]),
        dropout=float(config["dropout"]),
        residual_limit=float(config["residual_limit"]),
    ).to(device)
    head.load_state_dict(payload["state_dict"], strict=True)
    head.eval()
    for parameter in head.parameters():
        parameter.requires_grad_(False)
    return head, dict(payload)


def _normalised_copy(
    rows: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
) -> list[dict[str, Any]]:
    copied = [dict(row) for row in rows]
    train_mod._normalise_rows(
        copied,
        feature_contract=payload["feature_contract"],
        semantic_contract=payload["semantic_contract"],
    )
    return copied


def _split_frames(
    rows: Sequence[dict[str, Any]],
) -> dict[str, list[list[dict[str, Any]]]]:
    split_rows = train_mod._split_rows(
        rows,
        train_seeds=TRAIN_SEEDS,
        val_seeds=VAL_SEEDS,
        test_seeds=TEST_SEEDS,
    )
    return {
        name: train_mod._group_frames(values)
        for name, values in split_rows.items()
    }


def _frame_key(frame: Sequence[Mapping[str, Any]]) -> tuple[Any, ...]:
    return tuple(frame[0]["frame_key"])


def _frame_pair_stats(
    prediction: np.ndarray,
    target: np.ndarray,
) -> dict[str, Any]:
    correct, pairs = train_mod._pair_concordance(
        prediction, target, epsilon=EPSILON
    )
    target_best = float(np.min(target))
    best_index = int(np.argmin(prediction))
    return {
        "correct": int(correct),
        "pairs": int(pairs),
        "pair_concordance": float(correct / pairs) if pairs else float("nan"),
        "spearman": float(train_mod._spearman(prediction, target)),
        "top1": float(target[best_index] <= target_best + EPSILON),
    }


def _bootstrap_pair_difference(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    repeats: int = 5000,
) -> dict[str, Any]:
    valid = [
        row for row in rows
        if int(row["new"]["pairs"]) > 0 and int(row["old"]["pairs"]) > 0
    ]
    if not valid:
        return {
            "frames_with_non_tie": 0,
            "run_clusters": 0,
            "bootstrap_unit": "paired_run_load_seed",
            "old_pair_concordance": None,
            "new_pair_concordance": None,
            "delta_pair_concordance": None,
            "delta_ci95": [None, None],
        }
    old_correct = sum(int(row["old"]["correct"]) for row in valid)
    new_correct = sum(int(row["new"]["correct"]) for row in valid)
    old_pairs = sum(int(row["old"]["pairs"]) for row in valid)
    new_pairs = sum(int(row["new"]["pairs"]) for row in valid)

    # Frames from the same closed-loop run are correlated.  Resampling frames
    # independently would treat them as extra experimental replicates and make
    # the CI too narrow.  Keep every frame from one (load, seed) run together,
    # then bootstrap those paired run clusters.
    grouped: dict[tuple[str, int], dict[str, int]] = {}
    for row in valid:
        frame_key = row.get("frame_key")
        if not isinstance(frame_key, Sequence) or len(frame_key) < 2:
            raise ValueError("paired bootstrap requires frame_key=(load, seed, tick)")
        run_key = (str(frame_key[0]), int(frame_key[1]))
        cluster = grouped.setdefault(
            run_key,
            {
                "old_correct": 0,
                "new_correct": 0,
                "old_pairs": 0,
                "new_pairs": 0,
            },
        )
        cluster["old_correct"] += int(row["old"]["correct"])
        cluster["new_correct"] += int(row["new"]["correct"])
        cluster["old_pairs"] += int(row["old"]["pairs"])
        cluster["new_pairs"] += int(row["new"]["pairs"])

    clusters = list(grouped.values())
    rng = np.random.default_rng(int(seed))
    n_clusters = len(clusters)
    draws = rng.integers(
        0, n_clusters, size=(int(repeats), n_clusters)
    )
    old_correct_values = np.asarray(
        [cluster["old_correct"] for cluster in clusters], dtype=np.float64
    )
    new_correct_values = np.asarray(
        [cluster["new_correct"] for cluster in clusters], dtype=np.float64
    )
    old_pair_values = np.asarray(
        [cluster["old_pairs"] for cluster in clusters], dtype=np.float64
    )
    new_pair_values = np.asarray(
        [cluster["new_pairs"] for cluster in clusters], dtype=np.float64
    )
    sampled_old_correct = old_correct_values[draws].sum(axis=1)
    sampled_new_correct = new_correct_values[draws].sum(axis=1)
    sampled_old_pairs = old_pair_values[draws].sum(axis=1)
    sampled_new_pairs = new_pair_values[draws].sum(axis=1)
    deltas = (
        sampled_new_correct / sampled_new_pairs
        - sampled_old_correct / sampled_old_pairs
    )
    return {
        "frames_with_non_tie": len(valid),
        "run_clusters": n_clusters,
        "bootstrap_unit": "paired_run_load_seed",
        "bootstrap_repeats": int(repeats),
        "old_pair_concordance": float(old_correct / old_pairs),
        "new_pair_concordance": float(new_correct / new_pairs),
        "delta_pair_concordance": float(
            new_correct / new_pairs - old_correct / old_pairs
        ),
        "delta_ci95": [
            float(np.quantile(deltas, 0.025)),
            float(np.quantile(deltas, 0.975)),
        ],
    }


def _paired_evaluate(
    old_frames: Sequence[Sequence[Mapping[str, Any]]],
    new_frames: Sequence[Sequence[Mapping[str, Any]]],
    old_head: torch.nn.Module,
    new_head: torch.nn.Module,
    *,
    device: torch.device,
    bootstrap_seed: int,
) -> dict[str, Any]:
    if len(old_frames) != len(new_frames):
        raise ValueError("old/new frame counts differ")
    rows: list[dict[str, Any]] = []
    old_head.eval()
    new_head.eval()
    with torch.inference_mode():
        for old_frame, new_frame in zip(old_frames, new_frames):
            if _frame_key(old_frame) != _frame_key(new_frame):
                raise ValueError("old/new frame keys differ")
            old_contexts = [tuple(row["context_key"]) for row in old_frame]
            new_contexts = [tuple(row["context_key"]) for row in new_frame]
            if old_contexts != new_contexts:
                raise ValueError("old/new context ordering differs")
            features_old = torch.stack(
                [row["features"] for row in old_frame]
            ).to(device)
            semantic_old = torch.stack(
                [row["semantic"] for row in old_frame]
            ).to(device)
            features_new = torch.stack(
                [row["features"] for row in new_frame]
            ).to(device)
            semantic_new = torch.stack(
                [row["semantic"] for row in new_frame]
            ).to(device)
            old_prediction = old_head(features_old, semantic_old).cpu().numpy()
            new_prediction = new_head(features_new, semantic_new).cpu().numpy()
            target = np.asarray(
                [float(row["target_cost"]) for row in old_frame],
                dtype=np.float64,
            )
            if not np.allclose(
                target,
                np.asarray([float(row["target_cost"]) for row in new_frame]),
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError("old/new target rows differ")
            rows.append({
                "frame_key": list(_frame_key(old_frame)),
                "load": str(old_frame[0]["load"]),
                "old": _frame_pair_stats(old_prediction, target),
                "new": _frame_pair_stats(new_prediction, target),
            })

    def aggregate(values: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        pair_old = sum(int(row["old"]["correct"]) for row in values)
        pair_new = sum(int(row["new"]["correct"]) for row in values)
        pairs_old = sum(int(row["old"]["pairs"]) for row in values)
        pairs_new = sum(int(row["new"]["pairs"]) for row in values)
        old_pair_frames = [
            float(row["old"]["pair_concordance"])
            for row in values if math.isfinite(float(row["old"]["pair_concordance"]))
        ]
        new_pair_frames = [
            float(row["new"]["pair_concordance"])
            for row in values if math.isfinite(float(row["new"]["pair_concordance"]))
        ]
        old_spear = [
            float(row["old"]["spearman"])
            for row in values if math.isfinite(float(row["old"]["spearman"]))
        ]
        new_spear = [
            float(row["new"]["spearman"])
            for row in values if math.isfinite(float(row["new"]["spearman"]))
        ]
        old_top = [float(row["old"]["top1"]) for row in values]
        new_top = [float(row["new"]["top1"]) for row in values]
        return {
            "frames": len(values),
            "non_tie_pairs_old": pairs_old,
            "non_tie_pairs_new": pairs_new,
            "old_pair_concordance_pair_weighted": (
                pair_old / pairs_old if pairs_old else None
            ),
            "new_pair_concordance_pair_weighted": (
                pair_new / pairs_new if pairs_new else None
            ),
            "delta_pair_concordance_pair_weighted": (
                pair_new / pairs_new - pair_old / pairs_old
                if pairs_old and pairs_new else None
            ),
            "old_pair_concordance_frame_mean": (
                float(np.mean(old_pair_frames)) if old_pair_frames else None
            ),
            "new_pair_concordance_frame_mean": (
                float(np.mean(new_pair_frames)) if new_pair_frames else None
            ),
            "old_spearman_frame_mean": float(np.mean(old_spear)) if old_spear else None,
            "new_spearman_frame_mean": float(np.mean(new_spear)) if new_spear else None,
            "old_top1_frame_mean": float(np.mean(old_top)),
            "new_top1_frame_mean": float(np.mean(new_top)),
            "delta_top1_frame_mean": float(np.mean(new_top) - np.mean(old_top)),
        }

    result: dict[str, Any] = {
        "overall": aggregate(rows),
        "by_load": {
            load: aggregate([row for row in rows if row["load"] == load])
            for load in LOADS
        },
        "paired_bootstrap": {},
    }
    for name, values in [("overall", rows)] + [
        (load, [row for row in rows if row["load"] == load])
        for load in LOADS
    ]:
        result["paired_bootstrap"][name] = _bootstrap_pair_difference(
            values, seed=bootstrap_seed + len(name)
        )
    return result


def _standard_evaluate(
    head: torch.nn.Module,
    frames: Mapping[str, Sequence[Sequence[Mapping[str, Any]]]],
    *,
    device: torch.device,
    seed_offset: int,
) -> dict[str, Any]:
    return {
        split: train_mod._evaluate_score(
            head,
            split_frames,
            device=device,
            epsilon=EPSILON,
            bootstrap_seed=20260810 + seed_offset + index,
        )
        for index, (split, split_frames) in enumerate(frames.items())
    }


def run_control(
    *,
    data_root: Path,
    audit_path: Path,
    audit_sha_path: Path,
    old_checkpoint: Path,
    new_checkpoint: Path,
    world_model_path: Path,
    phi_head_path: Path,
    layout_config: Path,
    output_root: Path,
    torch_threads: int,
) -> dict[str, Any]:
    audit = _verify_label_audit(data_root, audit_path, audit_sha_path)
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite {output_root}")
    data_paths = train_mod._expand_data_paths([str(data_root / "runs")])
    if len(data_paths) != 60:
        raise ValueError(f"expected 60 data paths, found {len(data_paths)}")
    torch.set_num_threads(max(1, int(torch_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device("cpu")
    config = load_config(str(layout_config))
    world = WorldState(config)
    expected_edge_index, node_map, *_ = build_static_graph(world.map_state)
    layout = build_station_layout(world, node_map, expected_edge_index)
    model, model_config = load_frozen_world_model(
        str(world_model_path),
        train_mod._load_raw_samples(data_paths[0])[0],
        len(layout.station_ids),
        str(device),
    )
    phi_head, phi_payload = load_frozen_station_head(
        phi_head_path,
        model_checkpoint_path=world_model_path,
        device=device,
    )
    rows, extraction_audit = train_mod._extract_compact_rows(
        data_paths=data_paths,
        model=model,
        model_config=model_config,
        phi_head=phi_head,
        phi_payload=phi_payload,
        layout=layout,
        expected_edge_index=expected_edge_index,
        device=device,
        horizon=HORIZON,
        allowed_seeds=set(SEEDS),
        max_files=None,
    )
    del model, phi_head
    if len(rows) != 3124:
        raise ValueError(f"unexpected compact context row count: {len(rows)}")
    old_head, old_payload = _load_head(
        old_checkpoint,
        device=device,
        world_model_hash=sha256_file(world_model_path),
        phi_hash=sha256_file(phi_head_path),
    )
    new_head, new_payload = _load_head(
        new_checkpoint,
        device=device,
        world_model_hash=sha256_file(world_model_path),
        phi_hash=sha256_file(phi_head_path),
    )
    if old_payload["head_config"] != new_payload["head_config"]:
        raise ValueError("old/new head architecture configs differ")
    raw_feature_dim = int(rows[0]["features_raw"].numel())
    if int(old_payload["head_config"]["feature_dim"]) != raw_feature_dim:
        raise ValueError("old head feature dimension does not match extracted rows")
    if int(new_payload["head_config"]["feature_dim"]) != raw_feature_dim:
        raise ValueError("new head feature dimension does not match extracted rows")
    old_rows = _normalised_copy(rows, old_payload)
    new_rows = _normalised_copy(rows, new_payload)
    old_frames = _split_frames(old_rows)
    new_frames = _split_frames(new_rows)
    for split in ("train", "val", "test"):
        if len(old_frames[split]) != len(new_frames[split]):
            raise ValueError(f"{split}: old/new frame counts differ after normalisation")
    old_standard = _standard_evaluate(old_head, old_frames, device=device, seed_offset=0)
    new_standard = _standard_evaluate(new_head, new_frames, device=device, seed_offset=100)
    paired = {
        split: _paired_evaluate(
            old_frames[split],
            new_frames[split],
            old_head,
            new_head,
            device=device,
            bootstrap_seed=20260810 + index * 100,
        )
        for index, split in enumerate(("train", "val", "test"))
    }
    result = {
        "schema_version": "context_j_head_offline_control_v1",
        "development_only": True,
        "online_connection_allowed": False,
        "protocol_sha256": PROTOCOL_SHA256,
        "data_root": data_root.as_posix(),
        "label_audit": {
            "path": audit_path.as_posix(),
            "sha256": sha256_file(audit_path),
            "expected_runs": audit["expected_runs"],
            "completed_runs": audit["completed_runs"],
            "signal_audit": audit["signal_audit"],
        },
        "source_contract": {
            "world_model": {
                "path": world_model_path.as_posix(),
                "sha256": sha256_file(world_model_path),
            },
            "station_phi": {
                "path": phi_head_path.as_posix(),
                "sha256": sha256_file(phi_head_path),
            },
            "layout_config": {
                "path": layout_config.as_posix(),
                "sha256": sha256_file(layout_config),
            },
            "source_file_count": len(data_paths),
            "compact_context_rows": len(rows),
            "extraction_audit": extraction_audit,
            "normalisation": (
                "each checkpoint uses its own frozen train-only feature and "
                "semantic contracts; raw rows and targets are identical"
            ),
        },
        "splits": {
            "train": {"old_531_540": old_standard["train"],
                      "new_571_590": new_standard["train"],
                      "paired": paired["train"]},
            "val": {"old_531_540": old_standard["val"],
                    "new_571_590": new_standard["val"],
                    "paired": paired["val"]},
            "test": {"old_531_540": old_standard["test"],
                     "new_571_590": new_standard["test"],
                     "paired": paired["test"]},
        },
        "checkpoints": {
            "old_531_540": {
                "path": old_checkpoint.as_posix(),
                "sha256": sha256_file(old_checkpoint),
                "feature_contract": old_payload["feature_contract"],
                "semantic_weights": old_payload["semantic_weights"],
            },
            "new_571_590": {
                "path": new_checkpoint.as_posix(),
                "sha256": sha256_file(new_checkpoint),
                "feature_contract": new_payload["feature_contract"],
                "semantic_weights": new_payload["semantic_weights"],
            },
        },
        "primary_test_comparison": paired["test"],
        "interpretation": {
            "primary_question": (
                "on identical new held-out seeds 587--590, does the expanded-data "
                "head outperform the old 531--540 head?"
            ),
            "not_an_online_result": True,
            "within_context_robot_scorer": "unchanged S1; not evaluated here",
            "lower_score_is_better": True,
            "pair_metric_is_primary": True,
            "top1_is_tie_sensitive": True,
        },
        "runtime_seconds": None,
    }
    output_root.mkdir(parents=True, exist_ok=False)
    path = output_root / "control_comparison.json"
    path.write_text(
        json.dumps(_safe(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = [
        f"{sha256_file(path)}  {path.name}",
    ]
    (output_root / "control_outputs.sha256").write_text(
        "\n".join(manifest) + "\n", encoding="utf-8", newline="\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--audit", type=Path, default=LABEL_AUDIT)
    parser.add_argument("--audit-sha", type=Path, default=LABEL_AUDIT_SHA)
    parser.add_argument("--old-checkpoint", type=Path, default=OLD_HEAD)
    parser.add_argument("--new-checkpoint", type=Path, default=NEW_HEAD)
    parser.add_argument("--world-model", type=Path, default=WORLD_MODEL)
    parser.add_argument("--phi-head", type=Path, default=DEFAULT_PHI_HEAD)
    parser.add_argument("--layout-config", type=Path, default=LAYOUT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--torch-threads", type=int, default=16)
    args = parser.parse_args()
    started = time.time()
    result = run_control(
        data_root=args.data_root,
        audit_path=args.audit,
        audit_sha_path=args.audit_sha,
        old_checkpoint=args.old_checkpoint,
        new_checkpoint=args.new_checkpoint,
        world_model_path=args.world_model,
        phi_head_path=args.phi_head,
        layout_config=args.layout_config,
        output_root=args.output_root,
        torch_threads=args.torch_threads,
    )
    result["runtime_seconds"] = time.time() - started
    path = args.output_root / "control_comparison.json"
    path.write_text(
        json.dumps(_safe(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (args.output_root / "control_outputs.sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n",
        encoding="utf-8",
        newline="\n",
    )
    print("=" * 88)
    print("Context-J old-vs-new offline control")
    print("=" * 88)
    for split in ("train", "val", "test"):
        p = result["splits"][split]["paired"]["overall"]
        print(
            split,
            "old=", p["old_pair_concordance_pair_weighted"],
            "new=", p["new_pair_concordance_pair_weighted"],
            "delta=", p["delta_pair_concordance_pair_weighted"],
        )
    print("report =", path)
    print("runtime_seconds =", result["runtime_seconds"])


if __name__ == "__main__":
    main()
