"""Run the isolated repaired-LongRisk Combo-S1+J1 PIBT supplement.

This module deliberately does not alter the canonical PIBT factorial outputs.
It reuses the frozen paired manifests and the audited single-arm implementation
from :mod:`run_phase_c_pibt_planner_study`, while binding explicit repaired
checkpoint/J1 artifacts for two arms only:

* repaired PP-WM + PIBT (zero-shot transfer), and
* PIBT-specific repaired WM + PIBT (on-policy transfer).

The small compatibility layer patches the imported protocol constants only in
this process before calling the canonical runner.  No repository constants or
existing result files are changed.  The PIBT-specific LongRiskHead must be
trained separately with ``train_long_risk_head_only.py --planner-specific-source``
before evaluation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping

from WorldModel.evaluation import phase_c_pibt_planner_study_protocol as protocol
from WorldModel.evaluation import run_phase_c_pibt_planner_study as study


SCHEMA_VERSION = "phase_c_pibt_repaired_longrisk_combo_supplement_v1"
BASE_ROOT = Path("WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1")
DEFAULT_OUTPUT_ROOT = BASE_ROOT / "phasec_pibt_repaired_longrisk_combo_551_560_v1"

PP_REPAIRED_CHECKPOINT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_longrisk_head_repair_681_700_v1/"
    "training/long_risk_head_only_v1/best_long_risk_world_model.pt"
)
PP_REPAIRED_PSI_HEAD = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_longrisk_head_repair_681_700_v1/"
    "training/station_head_rebound_v1/best_station_congestion_head.pt"
)
PP_SCALE_CONTRACT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
    "station_congestion_head_region_dev_511_520_v1/"
    "station_congestion_scale_contract.json"
)

PIBT_SOURCE_CHECKPOINT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_pibt_round1_v2/"
    "model_round1_v1/best_regret_world_model.pt"
)
PIBT_SOURCE_PSI_HEAD = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_pibt_round1_v2/"
    "station_congestion_head_region_pibt_461_470_v1/linear_head_v1/"
    "best_station_congestion_head.pt"
)
PIBT_SCALE_CONTRACT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_pibt_round1_v2/"
    "station_congestion_head_region_pibt_461_470_v1/"
    "station_congestion_scale_contract.json"
)
PIBT_TRAIN_ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_pibt_round1_v2"
)
PIBT_FUSED_ROOT = PIBT_TRAIN_ROOT / "fused_seed_split_v1"
PIBT_LONGRISK_REPAIR_ROOT = (
    DEFAULT_OUTPUT_ROOT / "pibt_longrisk_head_repair"
)
PIBT_REBOUND_ROOT = DEFAULT_OUTPUT_ROOT / "pibt_station_head_rebound"

ZERO_BASE_ROOT = BASE_ROOT / "phasec_pibt_cross_planner_ppweights_551_560_v2"
ADAPTED_BASE_ROOT = BASE_ROOT / "phasec_pibt_retrained_weights_551_560_v2"
MANIFEST_CONTRACT = ZERO_BASE_ROOT / "validation/phase_c_pibt_input_manifests.json"

PP_ARM = "combo_ppwm_pibt"
PIBT_ARM = "combo_pibtwm_pibt"
LOADS = tuple(protocol.LOADS)
SEEDS = tuple(protocol.EVAL_SEEDS)
TICKS = int(protocol.TICKS)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite changed output: {path}")
        return
    fd, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _set_protocol_value(name: str, value: Any) -> None:
    """Patch one imported constant in both protocol and runner namespaces."""

    setattr(protocol, name, value)
    # The canonical runner imported several constants by value.  Updating its
    # namespace is required for the freeze/policy code, while the helper
    # functions (checkpoint_for_arm/psi_artifacts_for_arm) read protocol's
    # globals directly.
    if hasattr(study, name):
        setattr(study, name, value)


def _configure_artifacts(
    *,
    campaign: str,
    checkpoint: Path,
    psi_head: Path,
    psi_scale: Path,
) -> None:
    if campaign == "zero_shot":
        _set_protocol_value("PP_TRAINED_CHECKPOINT", checkpoint)
        _set_protocol_value("PSI_HEAD_CHECKPOINT", psi_head)
        _set_protocol_value("PSI_SCALE_CONTRACT", psi_scale)
    elif campaign == "adapted":
        _set_protocol_value("PIBT_TRAINED_CHECKPOINT", checkpoint)
        _set_protocol_value("PIBT_PSI_HEAD_CHECKPOINT", psi_head)
        _set_protocol_value("PIBT_PSI_SCALE_CONTRACT", psi_scale)
    else:
        raise ValueError(f"unsupported campaign: {campaign}")

    # Include this supplement and the opt-in trainer in the frozen provenance
    # bundle.  Keep all canonical entries unchanged.
    submission_path = Path(
        os.environ.get(
            "PIBT_REPAIRED_SUBMISSION_PATH",
            "WorldModel/evaluation/"
            "run_phase_c_pibt_repaired_longrisk_combo_60cpu.slurm",
        )
    )
    source_files = dict(study.SOURCE_FILES)
    source_files.update({
        "repaired_supplement_runner": Path(__file__),
        "repaired_supplement_submission": submission_path,
        "planner_specific_longrisk_trainer": Path(
            "WorldModel/training/train_long_risk_head_only.py"
        ),
        "station_head_rebinder": Path(
            "WorldModel/training/rebind_station_congestion_head_checkpoint.py"
        ),
    })
    study.SOURCE_FILES = source_files


def _manifest_contract(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return study._load_manifest_contract(path)


def _manifest_path(load: str, seed: int, contract: Mapping[str, Any]) -> Path:
    row = (contract.get("cells") or {}).get(f"{load}:seed{seed}") or {}
    path = Path(str(row.get("path") or ""))
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _required_paths_for_preflight(args: argparse.Namespace) -> list[Path]:
    pibt_source = Path(args.pibt_source_checkpoint)
    pibt_head = Path(args.pibt_source_psi_head)
    required = [
        Path(args.manifest_contract),
        Path(args.pp_repaired_checkpoint),
        Path(args.pp_repaired_checkpoint).parent / "tensor_audit.json",
        Path(args.pp_repaired_checkpoint).parent / "train_summary.json",
        Path(args.pp_repaired_psi_head),
        Path(args.pp_repaired_psi_head).parent / "rebind_summary.json",
        Path(args.pp_scale_contract),
        pibt_source,
        pibt_head,
        Path(args.pibt_scale_contract),
        Path(args.zero_base_root) / "phase_c_pibt_planner_study_frozen_protocol.json",
        Path(args.zero_base_root) / "validation/summary.json",
        Path(args.zero_base_root) / "validation/validated_outputs.sha256",
        Path(args.adapted_base_root)
        / "phase_c_pibt_planner_study_frozen_protocol.json",
        Path(args.adapted_base_root) / "validation/summary.json",
        Path(args.adapted_base_root) / "validation/validated_outputs.sha256",
    ]
    if args.require_training_data:
        required.extend([
            PIBT_TRAIN_ROOT / "snapshots_461_470/snapshot_collection_outputs.sha256",
            PIBT_TRAIN_ROOT / "datasets_461_470/replayed_datasets.sha256",
            PIBT_TRAIN_ROOT / "long_risk_w200_461_470/long_risk_outputs.sha256",
            PIBT_FUSED_ROOT / "phase_c_round1_fused.pt",
            PIBT_FUSED_ROOT / "splits.json",
            PIBT_FUSED_ROOT / "manifest.json",
            PIBT_FUSED_ROOT / "quality_report.json",
            PIBT_FUSED_ROOT / "validated_outputs.sha256",
        ])
    return required


def _validate_repair_output(
    checkpoint: Path,
    *,
    require_planner_specific: bool,
) -> None:
    audit_path = checkpoint.parent / "tensor_audit.json"
    summary_path = checkpoint.parent / "train_summary.json"
    for path in (checkpoint, audit_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = read_json(audit_path)
    summary = read_json(summary_path)
    checks = audit.get("checks") or {}
    required = (
        "all_non_long_risk_tensors_bitwise_equal",
        "long_risk_head_updated",
        "cost_head_bitwise_equal",
        "model_config_equal",
        "action_schema_equal",
    )
    if not audit.get("passed") or not all(bool(checks.get(name)) for name in required):
        raise ValueError(f"failed LongRiskHead tensor audit: {audit_path}")
    if summary.get("output_checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError(f"LongRiskHead checkpoint hash mismatch: {checkpoint}")
    if require_planner_specific:
        if not bool((summary.get("training") or {}).get("planner_specific_source")):
            raise ValueError("PIBT repair was not trained in planner-specific mode")
        if not bool(audit.get("planner_specific_source")):
            raise ValueError("PIBT tensor audit lacks planner-specific provenance")


def _validate_rebound_output(head: Path) -> None:
    summary_path = head.parent / "rebind_summary.json"
    if not head.is_file() or not summary_path.is_file():
        raise FileNotFoundError(head if not head.is_file() else summary_path)
    summary = read_json(summary_path)
    if summary.get("output_sha256") != sha256_file(head):
        raise ValueError(f"rebound J1 checkpoint hash mismatch: {head}")
    if not bool((summary.get("audit") or {}).get("passed")):
        raise ValueError(f"rebound J1 tensor-equivalence audit failed: {head}")


def preflight(args: argparse.Namespace) -> None:
    missing = [path for path in _required_paths_for_preflight(args) if not path.is_file()]
    if missing:
        lines = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "repaired PIBT supplement prerequisites are missing; sync or restore "
            "these files/directories before submitting:\n" + lines
        )
    contract = _manifest_contract(Path(args.manifest_contract))
    if tuple(contract.get("loads") or ()) != LOADS or tuple(
        int(seed) for seed in contract.get("seeds") or ()
    ) != SEEDS:
        raise ValueError("manifest contract is not the frozen low/mid/high 551-560 block")
    for load in LOADS:
        for seed in SEEDS:
            _manifest_path(load, seed, contract)

    _validate_repair_output(
        Path(args.pp_repaired_checkpoint), require_planner_specific=False
    )
    _validate_rebound_output(Path(args.pp_repaired_psi_head))
    if args.require_training_data:
        splits = read_json(PIBT_FUSED_ROOT / "splits.json")
        expected_split = {
            "train": [str(seed) for seed in range(461, 468)],
            "val": ["468", "469"],
            "test": ["470"],
        }
        if splits.get("split_unit") != "source_seed" or splits.get(
            "explicit_seeds"
        ) != expected_split:
            raise ValueError("PIBT fused split is not the frozen 461-470 seed split")
        quality = read_json(PIBT_FUSED_ROOT / "quality_report.json")
        if int(quality.get("total_long_risk_samples", 0)) <= 0 or int(
            quality.get("total_long_risk_groups", 0)
        ) <= 0:
            raise ValueError("PIBT fused dataset lacks LongRisk supervision")

    _configure_artifacts(
        campaign="zero_shot",
        checkpoint=Path(args.pp_repaired_checkpoint),
        psi_head=Path(args.pp_repaired_psi_head),
        psi_scale=Path(args.pp_scale_contract),
    )
    study._validate_psi_artifacts(
        Path(args.pp_repaired_psi_head),
        Path(args.pp_scale_contract),
        Path(args.pp_repaired_checkpoint),
        require_formal=False,
    )
    _configure_artifacts(
        campaign="adapted",
        checkpoint=Path(args.pibt_source_checkpoint),
        psi_head=Path(args.pibt_source_psi_head),
        psi_scale=Path(args.pibt_scale_contract),
    )
    study._validate_psi_artifacts(
        Path(args.pibt_source_psi_head),
        Path(args.pibt_scale_contract),
        Path(args.pibt_source_checkpoint),
        require_formal=True,
    )
    print("repaired PIBT supplement preflight PASS")


def _freeze_one(
    *,
    campaign: str,
    output_root: Path,
    checkpoint: Path,
    psi_head: Path,
    psi_scale: Path,
    manifest_contract: Path,
    source_zero_shot_root: Path,
) -> Path:
    _configure_artifacts(
        campaign=campaign,
        checkpoint=checkpoint,
        psi_head=psi_head,
        psi_scale=psi_scale,
    )
    namespace = SimpleNamespace(
        output_root=str(output_root),
        checkpoint=str(checkpoint),
        campaign=campaign,
        manifest_contract=str(manifest_contract),
        source_zero_shot_root=str(source_zero_shot_root),
    )
    study._freeze_evaluation(namespace)
    return output_root / study.EVAL_BUNDLE_FILENAME


def freeze(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    _validate_repair_output(
        Path(args.pp_repaired_checkpoint), require_planner_specific=False
    )
    _validate_rebound_output(Path(args.pp_repaired_psi_head))
    _validate_repair_output(
        Path(args.pibt_repaired_checkpoint), require_planner_specific=True
    )
    _validate_rebound_output(Path(args.pibt_rebound_psi_head))
    pp_root = root / "pp_repaired"
    pibt_root = root / "pibt_repaired"
    _freeze_one(
        campaign="zero_shot",
        output_root=pp_root,
        checkpoint=Path(args.pp_repaired_checkpoint),
        psi_head=Path(args.pp_repaired_psi_head),
        psi_scale=Path(args.pp_scale_contract),
        manifest_contract=Path(args.manifest_contract),
        source_zero_shot_root=Path(args.zero_base_root),
    )
    _freeze_one(
        campaign="adapted",
        output_root=pibt_root,
        checkpoint=Path(args.pibt_repaired_checkpoint),
        psi_head=Path(args.pibt_rebound_psi_head),
        psi_scale=Path(args.pibt_scale_contract),
        manifest_contract=Path(args.manifest_contract),
        source_zero_shot_root=Path(args.zero_base_root),
    )
    print(f"frozen repaired bundles under {root}")


def run_one(args: argparse.Namespace) -> None:
    if args.arm == PP_ARM:
        campaign = "zero_shot"
        checkpoint = Path(args.pp_repaired_checkpoint)
        psi_head = Path(args.pp_repaired_psi_head)
        psi_scale = Path(args.pp_scale_contract)
        root = Path(args.output_root) / "pp_repaired"
    elif args.arm == PIBT_ARM:
        campaign = "adapted"
        checkpoint = Path(args.pibt_repaired_checkpoint)
        psi_head = Path(args.pibt_rebound_psi_head)
        psi_scale = Path(args.pibt_scale_contract)
        root = Path(args.output_root) / "pibt_repaired"
    else:
        raise ValueError(f"unsupported supplement arm: {args.arm}")
    _configure_artifacts(
        campaign=campaign,
        checkpoint=checkpoint,
        psi_head=psi_head,
        psi_scale=psi_scale,
    )
    bundle = root / study.EVAL_BUNDLE_FILENAME
    if not bundle.is_file():
        raise FileNotFoundError(bundle)
    manifest = _manifest_path(args.load, int(args.seed), _manifest_contract(Path(args.manifest_contract)))
    namespace = SimpleNamespace(
        campaign=campaign,
        arm=args.arm,
        load=args.load,
        seed=int(args.seed),
        ticks=int(args.ticks),
        manifest_path=str(manifest),
        output_root=str(root),
        frozen_bundle=str(bundle),
    )
    study._run_arm(namespace)


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _mean_std(values: Iterable[float]) -> dict[str, Any]:
    vals = [float(value) for value in values]
    if not vals:
        return {"n": 0, "mean": None, "std": None}
    mean = sum(vals) / len(vals)
    variance = sum((value - mean) ** 2 for value in vals) / max(len(vals) - 1, 1)
    return {"n": len(vals), "mean": mean, "std": math.sqrt(variance)}


def _read_run(path: Path, *, arm: str, load: str, seed: int, contract: Mapping[str, Any]) -> dict[str, Any]:
    payload = read_json(path)
    meta = payload.get("meta") or {}
    manifest = payload.get("manifest") or {}
    cell = (contract.get("cells") or {}).get(f"{load}:seed{seed}") or {}
    checks = {
        "schema": payload.get("schema_version") == study.RUN_SCHEMA_VERSION,
        "arm": meta.get("arm_key") == arm,
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == seed,
        "ticks": int(meta.get("ticks", -1)) == TICKS,
        "manifest": manifest.get("content_sha256") == cell.get("content_sha256"),
        "audit": bool((payload.get("audit") or {}).get("passed")),
        "combo_signal": (payload.get("metrics") or {}).get("energy_drift_signal") == "combo",
        "j1": (payload.get("metrics") or {}).get("psi_dispatch_mode") == "j_ascending",
        "j1_binding": bool((payload.get("metrics") or {}).get("psi_dispatch_encoder_contract_verified")),
    }
    if not all(checks.values()):
        raise RuntimeError(f"invalid repaired run {path}: " + ", ".join(name for name, ok in checks.items() if not ok))
    return payload


def _read_unaffected_baseline(
    path: Path,
    *,
    arm: str,
    load: str,
    seed: int,
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    payload = read_json(path)
    meta = payload.get("meta") or {}
    manifest = payload.get("manifest") or {}
    cell = (contract.get("cells") or {}).get(f"{load}:seed{seed}") or {}
    checks = {
        "schema": payload.get("schema_version") == study.RUN_SCHEMA_VERSION,
        "arm": meta.get("arm_key") == arm,
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == seed,
        "ticks": int(meta.get("ticks", -1)) == TICKS,
        "manifest": manifest.get("content_sha256") == cell.get("content_sha256"),
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if not all(checks.values()):
        raise RuntimeError(
            f"invalid unaffected baseline {path}: "
            + ", ".join(name for name, ok in checks.items() if not ok)
        )
    return payload


def _paired_delta(values_left: list[float], values_right: list[float]) -> dict[str, Any]:
    return _mean_std([right - left for left, right in zip(values_left, values_right)])


def summarize(args: argparse.Namespace) -> None:
    root = Path(args.output_root)
    contract = _manifest_contract(Path(args.manifest_contract))
    rows: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for load in LOADS:
        for seed in SEEDS:
            rows[(load, seed)] = {}
            pp_path = root / "pp_repaired/per_arm" / PP_ARM / f"{load}_seed{seed}.json"
            pibt_path = root / "pibt_repaired/per_arm" / PIBT_ARM / f"{load}_seed{seed}.json"
            rows[(load, seed)]["pp_repaired"] = _read_run(pp_path, arm=PP_ARM, load=load, seed=seed, contract=contract)
            rows[(load, seed)]["pibt_repaired"] = _read_run(pibt_path, arm=PIBT_ARM, load=load, seed=seed, contract=contract)
            # Unchanged baselines are read-only inputs.  The old Combo rows are
            # intentionally never loaded as a baseline here.
            rows[(load, seed)]["greedy"] = _read_unaffected_baseline(
                Path(args.zero_base_root)
                / "per_arm/greedy_pibt"
                / f"{load}_seed{seed}.json",
                arm="greedy_pibt",
                load=load,
                seed=seed,
                contract=contract,
            )
            rows[(load, seed)]["phasec_ppwm"] = _read_unaffected_baseline(
                Path(args.zero_base_root)
                / "per_arm/phasec_ppwm_pibt"
                / f"{load}_seed{seed}.json",
                arm="phasec_ppwm_pibt",
                load=load,
                seed=seed,
                contract=contract,
            )
            rows[(load, seed)]["phasec_pibtwm"] = _read_unaffected_baseline(
                Path(args.adapted_base_root)
                / "per_arm/phasec_pibtwm_pibt"
                / f"{load}_seed{seed}.json",
                arm="phasec_pibtwm_pibt",
                load=load,
                seed=seed,
                contract=contract,
            )

    metrics = (
        "completed_orders",
        "avg_task_duration",
        "avg_excess_delay",
        "deadlock_ratio_mean",
        "congestion_events",
        "severe_events",
        "pibt_wait_decision_ratio",
    )
    aggregate: list[dict[str, Any]] = []
    contrasts: list[dict[str, Any]] = []
    for label in ("pp_repaired", "pibt_repaired", "greedy", "phasec_ppwm", "phasec_pibtwm"):
        for load in LOADS:
            for metric in metrics:
                values = [_number(rows[(load, seed)][label].get("metrics", {}).get(metric)) for seed in SEEDS]
                clean = [value for value in values if value is not None]
                aggregate.append({"arm": label, "load": load, "metric": metric, **_mean_std(clean)})

    comparisons = (
        ("pp_repaired_minus_greedy", "greedy", "pp_repaired"),
        ("pp_repaired_minus_phasec", "phasec_ppwm", "pp_repaired"),
        ("pibt_repaired_minus_greedy", "greedy", "pibt_repaired"),
        ("pibt_repaired_minus_phasec", "phasec_pibtwm", "pibt_repaired"),
    )
    for name, baseline, candidate in comparisons:
        for metric in metrics:
            for load in LOADS:
                left = [_number(rows[(load, seed)][baseline].get("metrics", {}).get(metric)) for seed in SEEDS]
                right = [_number(rows[(load, seed)][candidate].get("metrics", {}).get(metric)) for seed in SEEDS]
                pairs = [(a, b) for a, b in zip(left, right) if a is not None and b is not None]
                summary = _paired_delta([a for a, _ in pairs], [b for _, b in pairs])
                contrasts.append({"comparison": name, "metric": metric, "load": load, **summary})

    validation = root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    with (validation / "aggregate_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["arm", "load", "metric", "n", "mean", "std"])
        writer.writeheader()
        writer.writerows(aggregate)
    with (validation / "paired_contrasts.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison", "metric", "load", "n", "mean", "std"])
        writer.writeheader()
        writer.writerows(contrasts)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "revalidate LongRisk-dependent Combo-S1+J1 across PIBT without rerunning unaffected baselines",
        "loads": list(LOADS),
        "seeds": list(SEEDS),
        "ticks": TICKS,
        "station_admission": "physical-only",
        "arms": {
            "pp_repaired": "repaired PP-WM + Combo-S1+J1 + PIBT",
            "pibt_repaired": "PIBT-specific repaired WM + Combo-S1+J1 + PIBT",
        },
        "baselines_reused_read_only": ["Greedy+PIBT", "PhaseC(PP-trained WM)+PIBT", "PhaseC(PIBT-trained WM)+PIBT"],
        "invalid_not_used": ["historical Combo PP-WM + PIBT", "historical Combo PIBT-WM + PIBT"],
        "artifact_sha256": {
            "pp_repaired_checkpoint": sha256_file(Path(args.pp_repaired_checkpoint)),
            "pibt_repaired_checkpoint": sha256_file(Path(args.pibt_repaired_checkpoint)),
            "pp_repaired_j1": sha256_file(Path(args.pp_repaired_psi_head)),
            "pibt_repaired_j1": sha256_file(Path(args.pibt_rebound_psi_head)),
            "manifest_contract": sha256_file(Path(args.manifest_contract)),
        },
        "aggregate": aggregate,
        "paired_contrasts": contrasts,
    }
    write_json(validation / "summary.json", summary)
    validated = [
        root / "pp_repaired" / study.EVAL_BUNDLE_FILENAME,
        root / "pibt_repaired" / study.EVAL_BUNDLE_FILENAME,
        validation / "aggregate_summary.csv",
        validation / "paired_contrasts.csv",
        validation / "summary.json",
        Path(args.pibt_repaired_checkpoint),
        Path(args.pibt_rebound_psi_head),
    ]
    validated.extend(
        root / campaign / "per_arm" / arm / f"{load}_seed{seed}.json"
        for campaign, arm in (
            ("pp_repaired", PP_ARM),
            ("pibt_repaired", PIBT_ARM),
        )
        for load in LOADS
        for seed in SEEDS
    )
    study._write_text_exact(
        validation / "validated_outputs.sha256",
        "".join(
            f"{sha256_file(path)}  {path.as_posix()}\n"
            for path in sorted(validated, key=lambda item: item.as_posix())
        ),
    )
    print(f"[complete] {validation / 'summary.json'}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("preflight", "freeze", "run-arm", "summary"))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--manifest-contract", default=str(MANIFEST_CONTRACT))
    parser.add_argument("--zero-base-root", default=str(ZERO_BASE_ROOT))
    parser.add_argument("--adapted-base-root", default=str(ADAPTED_BASE_ROOT))
    parser.add_argument("--pp-repaired-checkpoint", default=str(PP_REPAIRED_CHECKPOINT))
    parser.add_argument("--pp-repaired-psi-head", default=str(PP_REPAIRED_PSI_HEAD))
    parser.add_argument("--pp-scale-contract", default=str(PP_SCALE_CONTRACT))
    parser.add_argument("--pibt-source-checkpoint", default=str(PIBT_SOURCE_CHECKPOINT))
    parser.add_argument("--pibt-source-psi-head", default=str(PIBT_SOURCE_PSI_HEAD))
    parser.add_argument("--pibt-scale-contract", default=str(PIBT_SCALE_CONTRACT))
    parser.add_argument("--pibt-repaired-checkpoint", required=False, default=str(PIBT_LONGRISK_REPAIR_ROOT / "best_long_risk_world_model.pt"))
    parser.add_argument("--pibt-rebound-psi-head", required=False, default=str(PIBT_REBOUND_ROOT / "best_station_congestion_head.pt"))
    parser.add_argument("--require-training-data", action="store_true")
    parser.add_argument("--arm", choices=(PP_ARM, PIBT_ARM))
    parser.add_argument("--load", choices=LOADS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--ticks", type=int, default=TICKS)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.mode == "preflight":
        preflight(args)
    elif args.mode == "freeze":
        freeze(args)
    elif args.mode == "run-arm":
        if args.arm is None or args.load is None or args.seed is None:
            raise SystemExit("--mode run-arm requires --arm, --load, and --seed")
        run_one(args)
    elif args.mode == "summary":
        summarize(args)


if __name__ == "__main__":
    main()
