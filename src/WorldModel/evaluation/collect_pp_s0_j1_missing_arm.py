"""Collect the missing PP ``S0+J1`` factorial arm.

The existing 691--700 PP campaign contains S0+J0, S1+J0, and S1+J1, but
does not contain the J1-only cell S0+J1.  This script evaluates exactly one
new policy cell:

* S0: the frozen Phase-C robot scorer (no long-risk conversion);
* J1: one static ascending-:math:`J` context order per proposal batch;
* PP: the ``PrioritizedPathPlanner`` from ``Config/world_model_config_PP_48_*``;
* physical-only station admission, as in the existing PP campaign.

The script never changes the existing campaign.  By default it writes to a
new sibling directory and materializes a self-contained copy of every paired
order manifest.  If the original manifests are not present in the checkout,
``--recover-manifests`` can regenerate them with Greedy and accepts them only
when their canonical content hash agrees with the already recorded Greedy
result.  A hash mismatch aborts the run.

Typical workflow (prepare once, then run cells in parallel):

    python -m WorldModel.evaluation.collect_pp_s0_j1_missing_arm \
        --mode prepare --recover-manifests
    python -m WorldModel.evaluation.collect_pp_s0_j1_missing_arm \
        --mode run --load high --seed 691
    python -m WorldModel.evaluation.collect_pp_s0_j1_missing_arm \
        --mode summary

The ``run`` mode accepts no planner override: the PP planner is taken from
the frozen PP configuration, so this is not a PP-to-PIBT transfer experiment.
Run jobs only after ``--mode prepare`` has created the shared protocol; this
prevents concurrent jobs from creating incompatible partial protocols.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

# Support both the documented ``python -m ...`` invocation and direct
# execution via the file path.  The latter otherwise places only
# ``WorldModel/evaluation`` on sys.path, hiding the repository-level
# ``Policies`` and ``WorldState`` packages.  Inserting the repository root is
# harmless when it is already present.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from Policies.TaskAssigner import GreedyTaskAssigner
from Policies.TaskAssigner.WorldModelTaskAssigner.psi_dispatch_context_assigner import (
    PsiDispatchContextWorldModelTaskAssigner,
)
from WorldModel.core.long_risk_schema import (
    LONG_RISK_SCHEMA_VERSION,
    long_risk_runtime_contract,
)
from WorldModel.evaluation.evaluate_online_v6 import _run_one_assigner
from WorldModel.evaluation.phase_c_long_risk_head_repair_protocol import (
    LOAD_CONFIGS,
    LOADS,
    ONLINE_TEST_SEEDS,
    PHASEC_CONFIG,
    PSI_SCALE_CONTRACT,
    REBOUND_PSI_HEAD,
    REPAIRED_CHECKPOINT,
    TICKS,
    TOP_M,
)
from WorldModel.evaluation.run_phase_c_s1_j1_long_risk_correction import (
    _max_mapping,
    _physical_only_audit_contract,
)
from WorldModel.evaluation.run_phase_c_station_admission_restoration import (
    _validate_manifest,
)
from WorldState.station_state import STATION_ADMISSION_PHYSICAL_ONLY


ARM_KEY = "s0_j1"
ARM_LABEL = "PhaseCS0StaticJ1"
OUTPUT_SCHEMA_VERSION = "phase_c_pp_s0_j1_missing_arm_output_v1"
PROTOCOL_SCHEMA_VERSION = "phase_c_pp_s0_j1_missing_arm_protocol_v1"
SUMMARY_SCHEMA_VERSION = "phase_c_pp_s0_j1_missing_arm_summary_v1"

SOURCE_ROOT = Path(
    "WorldModel/checkpoints/phaseC_wm_onpolicy_longrisk_head_repair_681_700_v1/"
    "paired_online_691_700_v1"
)
DEFAULT_OUTPUT_ROOT = SOURCE_ROOT.parent / "paired_online_691_700_s0_j1_v1"

PP_PLANNER_NAME = "PrioritizedPathPlanner"
PP_PLANNER_PARAMS = {"max_horizon": 100, "goal_reserve": 6}
ACTION_PATH_MODE = 0
ACTION_ROUTE_ENCODING = "canonical_bfs_service_cell_v1"

SUMMARY_METRICS = (
    "completed_orders",
    "completed_tasks",
    "avg_task_duration",
    "avg_excess_delay",
    "open_order_count",
    "pending_order_count",
    "stall_ratio_mean",
    "stall_ratio_max",
    "deadlock_ratio_mean",
    "deadlock_ratio_max",
    "congestion_events",
    "severe_events",
    "risk_rate_per_100",
    "assignment_time_ms_mean",
    "wall_time_s",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json_exact(path: Path, payload: Mapping[str, Any]) -> None:
    """Write once, or require an existing file to be byte-identical."""

    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite changed file: {path}")
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


def manifest_path(root: Path, load: str, seed: int) -> Path:
    return root / "order_manifests" / f"orders_{load}_seed{seed}.json"


def output_path(root: Path, load: str, seed: int) -> Path:
    return root / "per_arm" / ARM_KEY / f"{load}_seed{seed}.json"


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        result = float(value)
        if result == result and abs(result) != float("inf"):
            return result
    return None


def _mean_std(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "std": None}
    mean = sum(values) / len(values)
    if len(values) < 2:
        std = 0.0
    else:
        std = (
            sum((value - mean) ** 2 for value in values) / (len(values) - 1)
        ) ** 0.5
    return {"n": len(values), "mean": mean, "std": std}


def _reference_greedy(source_root: Path, load: str, seed: int) -> dict[str, Any] | None:
    """Read the manifest hash recorded by the existing Greedy result."""

    path = source_root / "per_arm" / "greedy" / f"{load}_seed{seed}.json"
    if not path.is_file():
        return None
    payload = read_json(path)
    meta = payload.get("meta") or {}
    manifest = payload.get("manifest") or {}
    metrics = payload.get("metrics") or {}
    expected_hash = (
        manifest.get("sha256")
        or manifest.get("content_sha256")
        or metrics.get("order_arrival_manifest_sha256")
    )
    expected_count = manifest.get("total_orders")
    if expected_count is None:
        expected_count = metrics.get("order_arrival_count")
    if not expected_hash or expected_count is None:
        raise ValueError(f"Greedy result lacks manifest metadata: {path}")
    return {
        "path": path.as_posix(),
        "file_sha256": sha256_file(path),
        "content_sha256": str(expected_hash),
        "total_orders": int(expected_count),
        "arm": meta.get("arm") or meta.get("arm_label"),
    }


def _validate_manifest_against_reference(
    path: Path,
    reference: Mapping[str, Any] | None,
) -> dict[str, Any]:
    manifest = _validate_manifest(path)
    if reference is None:
        raise FileNotFoundError(
            "a paired manifest requires an existing Greedy reference output; "
            f"none was found for {path}"
        )
    checks = {
        "content_sha256": manifest.get("manifest_sha256")
        == reference.get("content_sha256"),
        "total_orders": int(manifest.get("total_orders", -1))
        == int(reference.get("total_orders", -2)),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise ValueError(
            f"manifest does not reproduce the recorded Greedy stream: "
            f"{path}; failed={failed}"
        )
    return manifest


def _candidate_manifest_paths(
    source_root: Path,
    output_root: Path,
    manifest_root: Path | None,
    load: str,
    seed: int,
) -> list[Path]:
    candidates: list[Path] = []
    roots: list[Path] = []
    if manifest_root is not None:
        roots.append(manifest_root)
    roots.append(source_root)
    # Include a previously prepared local copy so that ``prepare`` is
    # idempotent even when the original campaign manifests are unavailable.
    roots.append(output_root)
    for root in roots:
        filename = f"orders_{load}_seed{seed}.json"
        # Accept either a bundle root (``.../order_manifests`` nested below)
        # or the manifest directory itself, independent of its basename.
        candidates.append(root / filename)
        candidates.append(root / "order_manifests" / filename)

    # Some historical payloads retain an absolute/relative manifest path even
    # when the manifest directory was not copied into the checkout.
    reference_path = (
        source_root / "per_arm" / "greedy" / f"{load}_seed{seed}.json"
    )
    if reference_path.is_file():
        recorded = (read_json(reference_path).get("manifest") or {}).get("path")
        if recorded:
            recorded_path = Path(str(recorded))
            candidates.append(recorded_path)
            if not recorded_path.is_absolute():
                # Historical JSON files were produced from several working
                # directories.  Try the repository root and the source
                # bundle's parent before declaring the manifest unavailable.
                candidates.append(Path.cwd() / recorded_path)
                candidates.append(source_root.parent / recorded_path)
    return list(dict.fromkeys(candidates))


def _materialize_copy(source: Path, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        if sha256_file(destination) != sha256_file(source):
            raise FileExistsError(
                f"destination manifest differs from source: {destination}"
            )
        return destination
    if destination.exists():
        raise FileExistsError(f"destination is not a regular file: {destination}")
    temporary = destination.with_name(f".{destination.name}.copy.tmp")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)
    return destination


def _recover_manifest(
    *,
    output_root: Path,
    load: str,
    seed: int,
    reference: Mapping[str, Any],
    ticks: int,
) -> tuple[Path, dict[str, Any]]:
    """Regenerate a Greedy manifest and accept it only after hash matching."""

    destination = manifest_path(output_root, load, seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        manifest = _validate_manifest_against_reference(destination, reference)
        return destination, manifest

    recovery_dir = output_root / ".manifest_recovery"
    recovery_dir.mkdir(parents=True, exist_ok=True)
    temporary = recovery_dir / f"orders_{load}_seed{seed}.json"
    if temporary.exists():
        temporary.unlink()

    print(f"[recover] Greedy PP manifest load={load} seed={seed}")
    with _physical_only_audit_contract() as holder:
        metrics = _run_one_assigner(
            LOAD_CONFIGS[load],
            GreedyTaskAssigner(),
            int(seed),
            int(ticks),
            trace_label="MissingS0J1ManifestRecoveryGreedyPP",
            save_order_manifest=str(temporary),
        )
    station_probe = holder.get("station_probe")
    if station_probe is None:
        raise RuntimeError("manifest recovery did not attach station audit probe")
    station_audit = station_probe.summary()
    if not bool(station_audit.get("passed")):
        raise RuntimeError(
            f"manifest recovery station audit failed: {load} seed={seed}"
        )
    if not temporary.is_file():
        raise FileNotFoundError(f"Greedy did not write {temporary}")
    manifest = _validate_manifest_against_reference(temporary, reference)
    if int(metrics.get("order_arrival_count", -1)) != int(
        manifest.get("total_orders", -2)
    ):
        raise RuntimeError("recovered manifest order count disagrees with metrics")
    os.replace(temporary, destination)
    return destination, manifest


def prepare_manifests(args: argparse.Namespace) -> dict[tuple[str, int], dict[str, Any]]:
    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    explicit_manifest_root = (
        Path(args.manifest_root) if args.manifest_root else None
    )
    records: dict[tuple[str, int], dict[str, Any]] = {}
    missing: list[str] = []

    for load in args.loads:
        for seed in args.seeds:
            reference = _reference_greedy(source_root, load, seed)
            found: Path | None = None
            for candidate in _candidate_manifest_paths(
                source_root, output_root, explicit_manifest_root, load, seed
            ):
                if candidate.is_file():
                    found = candidate
                    break

            recovered = False
            if found is None:
                if not args.recover_manifests:
                    missing.append(f"{load}:seed{seed}")
                    continue
                if reference is None:
                    raise FileNotFoundError(
                        "cannot recover a paired manifest without the existing "
                        f"Greedy reference result: {load} seed={seed}"
                    )
                found, manifest = _recover_manifest(
                    output_root=output_root,
                    load=load,
                    seed=seed,
                    reference=reference,
                    ticks=args.ticks,
                )
                recovered = True
            else:
                manifest = _validate_manifest_against_reference(found, reference)

            local = manifest_path(output_root, load, seed)
            if found.resolve() != local.resolve():
                _materialize_copy(found, local)
                manifest = _validate_manifest_against_reference(local, reference)
            records[(load, seed)] = {
                "load": load,
                "seed": int(seed),
                "source_path": found.as_posix(),
                "local_path": local.as_posix(),
                "source_file_sha256": sha256_file(found),
                "local_file_sha256": sha256_file(local),
                "content_sha256": manifest.get("manifest_sha256"),
                "total_orders": int(manifest.get("total_orders", 0)),
                "recovered_by_greedy": bool(recovered),
                "greedy_reference": dict(reference) if reference else None,
            }

    if missing:
        raise FileNotFoundError(
            "paired PP manifests are missing for: "
            + ", ".join(missing)
            + ". Restore them or rerun with --recover-manifests."
        )
    return records


def dry_run_prepare(args: argparse.Namespace) -> None:
    """Inspect manifest availability without creating files or simulations."""

    source_root = Path(args.source_root)
    output_root = Path(args.output_root)
    explicit_manifest_root = (
        Path(args.manifest_root) if args.manifest_root else None
    )
    found_count = 0
    missing_count = 0
    for load in args.loads:
        for seed in args.seeds:
            reference = _reference_greedy(source_root, load, seed)
            found: Path | None = None
            for candidate in _candidate_manifest_paths(
                source_root, output_root, explicit_manifest_root, load, seed
            ):
                if candidate.is_file():
                    found = candidate
                    break
            if found is None:
                missing_count += 1
                if args.recover_manifests and reference is not None:
                    action = "would recover with Greedy+hash audit"
                elif args.recover_manifests:
                    action = "blocked: Greedy reference JSON is missing"
                else:
                    action = "requires --recover-manifests or --manifest-root"
                print(f"[missing] {load}:seed{seed}; {action}")
                continue
            try:
                manifest = _validate_manifest_against_reference(found, reference)
                found_count += 1
                print(
                    f"[ok] {load}:seed{seed} "
                    f"manifest={found} orders={manifest.get('total_orders')}"
                )
            except Exception as exc:
                missing_count += 1
                print(f"[invalid] {load}:seed{seed} {found}: {exc}")
    print(
        f"[dry-run] checked={found_count + missing_count} "
        f"available={found_count} missing_or_invalid={missing_count}; "
        "no files written and no simulation launched"
    )


def _validate_pp_configs() -> dict[str, dict[str, Any]]:
    configs: dict[str, dict[str, Any]] = {}
    for load in LOADS:
        path = Path(LOAD_CONFIGS[load])
        if not path.is_file():
            raise FileNotFoundError(path)
        payload = read_json(path)
        planner = ((payload.get("policies") or {}).get("path_planner") or {})
        if planner.get("name") != PP_PLANNER_NAME:
            raise ValueError(f"{path} is not a PP configuration: {planner}")
        if dict(planner.get("params") or {}) != PP_PLANNER_PARAMS:
            raise ValueError(f"unexpected PP planner parameters in {path}")
        configs[load] = {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "planner": planner,
        }
    return configs


def _policy_contract() -> dict[str, Any]:
    body: dict[str, Any] = {
        "arm": ARM_KEY,
        "label": ARM_LABEL,
        "policy_family": "world_model_static_j1",
        "checkpoint": Path(REPAIRED_CHECKPOINT).as_posix(),
        "checkpoint_sha256": sha256_file(REPAIRED_CHECKPOINT),
        "robot_selector": "phasec_s0",
        "context_scheduler": "static_j1",
        "psi_context_mode": "j_ascending",
        "j_refresh": "once_per_proposal_batch",
        "psi_head_checkpoint": Path(REBOUND_PSI_HEAD).as_posix(),
        "psi_head_checkpoint_sha256": sha256_file(REBOUND_PSI_HEAD),
        "psi_scale_contract": Path(PSI_SCALE_CONTRACT).as_posix(),
        "psi_scale_contract_sha256": sha256_file(PSI_SCALE_CONTRACT),
        "energy_drift_signal": "none",
        "energy_scoring_mode": "off",
        "action_path_mode": ACTION_PATH_MODE,
        "action_route_encoding": ACTION_ROUTE_ENCODING,
        "path_planner": PP_PLANNER_NAME,
        "path_planner_params": dict(PP_PLANNER_PARAMS),
        "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
        "in_transit_committed_cap": None,
    }
    return {**body, "fingerprint_sha256": canonical_sha256(body)}


def _make_assigner() -> PsiDispatchContextWorldModelTaskAssigner:
    """Construct the explicit S0+J1 policy."""

    return PsiDispatchContextWorldModelTaskAssigner(
        psi_head_checkpoint=str(REBOUND_PSI_HEAD),
        psi_scale_contract=str(PSI_SCALE_CONTRACT),
        psi_context_mode="j_ascending",
        psi_trace_enabled=False,
        psi_trace_max_records=0,
        # This is the deliberate factorial opt-in: J1 is active, while the
        # within-context scorer remains the certified Phase-C S0 scorer.
        allow_phasec_s0_robot_scorer=True,
        checkpoint_path=str(REPAIRED_CHECKPOINT),
        top_m=TOP_M,
        action_path_mode=ACTION_PATH_MODE,
        energy_conv_random_flip_seed=0,
        **dict(PHASEC_CONFIG),
    )


def _audit(
    metrics: Mapping[str, Any],
    manifest: Mapping[str, Any],
    station_audit: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_sha = sha256_file(REPAIRED_CHECKPOINT)
    checks: dict[str, bool] = {
        "manifest_replayed": bool(metrics.get("order_arrival_replayed")),
        "manifest_hash": metrics.get("order_arrival_manifest_sha256")
        == manifest.get("manifest_sha256"),
        "manifest_count": int(metrics.get("order_arrival_count", -1))
        == int(manifest.get("total_orders", -2)),
        "physical_only_mode": station_audit.get("mode")
        == STATION_ADMISSION_PHYSICAL_ONLY,
        "station_audit_passed": bool(station_audit.get("passed")),
        "physical_capacity_clean": int(
            station_audit.get("physical_capacity_violation_count", -1)
        )
        == 0,
        "no_committed_capacity_contract": station_audit.get(
            "committed_capacity_contract"
        )
        == "not enforced for in-transit commitments",
        "model_used": int(metrics.get("model_assign_calls", 0)) > 0,
        "no_greedy_fallback": int(metrics.get("fallback_greedy_calls", 0)) == 0,
        "long_risk_schema": metrics.get("long_risk_schema_version")
        == LONG_RISK_SCHEMA_VERSION,
        "long_risk_contract": metrics.get("long_risk_runtime_contract")
        == long_risk_runtime_contract(),
        "action_path_mode": int(metrics.get("action_path_mode", -1))
        == ACTION_PATH_MODE,
        "action_route_encoding": metrics.get("action_route_encoding")
        == ACTION_ROUTE_ENCODING,
        "s0_energy_mode_off": metrics.get("energy_scoring_mode") == "off",
        "s0_no_conversion_contexts": int(
            metrics.get("energy_conv_contexts", 0)
        )
        == 0,
        "s0_robot_variant": metrics.get("psi_dispatch_robot_scorer_variant")
        == "phasec_s0",
        "s0_not_reported_as_s1": not bool(
            metrics.get("psi_dispatch_s1_within_context")
        ),
        "j1_mode": metrics.get("psi_dispatch_mode") == "j_ascending",
        "j1_head_loaded": bool(metrics.get("psi_dispatch_head_loaded")),
        "j1_encoder_contract_verified": bool(
            metrics.get("psi_dispatch_encoder_contract_verified")
        ),
        "j1_head_hash": metrics.get("psi_dispatch_head_checkpoint_sha256")
        == sha256_file(REBOUND_PSI_HEAD),
        "j1_scale_hash": metrics.get("psi_dispatch_scale_contract_sha256")
        == sha256_file(PSI_SCALE_CONTRACT),
        "j1_evaluated": int(metrics.get("psi_dispatch_eval_calls", 0)) > 0,
        "j1_contexts_seen": int(metrics.get("psi_dispatch_contexts_seen", 0))
        > 0,
        "j1_parent_scorer_preserved": metrics.get("psi_dispatch_robot_scorer")
        == "WorldModelTaskAssigner.select_robots_unmodified",
        "no_no_assign_candidate": not bool(
            metrics.get("psi_dispatch_no_assign_added")
        ),
        "no_hard_gate": not bool(metrics.get("psi_dispatch_hard_gate_added")),
        "no_demand_mutation": not bool(
            metrics.get("psi_dispatch_e_demand_modified")
        ),
        "no_attention_extension": not bool(
            metrics.get("psi_dispatch_attention_added")
        ),
        # ``world_model_path_planner_injected`` means that the model's action
        # preview shares the engine planner object; it is expected to be true
        # for native PP and is not the evaluator's override flag.  Native PP
        # is established by the frozen config audit above, so do not use this
        # field to distinguish PP from PIBT.
        "j1_encoder_binding": metrics.get(
            "psi_dispatch_source_encoder_checkpoint_sha256"
        )
        == checkpoint_sha,
    }
    return {"passed": all(checks.values()), "checks": checks}


def _resume_ok(
    path: Path,
    *,
    load: str,
    seed: int,
    manifest: Mapping[str, Any],
    policy: Mapping[str, Any],
    protocol_sha256: str,
) -> bool:
    if not path.is_file():
        return False
    payload = read_json(path)
    meta = payload.get("meta") or {}
    embedded = payload.get("manifest") or {}
    checks = {
        "schema": payload.get("schema_version") == OUTPUT_SCHEMA_VERSION,
        "protocol": meta.get("protocol_sha256") == protocol_sha256,
        "arm": meta.get("arm_key") == ARM_KEY,
        "load": meta.get("load") == load,
        "seed": int(meta.get("seed", -1)) == seed,
        "ticks": int(meta.get("ticks", -1)) == TICKS,
        "manifest": embedded.get("content_sha256")
        == manifest.get("manifest_sha256"),
        "policy": (meta.get("policy_contract") or {}).get(
            "fingerprint_sha256"
        )
        == policy.get("fingerprint_sha256"),
        "audit": bool((payload.get("audit") or {}).get("passed")),
    }
    if all(checks.values()):
        print(f"[resume] {path}")
        return True
    failed = [name for name, passed in checks.items() if not passed]
    raise ValueError(f"incompatible existing output {path}: {failed}")


def run_cell(
    *,
    output_root: Path,
    load: str,
    seed: int,
    manifest_record: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> Path:
    local_manifest = Path(str(manifest_record["local_path"]))
    manifest = _validate_manifest(local_manifest)
    expected_manifest_sha = str(manifest_record.get("content_sha256", ""))
    if expected_manifest_sha and manifest.get("manifest_sha256") != expected_manifest_sha:
        raise ValueError(
            f"manifest lineage changed before run: {local_manifest}; "
            f"expected={expected_manifest_sha}, "
            f"got={manifest.get('manifest_sha256')}"
        )
    policy = _policy_contract()
    output = output_path(output_root, load, seed)
    if _resume_ok(
        output,
        load=load,
        seed=seed,
        manifest=manifest,
        policy=policy,
        protocol_sha256=str(protocol["protocol_sha256"]),
    ):
        return output

    print(f"[run] arm={ARM_KEY} load={load} seed={seed} planner={PP_PLANNER_NAME}")
    assigner = _make_assigner()
    with _physical_only_audit_contract() as holder:
        metrics = _run_one_assigner(
            LOAD_CONFIGS[load],
            assigner,
            int(seed),
            TICKS,
            trace_label=ARM_LABEL,
            recorded_orders_path=str(local_manifest),
        )
    metrics.update(assigner.psi_dispatch_metrics())
    station_probe = holder.get("station_probe")
    if station_probe is None:
        raise RuntimeError("S0+J1 run did not attach station audit probe")
    station_audit = station_probe.summary()
    metrics.update({
        "station_admission_mode": STATION_ADMISSION_PHYSICAL_ONLY,
        "station_capacity_rejections": int(
            station_audit.get("capacity_rejections", 0)
        ),
        "station_max_occupancy": _max_mapping(
            station_audit.get("max_occupancy")
        ),
        "station_max_committed_load": _max_mapping(
            station_audit.get("max_committed_load")
        ),
        "station_ticks_any_committed_over_capacity": int(
            station_audit.get("ticks_with_any_committed_over_capacity", 0)
        ),
    })
    audit = _audit(metrics, manifest, station_audit)
    if not audit["passed"]:
        failed = [name for name, passed in audit["checks"].items() if not passed]
        raise RuntimeError("S0+J1 PP audit failed: " + ", ".join(failed))

    payload = {
        "schema_version": OUTPUT_SCHEMA_VERSION,
        "meta": {
            "protocol_sha256": protocol["protocol_sha256"],
            "protocol_path": protocol["protocol_path"],
            "arm_key": ARM_KEY,
            "arm": ARM_LABEL,
            "arm_label": ARM_LABEL,
            "load": load,
            "seed": int(seed),
            "ticks": TICKS,
            "config": LOAD_CONFIGS[load],
            "path_planner": PP_PLANNER_NAME,
            "path_planner_params": dict(PP_PLANNER_PARAMS),
            "station_admission": STATION_ADMISSION_PHYSICAL_ONLY,
            "manifest_mode": "replayed",
            "policy_contract": policy,
        },
        "manifest": {
            "path": local_manifest.as_posix(),
            "sha256": manifest.get("manifest_sha256"),
            "file_sha256": sha256_file(local_manifest),
            "content_sha256": manifest.get("manifest_sha256"),
            "total_orders": manifest.get("total_orders"),
        },
        "station_audit": station_audit,
        "audit": audit,
        "metrics": metrics,
        # Keep the same convenience aliases used by the existing online
        # bundle, so generic validation/plotting utilities can consume this
        # newly collected arm without a schema adapter.
        "arm": ARM_LABEL,
        "arm_label": ARM_LABEL,
        "load": load,
        "seed": int(seed),
        "ticks": TICKS,
        "manifest_mode": "replayed",
        "manifest_path": local_manifest.as_posix(),
        "policy_contract": policy,
        "passed": bool(audit["passed"]),
        "checks": audit["checks"],
    }
    write_json_exact(output, payload)
    print(f"[done] {output}")
    return output


def write_protocol(
    *,
    output_root: Path,
    source_root: Path,
    manifest_records: Mapping[tuple[str, int], Mapping[str, Any]],
    config_records: Mapping[str, Mapping[str, Any]],
    ticks: int,
) -> dict[str, Any]:
    protocol_path = output_root / "s0_j1_pp_protocol.json"
    requested_loads = tuple(
        load for load in LOADS
        if any(key[0] == load for key in manifest_records)
    )
    requested_seeds = tuple(sorted({int(key[1]) for key in manifest_records}))
    body: dict[str, Any] = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "purpose": "collect the missing PP S0+J1 factorial cell",
        "arm": {
            "key": ARM_KEY,
            "label": ARM_LABEL,
            "robot_selector": "S0 / Phase-C scorer",
            "context_scheduler": "J1 / static ascending-J once per proposal batch",
            "factorial_cell": "S0+J1",
        },
        "source_root": source_root.as_posix(),
        "output_root": output_root.as_posix(),
        "loads": list(requested_loads),
        "seeds": list(requested_seeds),
        "ticks": int(ticks),
        "planner": {
            "name": PP_PLANNER_NAME,
            "params": dict(PP_PLANNER_PARAMS),
            "override_passed_to_evaluator": False,
        },
        "action_encoding": {
            "action_path_mode": ACTION_PATH_MODE,
            "route_encoding": ACTION_ROUTE_ENCODING,
        },
        "station_admission": {
            "mode": STATION_ADMISSION_PHYSICAL_ONLY,
            "physical_occupancy_cap_enforced": True,
            "in_transit_committed_cap_enforced": False,
            "station_queue": False,
            "eta_controller": False,
        },
        "checkpoint": {
            "world_model": Path(REPAIRED_CHECKPOINT).as_posix(),
            "world_model_sha256": sha256_file(REPAIRED_CHECKPOINT),
            "j1_head": Path(REBOUND_PSI_HEAD).as_posix(),
            "j1_head_sha256": sha256_file(REBOUND_PSI_HEAD),
            "j1_scale_contract": Path(PSI_SCALE_CONTRACT).as_posix(),
            "j1_scale_contract_sha256": sha256_file(PSI_SCALE_CONTRACT),
            "long_risk_runtime_contract": long_risk_runtime_contract(),
        },
        "configs": dict(config_records),
        "policy_contract": _policy_contract(),
        "manifest_lineage": [
            dict(manifest_records[key])
            for key in sorted(manifest_records, key=lambda item: (item[0], item[1]))
        ],
        "safety": {
            "existing_outputs_overwritten": False,
            "manifest_hash_must_match_recorded_greedy": True,
            "training_on_online_test_seeds": False,
            "planner_transfer": False,
        },
    }
    protocol_sha = canonical_sha256(body)
    payload = {**body, "protocol_sha256": protocol_sha}
    write_json_exact(protocol_path, payload)
    return {
        "protocol_path": protocol_path.as_posix(),
        "protocol_sha256": protocol_sha,
        "payload": payload,
    }


def load_protocol(output_root: Path) -> dict[str, Any]:
    path = output_root / "s0_j1_pp_protocol.json"
    payload = read_json(path)
    body = dict(payload)
    claimed = str(body.pop("protocol_sha256", ""))
    if claimed != canonical_sha256(body):
        raise ValueError(f"S0+J1 protocol hash mismatch: {path}")
    if payload.get("schema_version") != PROTOCOL_SCHEMA_VERSION:
        raise ValueError(f"wrong S0+J1 protocol schema: {path}")
    return {
        "protocol_path": path.as_posix(),
        "protocol_sha256": claimed,
        "payload": payload,
    }


def _protocol_manifest_records(
    protocol: Mapping[str, Any],
    loads: Sequence[str],
    seeds: Sequence[int],
) -> dict[tuple[str, int], dict[str, Any]]:
    """Return the requested manifest lineage cells from a frozen protocol.

    A run may be launched one cell at a time after a full ``prepare``.  In
    that workflow the protocol must remain byte-for-byte frozen; rebuilding it
    from the single requested cell would invalidate the provenance hash.
    """

    payload = protocol.get("payload") or {}
    records = payload.get("manifest_lineage") or []
    index: dict[tuple[str, int], dict[str, Any]] = {}
    for row in records:
        if not isinstance(row, Mapping):
            continue
        key = (str(row.get("load")), int(row.get("seed", -1)))
        index[key] = dict(row)
    missing = [
        f"{load}:seed{seed}"
        for load in loads
        for seed in seeds
        if (load, int(seed)) not in index
    ]
    if missing:
        raise FileNotFoundError(
            "frozen S0+J1 protocol lacks requested manifest cells: "
            + ", ".join(missing)
            + ". Run --mode prepare for those cells in a new output root."
        )
    # Verify that the copied manifests still agree with their lineage hashes;
    # this catches accidental edits before any simulation is launched.
    checked: dict[tuple[str, int], dict[str, Any]] = {}
    for load in loads:
        for seed in seeds:
            row = index[(load, int(seed))]
            local = Path(str(row["local_path"]))
            if not local.is_file():
                raise FileNotFoundError(local)
            manifest = _validate_manifest(local)
            expected = str(row.get("content_sha256", ""))
            if expected and manifest.get("manifest_sha256") != expected:
                raise ValueError(
                    f"frozen manifest hash changed: {local}; "
                    f"expected={expected}, got={manifest.get('manifest_sha256')}"
                )
            checked[(load, int(seed))] = row
    return checked


def _verify_frozen_runtime_contract(
    protocol: Mapping[str, Any],
    current_source_root: Path | None = None,
) -> None:
    """Reject runtime artifact drift after the protocol was prepared."""

    payload = protocol.get("payload") or {}
    if current_source_root is not None:
        recorded_source = payload.get("source_root")
        if recorded_source:
            try:
                same_source = (
                    Path(str(recorded_source)).resolve()
                    == current_source_root.resolve()
                )
            except OSError:
                same_source = str(recorded_source) == str(current_source_root)
            if not same_source:
                raise ValueError(
                    "source root changed after protocol preparation: "
                    f"expected={recorded_source}, got={current_source_root}"
                )
    configs = payload.get("configs") or {}
    current_configs = _validate_pp_configs()
    for load, expected in configs.items():
        current = current_configs.get(str(load))
        if current is None or current.get("sha256") != expected.get("sha256"):
            raise ValueError(
                f"PP config changed after protocol preparation: {load}; "
                f"expected={expected.get('sha256')}, "
                f"got={(current or {}).get('sha256')}"
            )

    checkpoint = payload.get("checkpoint") or {}
    artifact_hashes = {
        "world_model": REPAIRED_CHECKPOINT,
        "j1_head": REBOUND_PSI_HEAD,
        "j1_scale_contract": PSI_SCALE_CONTRACT,
    }
    for key, path in artifact_hashes.items():
        expected = str(checkpoint.get(f"{key}_sha256", ""))
        actual = sha256_file(path)
        if expected and actual != expected:
            raise ValueError(
                f"runtime artifact changed after protocol preparation: {path}; "
                f"expected={expected}, got={actual}"
            )
    expected_policy = payload.get("policy_contract") or {}
    actual_policy = _policy_contract()
    if expected_policy.get("fingerprint_sha256") != actual_policy.get(
        "fingerprint_sha256"
    ):
        raise ValueError(
            "S0+J1 policy contract changed after protocol preparation"
        )


def _source_arm_metrics(
    source_root: Path,
    record_index: Mapping[tuple[str, int], Mapping[str, Any]],
    source_arms: Mapping[str, str],
    loads: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, dict[str, dict[str, dict[str, Any]]]]:
    """Load and audit the three existing PP cells used for contrasts."""

    result: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for cell, arm_dir in source_arms.items():
        for load in loads:
            for seed in seeds:
                path = source_root / "per_arm" / arm_dir / f"{load}_seed{seed}.json"
                if not path.is_file():
                    raise FileNotFoundError(
                        f"required PP factorial arm output is missing: {path}"
                    )
                payload = read_json(path)
                if not bool((payload.get("audit") or {}).get("passed")):
                    raise ValueError(f"failed source-arm audit: {path}")
                meta = payload.get("meta") or {}
                declared_config = meta.get("config")
                if (
                    declared_config is not None
                    and declared_config != LOAD_CONFIGS[load]
                ):
                    raise ValueError(
                        f"source arm is not from the frozen PP config family: {path}; "
                        f"config={declared_config}, expected={LOAD_CONFIGS[load]}"
                    )
                if meta.get("station_admission") not in (
                    None,
                    STATION_ADMISSION_PHYSICAL_ONLY,
                ):
                    raise ValueError(
                        f"source arm station-admission mode is not physical-only: {path}"
                    )
                declared_planner = meta.get("path_planner")
                if declared_planner not in (None, PP_PLANNER_NAME):
                    raise ValueError(
                        f"source arm uses a non-PP planner: {path}; "
                        f"planner={declared_planner}"
                    )
                source_manifest = payload.get("manifest") or {}
                source_manifest_sha = source_manifest.get(
                    "content_sha256", source_manifest.get("sha256")
                )
                target_manifest_sha = str(
                    record_index[(load, int(seed))].get("content_sha256", "")
                )
                if source_manifest_sha != target_manifest_sha:
                    raise ValueError(
                        f"source arm is not paired with S0+J1 manifest: {path}; "
                        f"expected={target_manifest_sha}, got={source_manifest_sha}"
                    )
                source_audit = payload.get("audit") or {}
                source_checks = source_audit.get("checks") or {}
                if source_checks.get("manifest_replayed") is not True:
                    raise ValueError(
                        f"source arm was not replayed from a paired manifest: {path}"
                    )
                metrics = payload.get("metrics") or {}
                result.setdefault(cell, {}).setdefault(load, {})[str(seed)] = {
                    metric: metrics.get(metric) for metric in SUMMARY_METRICS
                }
    return result


def _csv_text(rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    from io import StringIO

    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({field: row.get(field) for field in fieldnames})
    return stream.getvalue()


def summarize(args: argparse.Namespace) -> Path:
    output_root = Path(args.output_root)
    protocol = load_protocol(output_root)
    _verify_frozen_runtime_contract(protocol, Path(args.source_root))
    records = protocol["payload"].get("manifest_lineage") or []
    record_index = {
        (str(row["load"]), int(row["seed"])): row for row in records
    }
    rows: list[dict[str, Any]] = []
    for load in args.loads:
        for seed in args.seeds:
            key = (load, int(seed))
            if key not in record_index:
                raise ValueError(f"protocol lacks manifest cell {load}:seed{seed}")
            path = output_path(output_root, load, int(seed))
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = read_json(path)
            if payload.get("schema_version") != OUTPUT_SCHEMA_VERSION:
                raise ValueError(f"wrong output schema: {path}")
            if not bool((payload.get("audit") or {}).get("passed")):
                raise ValueError(f"failed output audit: {path}")
            if (payload.get("meta") or {}).get("protocol_sha256") != protocol[
                "protocol_sha256"
            ]:
                raise ValueError(f"output belongs to a different protocol: {path}")
            if (payload.get("manifest") or {}).get("content_sha256") != str(
                record_index[key].get("content_sha256", "")
            ):
                raise ValueError(f"output manifest is not the frozen paired stream: {path}")
            metrics = payload.get("metrics") or {}
            row = {"arm": ARM_KEY, "load": load, "seed": int(seed)}
            row.update({metric: metrics.get(metric) for metric in SUMMARY_METRICS})
            rows.append(row)

    validation = output_root / "validation"
    validation.mkdir(parents=True, exist_ok=True)
    fields = ["arm", "load", "seed", *SUMMARY_METRICS]
    paired_csv = validation / "paired_rows.csv"
    write_text_exact(paired_csv, _csv_text(rows, fields))

    aggregate: dict[str, Any] = {}
    for load in args.loads:
        aggregate[load] = {}
        load_rows = [row for row in rows if row["load"] == load]
        for metric in SUMMARY_METRICS:
            values = [
                value
                for row in load_rows
                if (value := _number(row.get(metric))) is not None
            ]
            aggregate[load][metric] = _mean_std(values)

    source_root = Path(args.source_root)
    source_arms = {
        "s0_j0": "repaired_phasec",
        "s1_j0": "repaired_combo_j0",
        "s1_j1": "repaired_combo_j1",
    }
    contrasts = _source_arm_metrics(
        source_root,
        record_index,
        source_arms,
        args.loads,
        args.seeds,
    )

    # A compact raw four-cell table is useful for downstream paper scripts:
    # it preserves the three existing PP cells alongside the newly collected
    # cell, before any contrast is computed.
    factorial_rows: list[dict[str, Any]] = []
    source_cell_labels = {
        "s0_j0": "repaired_phasec",
        "s1_j0": "repaired_combo_j0",
        "s1_j1": "repaired_combo_j1",
    }
    for load in args.loads:
        for seed in args.seeds:
            target = next(
                row
                for row in rows
                if row["load"] == load and row["seed"] == int(seed)
            )
            factorial_rows.append(
                {
                    "cell": "s0_j1",
                    "source_arm": "new_s0_j1",
                    **{
                        field: target.get(field)
                        for field in ["load", "seed", *SUMMARY_METRICS]
                    },
                }
            )
            for cell, arm_dir in source_cell_labels.items():
                source_metrics = (
                    contrasts.get(cell, {})
                    .get(load, {})
                    .get(str(seed))
                )
                if source_metrics is None:
                    continue
                factorial_rows.append(
                    {
                        "cell": cell,
                        "source_arm": arm_dir,
                        "load": load,
                        "seed": int(seed),
                        **source_metrics,
                    }
                )
    factorial_fields = [
        "cell",
        "source_arm",
        "load",
        "seed",
        *SUMMARY_METRICS,
    ]
    factorial_csv = validation / "paired_four_cell_rows.csv"
    write_text_exact(factorial_csv, _csv_text(factorial_rows, factorial_fields))
    expected_factorial_rows = 4 * len(args.loads) * len(args.seeds)
    if len(factorial_rows) != expected_factorial_rows:
        raise RuntimeError(
            "four-cell export is incomplete: "
            f"expected {expected_factorial_rows}, got {len(factorial_rows)}"
        )

    # Paired J1-only and four-cell contrasts are intentionally reported as
    # diagnostics; no claim or threshold is hard-coded here.
    contrast_rows: list[dict[str, Any]] = []
    for load in args.loads:
        for seed in args.seeds:
            new_row = next(
                row
                for row in rows
                if row["load"] == load and row["seed"] == seed
            )
            cells: dict[str, Mapping[str, Any]] = {"s0_j1": new_row}
            for cell, arm_dir in source_arms.items():
                cell_payload = (
                    contrasts.get(cell, {}).get(load, {}).get(str(seed))
                )
                if cell_payload is not None:
                    cells[cell] = cell_payload
            if len(cells) < 4:
                continue
            out = {"load": load, "seed": int(seed)}
            for metric in SUMMARY_METRICS:
                s0j0 = _number(cells["s0_j0"].get(metric))
                s0j1 = _number(cells["s0_j1"].get(metric))
                s1j0 = _number(cells["s1_j0"].get(metric))
                s1j1 = _number(cells["s1_j1"].get(metric))
                out[f"j1_at_s0__{metric}"] = (
                    s0j1 - s0j0
                    if s0j0 is not None and s0j1 is not None
                    else None
                )
                out[f"s1_at_j1__{metric}"] = (
                    s1j1 - s0j1
                    if s1j1 is not None and s0j1 is not None
                    else None
                )
                out[f"interaction__{metric}"] = (
                    (s1j1 - s1j0) - (s0j1 - s0j0)
                    if None not in (s0j0, s0j1, s1j0, s1j1)
                    else None
                )
            contrast_rows.append(out)
    contrast_fields = (
        ["load", "seed"]
        + [f"j1_at_s0__{metric}" for metric in SUMMARY_METRICS]
        + [f"s1_at_j1__{metric}" for metric in SUMMARY_METRICS]
        + [f"interaction__{metric}" for metric in SUMMARY_METRICS]
    )
    contrasts_csv = validation / "paired_four_cell_contrasts.csv"
    write_text_exact(contrasts_csv, _csv_text(contrast_rows, contrast_fields))
    expected_contrast_rows = len(args.loads) * len(args.seeds)
    if len(contrast_rows) != expected_contrast_rows:
        raise RuntimeError(
            "four-cell contrast export is incomplete: "
            f"expected {expected_contrast_rows}, got {len(contrast_rows)}"
        )

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "protocol_path": protocol["protocol_path"],
        "protocol_sha256": protocol["protocol_sha256"],
        "arm": {"key": ARM_KEY, "label": ARM_LABEL},
        "loads": list(args.loads),
        "seeds": [int(seed) for seed in args.seeds],
        "row_count": len(rows),
        "expected_row_count": len(args.loads) * len(args.seeds),
        "factorial_row_count": len(factorial_rows),
        "expected_factorial_row_count": 4 * len(args.loads) * len(args.seeds),
        "aggregate_mean_std": aggregate,
        "source_factorial_arms": source_arms,
        "contrast_semantics": {
            "j1_at_s0": "S0+J1 - S0+J0",
            "s1_at_j1": "S1+J1 - S0+J1",
            "interaction": "(S1+J1-S1+J0) - (S0+J1-S0+J0)",
        },
        "files": {
            "paired_rows": paired_csv.as_posix(),
            "paired_four_cell_rows": factorial_csv.as_posix(),
            "paired_four_cell_contrasts": contrasts_csv.as_posix(),
        },
    }
    summary_path = validation / "summary.json"
    write_json_exact(summary_path, summary)
    print(f"[complete] {summary_path}")
    return summary_path


def write_text_exact(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"refusing to overwrite changed file: {path}")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run", "summary"), required=True)
    parser.add_argument("--source-root", default=str(SOURCE_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--manifest-root",
        default=None,
        help=(
            "optional manifest directory or bundle root; it may contain "
            "orders_<load>_seed<seed>.json directly or under order_manifests/"
        ),
    )
    parser.add_argument(
        "--recover-manifests",
        action="store_true",
        help=(
            "regenerate absent manifests with Greedy and verify their "
            "recorded hashes (disabled by --dry-run)"
        ),
    )
    parser.add_argument("--loads", nargs="+", choices=LOADS, default=list(LOADS))
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(ONLINE_TEST_SEEDS),
    )
    parser.add_argument("--load", choices=LOADS, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--ticks", type=int, default=TICKS)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.ticks != TICKS:
        raise ValueError(f"PP protocol freezes ticks={TICKS}")
    args.loads = tuple(dict.fromkeys(args.loads))
    args.seeds = tuple(dict.fromkeys(int(seed) for seed in args.seeds))
    invalid = [seed for seed in args.seeds if seed not in ONLINE_TEST_SEEDS]
    if invalid:
        raise ValueError(
            f"seed(s) outside frozen PP 691--700 block: {invalid}"
        )
    if args.load is not None:
        args.loads = (args.load,)
    if args.seed is not None:
        if args.seed not in ONLINE_TEST_SEEDS:
            raise ValueError(f"seed outside frozen block: {args.seed}")
        args.seeds = (int(args.seed),)

    _validate_pp_configs()
    for path in (REPAIRED_CHECKPOINT, REBOUND_PSI_HEAD, PSI_SCALE_CONTRACT):
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    output_root = Path(args.output_root)
    source_root = Path(args.source_root)
    # The collector is intentionally additive.  Accidentally pointing the
    # output at the frozen source bundle would risk modifying the historical
    # campaign and, more importantly, would make its provenance ambiguous.
    if output_root.resolve() == source_root.resolve():
        raise ValueError(
            "--output-root must be a new directory; refusing to write into "
            "the existing PP campaign source root"
        )
    if args.dry_run:
        if args.mode == "summary":
            print(
                "[dry-run] summary mode would read completed S0+J1 outputs; "
                "no files written"
            )
            return
        dry_run_prepare(args)
        return
    if args.mode == "summary":
        summarize(args)
        return

    protocol_file = output_root / "s0_j1_pp_protocol.json"
    if args.mode == "run":
        if not protocol_file.is_file():
            raise FileNotFoundError(
                f"{protocol_file} is missing. Run --mode prepare first "
                "(preferably for all low/mid/high and seeds 691--700) so "
                "parallel cell jobs share one frozen protocol."
            )
        # Reuse the frozen protocol created by ``--mode prepare``.  This is
        # what permits independent Slurm jobs to run individual cells without
        # changing the protocol hash on every invocation.
        protocol = load_protocol(output_root)
        _verify_frozen_runtime_contract(protocol, source_root)
        records = _protocol_manifest_records(
            protocol, args.loads, args.seeds
        )
        print(
            f"[protocol] reused {protocol['protocol_path']} "
            f"protocol_sha256={protocol['protocol_sha256']}"
        )
    else:
        records = prepare_manifests(args)
        config_records = _validate_pp_configs()
        protocol = write_protocol(
            output_root=output_root,
            source_root=source_root,
            manifest_records=records,
            config_records=config_records,
            ticks=args.ticks,
        )
        recovered_count = sum(
            bool(row.get("recovered_by_greedy")) for row in records.values()
        )
        print(
            f"[prepared] cells={len(records)} "
            f"recovered_manifests={recovered_count} "
            f"protocol_sha256={protocol['protocol_sha256']}"
        )
        if args.mode == "prepare" or args.dry_run:
            return

    if args.dry_run:
        print(
            f"[dry-run] would run {len(args.loads) * len(args.seeds)} "
            f"S0+J1 PP cells"
        )
        return

    for load in args.loads:
        for seed in args.seeds:
            run_cell(
                output_root=output_root,
                load=load,
                seed=int(seed),
                manifest_record=records[(load, int(seed))],
                protocol=protocol,
            )


if __name__ == "__main__":
    main()
