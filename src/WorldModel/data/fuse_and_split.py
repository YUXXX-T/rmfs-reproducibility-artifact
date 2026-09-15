"""
Data Fusion, Deduplication & Stratified Split
==============================================
Merge multiple .pt data files from different generation runs,
detect duplicates, namespace group_ids, and produce a stratified
group-level train/val/test split.

Usage:
    python -m WorldModel.data.fuse_and_split \
      --inputs DataGen/wm_data/run1/data.pt DataGen/wm_data/run2/data.pt \
      --long-risk-inputs DataGen/wm_data/run1/long_risk.pt DataGen/wm_data/run2/long_risk.pt \
      --output-dir DataGen/wm_data/fused_v1 \
      --train-ratio 0.7 --val-ratio 0.15 --split-seed 42
"""

import argparse
import glob as globmod
import hashlib
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


REQUIRED_KEYS = {
    "node_history", "edge_index", "edge_features", "demand_context",
    "action_node", "action_global", "action_edge",
    "future_node_labels", "future_system_labels", "future_station_labels",
    "future_mask", "realized_cost", "heuristic_cost",
    "candidate_group_id", "decision_tick", "candidate_info", "fixed_context",
    "station_node_ids",
}

LONG_RISK_KEYS = (
    "risk_peak",
    "risk_cvar",
    "risk_terminal",
    "risk_delta_group",
    "risk_event",
)

LONG_RISK_RANK_DIMS = (
    "risk_peak",
    "risk_cvar",
    "risk_terminal",
    "risk_delta_group",
)

LYAPUNOV_COLLECTION_SCHEMA_KEY = "lyapunov_l0_collection_schema_version"
LEGACY_LYAPUNOV_COLLECTION_SCHEMA = "lyapunov_l0_collection_v1_legacy"
LYAPUNOV_COLLECTION_TO_SNAPSHOT_SCHEMA = {
    "lyapunov_l0_collection_v2": "lyapunov_l0_snapshot_v2",
    "lyapunov_l1_collection_v3": "lyapunov_l1_snapshot_v3",
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tensor_hash(t: torch.Tensor) -> str:
    return hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]


def _dict_hash(d: dict) -> str:
    return hashlib.sha256(
        json.dumps(d, sort_keys=True, default=str).encode()
    ).hexdigest()[:16]


def _find_meta_file(pt_path: str, pattern: str) -> Optional[str]:
    """Search for meta files matching pattern in the same directory as pt_path.

    Priority:
      1. File whose name contains the .pt stem (e.g., gen_config_high_load_seed42.json)
      2. If only one match exists, use it directly
      3. Otherwise None
    """
    directory = os.path.dirname(os.path.abspath(pt_path))
    pt_stem = Path(pt_path).stem
    candidates = globmod.glob(os.path.join(directory, pattern))
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    for c in candidates:
        if pt_stem in os.path.basename(c):
            return c
    for c in candidates:
        base = os.path.basename(c)
        seeds = re.findall(r"seed\d+", pt_stem)
        loads = re.findall(r"(?:high|medium|low|light|heavy)_load", pt_stem)
        if any(s in base for s in seeds) or any(l in base for l in loads):
            return c
    return None


def _infer_run_id(pt_path: str, gen_config: Optional[dict],
                  train_meta: Optional[dict]) -> str:
    """Infer run_id from meta or filename."""
    if gen_config:
        for key in ("gen_config_name", "scenario_name", "run_id"):
            if key in gen_config and gen_config[key]:
                return str(gen_config[key])
    if train_meta:
        for key in ("gen_config_name", "scenario_name", "run_id"):
            if key in train_meta and train_meta[key]:
                return str(train_meta[key])
    stem = Path(pt_path).stem
    if stem in ("wm_train_data", "data"):
        parent = Path(pt_path).parent.name
        if parent and parent not in ("wm_data", "DataGen"):
            return parent
    return stem


def _extract_seed_and_load(run_id: str, gen_config: Optional[dict],
                           train_meta: Optional[dict]) -> Tuple[str, str]:
    """Extract source_seed and source_load_level from meta or run_id."""
    seed = "unknown"
    load_level = "unknown"
    for meta in (gen_config, train_meta):
        if meta is None:
            continue
        if "seed" in meta:
            seed = str(meta["seed"])
        if "load_level" in meta:
            load_level = str(meta["load_level"])
    if seed == "unknown":
        m = re.search(r"seed(\d+)", run_id)
        if m:
            seed = m.group(1)
    if load_level == "unknown":
        m = re.search(r"(high|medium|low|light|heavy)_load", run_id)
        if m:
            load_level = m.group(1) + "_load"
    return seed, load_level


# ------------------------------------------------------------------
# Core functions
# ------------------------------------------------------------------

def _extract_shape_signature(sample: dict) -> dict:
    """Extract shape info from a sample for cross-file consistency checks."""
    sig = {}
    for key in ("node_history", "edge_index", "edge_features", "demand_context",
                "action_node", "action_global", "action_edge", "station_node_ids"):
        v = sample.get(key)
        if isinstance(v, torch.Tensor):
            sig[key] = tuple(v.shape)
    return sig


def _detect_lyapunov_collection_schema(samples: List[dict], path: str) -> Optional[str]:
    """Detect one L0 collection schema and reject ambiguous files.

    Historical L0 samples predate an explicit collection tag.  They are
    intentionally classified as a legacy v1 schema, rather than being
    treated as unlabelled data, so a cross-version fuse fails before any output is
    written.
    """
    schemas = set()
    for index, sample in enumerate(samples):
        schema = sample.get(LYAPUNOV_COLLECTION_SCHEMA_KEY)
        has_l0_payload = any(
            key in sample for key in (
                "lyapunov_l0_start",
                "lyapunov_l0_post_action",
                "lyapunov_l0_end",
                "lyapunov_l0_trajectory",
                "lyapunov_l0_progress",
            )
        )
        if schema is not None:
            schemas.add(str(schema))
        elif has_l0_payload:
            schemas.add(LEGACY_LYAPUNOV_COLLECTION_SCHEMA)

        expected_snapshot_schema = LYAPUNOV_COLLECTION_TO_SNAPSHOT_SCHEMA.get(
            str(schema)
        )
        if expected_snapshot_schema is not None and has_l0_payload:
            for endpoint in ("start", "post_action", "end"):
                snapshot = sample.get(f"lyapunov_l0_{endpoint}")
                if not isinstance(snapshot, dict):
                    raise ValueError(
                        f"{path}: sample[{index}] {schema} is missing "
                        f"lyapunov_l0_{endpoint}"
                    )
                if snapshot.get("schema_version") != expected_snapshot_schema:
                    raise ValueError(
                        f"{path}: sample[{index}] {endpoint} snapshot has "
                        f"schema={snapshot.get('schema_version')!r}, expected "
                        f"{expected_snapshot_schema!r}"
                    )
                for required in ("arrival_manifest", "traffic_diagnostics"):
                    if required not in snapshot:
                        raise ValueError(
                            f"{path}: sample[{index}] {schema} {endpoint} snapshot "
                            f"is missing {required!r}"
                        )

    if len(schemas) > 1:
        raise ValueError(
            f"{path}: mixed Lyapunov collection schemas inside one file: "
            f"{sorted(schemas)}"
        )
    return next(iter(schemas), None)


def load_and_validate(path: str) -> dict:
    """Load a .pt file, validate schema, search for meta, infer run_id.

    Returns dict with keys: path, samples, run_id, gen_config, train_meta,
    meta_missing, source_seed, source_load_level, has_meta, shape_signature,
    validation_warnings.
    """
    abs_path = os.path.abspath(path)
    samples = torch.load(abs_path, weights_only=False)
    if not isinstance(samples, list) or len(samples) == 0:
        raise ValueError(f"Expected non-empty list of samples in {path}")

    validation_warnings = []
    bad_samples = 0
    for i, s in enumerate(samples):
        missing = REQUIRED_KEYS - set(s.keys())
        if missing:
            bad_samples += 1
            if bad_samples <= 3:
                validation_warnings.append(
                    f"sample[{i}] missing keys: {missing}"
                )
    if bad_samples > 3:
        validation_warnings.append(
            f"... and {bad_samples - 3} more samples with missing keys"
        )
    if bad_samples > 0:
        print(f"  WARNING: {path}: {bad_samples}/{len(samples)} samples "
              f"have missing required keys", file=sys.stderr)

    shape_sig = _extract_shape_signature(samples[0])
    lyapunov_collection_schema = _detect_lyapunov_collection_schema(
        samples, abs_path
    )

    gc_path = _find_meta_file(abs_path, "gen_config*.json")
    tm_path = _find_meta_file(abs_path, "wm_train_meta*.json")
    gen_config = None
    train_meta = None
    if gc_path:
        with open(gc_path, "r", encoding="utf-8") as f:
            gen_config = json.load(f)
    if tm_path:
        with open(tm_path, "r", encoding="utf-8") as f:
            train_meta = json.load(f)

    has_meta = gen_config is not None or train_meta is not None
    run_id = _infer_run_id(abs_path, gen_config, train_meta)
    seed, load_level = _extract_seed_and_load(run_id, gen_config, train_meta)

    return {
        "path": abs_path,
        "samples": samples,
        "run_id": run_id,
        "gen_config": gen_config,
        "train_meta": train_meta,
        "meta_missing": not has_meta,
        "has_meta": has_meta,
        "source_seed": seed,
        "source_load_level": load_level,
        "shape_signature": shape_sig,
        "lyapunov_collection_schema": lyapunov_collection_schema,
        "validation_warnings": validation_warnings,
    }


def _as_float_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().clone().float().cpu()
    return torch.tensor(value, dtype=torch.float32)


def _to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.item())
    return float(value)


def _load_long_risk_file(path: str) -> List[dict]:
    lr_data = torch.load(os.path.abspath(path), weights_only=False)
    if not isinstance(lr_data, list):
        raise ValueError(f"Expected list of long-risk labels in {path}")
    return lr_data


def _group_long_risk_entries(lr_data: List[dict]) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for entry in lr_data:
        if not isinstance(entry, dict):
            continue
        gid = entry.get("candidate_group_id")
        if gid is None:
            continue
        if any(k not in entry for k in LONG_RISK_KEYS):
            continue
        groups[str(gid)].append(entry)
    return dict(groups)


def _pair_abs_diffs(entries: List[dict], dim: str) -> List[float]:
    vals = [_to_float(e[dim]) for e in entries if dim in e]
    diffs = []
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            d = abs(vals[i] - vals[j])
            if d > 1e-12:
                diffs.append(d)
    return diffs


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    vals = sorted(values)
    idx = int(round((len(vals) - 1) * min(max(q, 0.0), 1.0)))
    return vals[idx]


def _parse_csv_floats(raw: str, expected: int, name: str) -> List[float]:
    vals = [float(x.strip()) for x in raw.split(",") if x.strip()]
    if len(vals) != expected:
        raise ValueError(f"{name} must contain {expected} comma-separated floats")
    return vals


def _parse_csv_dims(raw: str) -> List[str]:
    aliases = {
        "peak": "risk_peak",
        "cvar": "risk_cvar",
        "terminal": "risk_terminal",
        "delta": "risk_delta_group",
        "delta_group": "risk_delta_group",
    }
    dims = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        dim = aliases.get(item, item)
        if dim not in LONG_RISK_RANK_DIMS:
            raise ValueError(f"Unknown long-risk filter dim: {item}")
        dims.append(dim)
    if not dims:
        raise ValueError("At least one long-risk filter dim is required")
    return dims


def _compute_long_risk_epsilons(paths: List[str], args) -> Dict[str, float]:
    dims = _parse_csv_dims(args.long_risk_filter_dims)
    if args.long_risk_epsilon_mode == "fixed":
        return {dim: args.long_risk_fixed_epsilon for dim in dims}

    diffs_by_dim = {dim: [] for dim in dims}
    for path in paths:
        if _is_skip_long_risk_path(path):
            continue
        lr_data = _load_long_risk_file(path)
        groups = _group_long_risk_entries(lr_data)
        for entries in groups.values():
            if len(entries) < 2:
                continue
            for dim in dims:
                diffs_by_dim[dim].extend(_pair_abs_diffs(entries, dim))

    eps = {}
    for dim, diffs in diffs_by_dim.items():
        q = _quantile(diffs, args.long_risk_epsilon_quantile)
        eps[dim] = max(args.long_risk_min_epsilon, q)
    return eps


def _group_risk_score(entries: List[dict], weights: Tuple[float, float, float]) -> float:
    vals = []
    for entry in entries:
        score = (
            weights[0] * _to_float(entry["risk_peak"])
            + weights[1] * _to_float(entry["risk_cvar"])
            + weights[2] * _to_float(entry["risk_terminal"])
        )
        vals.append(score)
    return sum(vals) / max(len(vals), 1)


def _risk_bin(score: float, cuts: Tuple[float, float]) -> str:
    if score < cuts[0]:
        return "safe"
    if score < cuts[1]:
        return "medium"
    return "dangerous"


def _is_informative_group(
    entries: List[dict],
    epsilons: Dict[str, float],
    min_pairs: int,
) -> Tuple[bool, Dict[str, int]]:
    pair_counts = {}
    for dim, eps in epsilons.items():
        count = sum(1 for d in _pair_abs_diffs(entries, dim) if d >= eps)
        pair_counts[dim] = count
    return max(pair_counts.values(), default=0) >= min_pairs, pair_counts


def _apply_long_risk_filter(
    lr_data: List[dict],
    filter_config: Optional[dict],
    source_name: str,
) -> Tuple[List[dict], Optional[dict]]:
    if filter_config is None or filter_config["mode"] == "none":
        return lr_data, None

    groups = _group_long_risk_entries(lr_data)
    rng = random.Random(filter_config["seed"] + int(hashlib.sha256(
        source_name.encode()
    ).hexdigest()[:8], 16))

    informative_gids = set()
    calibration_bins: Dict[str, List[str]] = defaultdict(list)
    risk_scores = {}
    pair_counts_by_gid = {}

    for gid, entries in groups.items():
        is_info, pair_counts = _is_informative_group(
            entries,
            filter_config["epsilons"],
            filter_config["min_pairs"],
        )
        pair_counts_by_gid[gid] = pair_counts
        score = _group_risk_score(entries, filter_config["risk_weights"])
        risk_scores[gid] = score
        if is_info:
            informative_gids.add(gid)
        else:
            calibration_bins[_risk_bin(score, filter_config["risk_bins"])].append(gid)

    uninformative_count = sum(len(v) for v in calibration_bins.values())
    if informative_gids:
        target_calib = int(round(
            len(informative_gids) * filter_config["calibration_ratio"]
            / max(1.0 - filter_config["calibration_ratio"], 1e-6)
        ))
    else:
        target_calib = int(round(uninformative_count * filter_config["calibration_ratio"]))
    target_calib = min(target_calib, uninformative_count)

    selected_calib = set()
    bin_names = ("safe", "medium", "dangerous")
    if target_calib > 0:
        base = target_calib // len(bin_names)
        remainder = target_calib % len(bin_names)
        for idx, bin_name in enumerate(bin_names):
            gids = list(calibration_bins.get(bin_name, []))
            rng.shuffle(gids)
            quota = base + (1 if idx < remainder else 0)
            selected_calib.update(gids[:min(quota, len(gids))])

        # Fill unused quota from bins with spare groups.
        if len(selected_calib) < target_calib:
            spare = []
            for bin_name in bin_names:
                for gid in calibration_bins.get(bin_name, []):
                    if gid not in selected_calib:
                        spare.append(gid)
            rng.shuffle(spare)
            selected_calib.update(spare[:target_calib - len(selected_calib)])

    selected_gids = informative_gids | selected_calib
    filtered = [
        entry for gid, entries in groups.items()
        if gid in selected_gids
        for entry in entries
    ]

    calib_by_bin = {
        name: sum(1 for gid in selected_calib
                  if _risk_bin(risk_scores[gid], filter_config["risk_bins"]) == name)
        for name in bin_names
    }
    all_bins = {
        name: len(calibration_bins.get(name, []))
        for name in bin_names
    }
    stats = {
        "mode": filter_config["mode"],
        "source": source_name,
        "original_labels": len(lr_data),
        "original_groups": len(groups),
        "retained_labels": len(filtered),
        "retained_groups": len(selected_gids),
        "informative_groups": len(informative_gids),
        "calibration_groups": len(selected_calib),
        "dropped_groups": len(groups) - len(selected_gids),
        "risk_bins_available": all_bins,
        "risk_bins_selected": calib_by_bin,
        "epsilons": filter_config["epsilons"],
        "min_pairs": filter_config["min_pairs"],
        "calibration_ratio": filter_config["calibration_ratio"],
    }
    return filtered, stats


def _is_skip_long_risk_path(path: str) -> bool:
    return path.lower() in ("none", "null", "-")


def merge_long_risk_labels(
    samples: List[dict],
    long_risk_path: str,
    filter_config: Optional[dict] = None,
) -> dict:
    """Merge one long-risk label file into one run's samples by candidate_key.

    This must happen before candidate_key namespacing. The fused sample then
    carries embedded long_risk_labels and no longer depends on a separate
    global label file whose keys could collide across runs.
    """
    lr_data = _load_long_risk_file(long_risk_path)
    filtered_lr_data, filter_stats = _apply_long_risk_filter(
        lr_data,
        filter_config,
        os.path.abspath(long_risk_path),
    )

    lr_map = {}
    duplicate_keys = 0
    incomplete = 0
    for entry in filtered_lr_data:
        key = entry.get("candidate_key") if isinstance(entry, dict) else None
        if not key:
            continue
        missing = [k for k in LONG_RISK_KEYS if k not in entry]
        if missing:
            incomplete += 1
            continue
        if key in lr_map:
            duplicate_keys += 1
        lr_map[key] = entry

    matched = 0
    missing_key_samples = 0
    for sample in samples:
        key = sample.get("candidate_key")
        if not key:
            missing_key_samples += 1
            continue
        entry = lr_map.get(key)
        if entry is None:
            continue
        sample["long_risk_labels"] = {
            k: _as_float_tensor(entry[k]) for k in LONG_RISK_KEYS
        }
        matched += 1

    return {
        "path": os.path.abspath(long_risk_path),
        "labels": len(lr_data),
        "labels_after_filter": len(filtered_lr_data),
        "unique_keys": len(lr_map),
        "duplicate_keys": duplicate_keys,
        "incomplete_labels": incomplete,
        "matched_samples": matched,
        "total_samples": len(samples),
        "missing_candidate_key_samples": missing_key_samples,
        "filter": filter_stats,
    }


def ensure_unique_run_ids(runs: List[dict]) -> List[dict]:
    """Ensure all run_ids are unique; auto-suffix __dupN on collision."""
    seen: Dict[str, int] = {}
    for run in runs:
        rid = run["run_id"]
        if rid not in seen:
            seen[rid] = 0
        else:
            seen[rid] += 1
            new_rid = f"{rid}__dup{seen[rid]}"
            print(f"  WARNING: run_id collision '{rid}' -> renamed to '{new_rid}'")
            run["run_id"] = new_rid
    return runs


def compute_sample_fingerprint(sample: dict) -> str:
    """Compute a deterministic fingerprint for dedup (before namespace)."""
    def format_optional_float(value) -> str:
        if value is None:
            return "none"
        return f"{float(value):.8f}"

    parts = [
        str(sample.get("candidate_group_id", "")),
        str(sample.get("decision_tick", "")),
        str(sample.get("action_type", "assign_robot")),
        _dict_hash(sample.get("candidate_info", {})),
        _dict_hash(sample.get("fixed_context", {})),
        format_optional_float(sample.get("realized_cost", 0.0)),
        format_optional_float(sample.get("heuristic_cost", 0.0)),
        str(bool(sample.get("heuristic_cost_valid", True))),
    ]
    ag = sample.get("action_global")
    if ag is not None and isinstance(ag, torch.Tensor):
        parts.append(_tensor_hash(ag))
    dc = sample.get("demand_context")
    if dc is not None and isinstance(dc, torch.Tensor):
        parts.append(_tensor_hash(dc))
    fsl = sample.get("future_system_labels")
    if fsl is not None and isinstance(fsl, torch.Tensor):
        parts.append(_tensor_hash(fsl))
    fstl = sample.get("future_station_labels")
    if fstl is not None and isinstance(fstl, torch.Tensor):
        parts.append(_tensor_hash(fstl))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def _compute_run_fingerprint(fingerprints: List[str]) -> str:
    """Run-level fingerprint from sorted sample fingerprints (with multiplicity)."""
    return hashlib.sha256(
        json.dumps(sorted(fingerprints)).encode()
    ).hexdigest()


def detect_duplicates(runs: List[dict]) -> Tuple[List[dict], List[dict], List[dict]]:
    """Detect exact and suspected duplicates across runs.

    Returns (included, excluded, warnings).
    Each entry in excluded/warnings has keys: run, reason, retained_by.
    """
    for run in runs:
        fps = [compute_sample_fingerprint(s) for s in run["samples"]]
        run["_sample_fingerprints"] = fps
        run["_run_fingerprint"] = _compute_run_fingerprint(fps)

    fp_to_runs: Dict[str, List[int]] = defaultdict(list)
    for i, run in enumerate(runs):
        fp_to_runs[run["_run_fingerprint"]].append(i)

    excluded_indices = set()
    excluded = []

    for fp, indices in fp_to_runs.items():
        if len(indices) <= 1:
            continue
        group_runs = [runs[i] for i in indices]
        retain_idx = indices[0]
        for i in indices:
            if runs[i]["has_meta"]:
                retain_idx = i
                break
        retain_path = os.path.normpath(runs[retain_idx]["path"])
        sorted_paths = sorted(
            (os.path.normpath(runs[i]["path"]), i) for i in indices
        )
        if not runs[retain_idx]["has_meta"]:
            retain_idx = sorted_paths[0][1]
            retain_path = sorted_paths[0][0]

        for _, idx in sorted_paths:
            if idx == retain_idx:
                continue
            excluded_indices.add(idx)
            excluded.append({
                "run": runs[idx],
                "reason": "exact_duplicate",
                "retained_by": retain_path,
            })

    warnings = []
    included_indices = [i for i in range(len(runs)) if i not in excluded_indices]
    for a_pos in range(len(included_indices)):
        for b_pos in range(a_pos + 1, len(included_indices)):
            ia = included_indices[a_pos]
            ib = included_indices[b_pos]
            ra, rb = runs[ia], runs[ib]
            na, nb = len(ra["samples"]), len(rb["samples"])
            if na == 0 or nb == 0:
                continue
            size_diff = abs(na - nb) / max(na, nb)
            if size_diff >= 0.05:
                continue
            check_n = min(50, na, nb)
            fps_a = set(ra["_sample_fingerprints"][:check_n])
            fps_b = set(rb["_sample_fingerprints"][:check_n])
            overlap = len(fps_a & fps_b) / max(len(fps_a), 1)
            if overlap > 0.8:
                warnings.append({
                    "run_a": ra["path"],
                    "run_b": rb["path"],
                    "reason": "suspected_duplicate",
                    "action": "keep",
                    "overlap": round(overlap, 3),
                    "size_diff": round(size_diff, 3),
                })

    included = [runs[i] for i in included_indices]
    return included, excluded, warnings


def namespace_samples(samples: List[dict], run_id: str,
                      source_seed: str, source_load_level: str) -> List[dict]:
    """Add source_* fields and namespace candidate ids.

    candidate_key is namespaced together with candidate_group_id so a fused
    dataset remains globally unambiguous across loads/seeds/runs.
    """
    for s in samples:
        s["source_run_id"] = run_id
        old_gid = s.get("candidate_group_id", "")
        s["source_candidate_group_id"] = old_gid
        s["candidate_group_id"] = f"{run_id}::{old_gid}"
        old_key = s.get("candidate_key")
        if old_key:
            s["source_candidate_key"] = old_key
            s["candidate_key"] = f"{run_id}::{old_key}"
        s["source_seed"] = source_seed
        s["source_load_level"] = source_load_level
    return samples


def _count_pairs(members: List[dict], epsilon: float = 0.01) -> int:
    count = 0
    costs = [m["realized_cost"] for m in members]
    for i in range(len(costs)):
        for j in range(i + 1, len(costs)):
            if abs(costs[i] - costs[j]) > epsilon:
                count += 1
    return count


def _group_by(samples: List[dict], key: str) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = defaultdict(list)
    for s in samples:
        groups[s.get(key, "unknown")].append(s)
    return dict(groups)


def _group_by_gid(samples: List[dict]) -> Dict[str, List[dict]]:
    return _group_by(samples, "candidate_group_id")


def _compute_group_baselines(samples: List[dict]):
    """Compute per-group mean of future_system_labels for B3 residual loss.

    Writes 'future_system_group_baseline' (K, 7) into each sample.
    """
    groups = _group_by_gid(samples)
    added = 0
    k_mismatch = 0
    for gid, members in groups.items():
        fsl_list = [s["future_system_labels"] for s in members
                    if "future_system_labels" in s]
        if not fsl_list:
            continue
        ks = [f.shape[0] for f in fsl_list]
        min_K = min(ks)
        if max(ks) != min_K:
            k_mismatch += 1
        truncated = [f[:min_K] for f in fsl_list]
        baseline = torch.stack(truncated).mean(dim=0)
        for s in members:
            s["future_system_group_baseline"] = baseline.clone()
            added += 1
    print(f"\n  B3 group baseline: added to {added}/{len(samples)} samples, "
          f"groups={len(groups)}, min_K_mismatch={k_mismatch}")


def _compute_split_stats(samples: List[dict]) -> dict:
    groups = _group_by_gid(samples)
    pairs = sum(_count_pairs(m) for m in groups.values())
    return {"samples": len(samples), "groups": len(groups), "pairs": pairs}


def _build_balance_report(
    samples: List[dict],
    train_gids: List[str],
    val_gids: List[str],
    test_gids: List[str],
) -> dict:
    split_sets = {
        "train": set(train_gids),
        "val": set(val_gids),
        "test": set(test_gids),
    }
    split_samples = {
        name: [
            sample for sample in samples
            if sample["candidate_group_id"] in gid_set
        ]
        for name, gid_set in split_sets.items()
    }
    report = {"by_run": {}, "by_seed": {}, "by_load": {}, "totals": {}}
    for dim_key, field in (
        ("by_run", "source_run_id"),
        ("by_seed", "source_seed"),
        ("by_load", "source_load_level"),
    ):
        values = set(sample.get(field, "unknown") for sample in samples)
        for value in sorted(values):
            report[dim_key][value] = {
                split_name: _compute_split_stats([
                    sample for sample in rows
                    if sample.get(field, "unknown") == value
                ])
                for split_name, rows in split_samples.items()
            }
    report["totals"] = {
        name: _compute_split_stats(rows)
        for name, rows in split_samples.items()
    }
    return report


def stratified_group_split(
    samples: List[dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[str], List[str], List[str], dict]:
    """Group-level split stratified by source_run_id.

    Returns (train_gids, val_gids, test_gids, balance_report).
    """
    gid_to_samples = _group_by_gid(samples)
    gid_to_run: Dict[str, str] = {}
    for gid, members in gid_to_samples.items():
        gid_to_run[gid] = members[0].get("source_run_id", "unknown")

    run_to_gids: Dict[str, List[str]] = defaultdict(list)
    for gid, rid in gid_to_run.items():
        run_to_gids[rid].append(gid)

    rng = random.Random(seed)
    train_gids, val_gids, test_gids = [], [], []

    for rid in sorted(run_to_gids.keys()):
        gids = sorted(run_to_gids[rid])
        rng.shuffle(gids)
        n = len(gids)
        n_train = max(1, int(n * train_ratio)) if n >= 3 else n
        n_val = max(1, int(n * val_ratio)) if n >= 3 else 0
        if n < 3:
            train_gids.extend(gids)
            continue
        train_gids.extend(gids[:n_train])
        val_gids.extend(gids[n_train:n_train + n_val])
        test_gids.extend(gids[n_train + n_val:])

    balance_report = _build_balance_report(
        samples, train_gids, val_gids, test_gids
    )

    return train_gids, val_gids, test_gids, balance_report


def explicit_seed_group_split(
    samples: List[dict],
    train_seeds: List[str],
    val_seeds: List[str],
    test_seeds: List[str],
) -> Tuple[List[str], List[str], List[str], dict]:
    """Assign whole simulation seeds to one split for Phase C isolation."""
    requested = {
        "train": {str(seed) for seed in train_seeds},
        "val": {str(seed) for seed in val_seeds},
        "test": {str(seed) for seed in test_seeds},
    }
    if any(not values for values in requested.values()):
        raise ValueError("explicit seed split requires non-empty train/val/test")
    if (
        requested["train"] & requested["val"]
        or requested["train"] & requested["test"]
        or requested["val"] & requested["test"]
    ):
        raise ValueError("explicit seed split lists must be disjoint")

    groups = _group_by_gid(samples)
    split_gids = {"train": [], "val": [], "test": []}
    observed = set()
    for gid, members in groups.items():
        member_seeds = {
            str(member.get("source_seed", "unknown")) for member in members
        }
        if len(member_seeds) != 1:
            raise ValueError(f"candidate group {gid} spans multiple seeds")
        seed = next(iter(member_seeds))
        observed.add(seed)
        destinations = [
            name for name, seeds in requested.items() if seed in seeds
        ]
        if len(destinations) != 1:
            raise ValueError(
                f"source seed {seed} is not assigned to exactly one split"
            )
        split_gids[destinations[0]].append(gid)

    expected = set().union(*requested.values())
    if observed != expected:
        raise ValueError(
            "explicit seed split does not exactly cover fused data: "
            f"observed={sorted(observed)}, expected={sorted(expected)}"
        )
    for values in split_gids.values():
        values.sort()
    report = _build_balance_report(
        samples,
        split_gids["train"],
        split_gids["val"],
        split_gids["test"],
    )
    report["split_unit"] = "source_seed"
    report["explicit_seeds"] = {
        name: sorted(values) for name, values in requested.items()
    }
    return (
        split_gids["train"],
        split_gids["val"],
        split_gids["test"],
        report,
    )


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fuse multiple .pt data files, deduplicate, and split",
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Paths to .pt data files")
    parser.add_argument("--long-risk-inputs", nargs="+", default=None,
                        help=("Optional per-input long-risk .pt files in the "
                              "same order as --inputs. Use 'none' or '-' to "
                              "skip a specific input. Labels are embedded "
                              "before namespacing."))
    parser.add_argument("--long-risk-filter-mode", type=str, default="none",
                        choices=["none", "informative_calibration"],
                        help=("Filter long-risk labels before embedding. "
                              "'informative_calibration' keeps all groups with "
                              "pairwise label margin and risk-stratified "
                              "calibration groups."))
    parser.add_argument("--long-risk-filter-dims", type=str,
                        default="terminal,cvar,peak,delta",
                        help=("Comma-separated dims for informative pair test: "
                              "terminal,cvar,peak,delta"))
    parser.add_argument("--long-risk-epsilon-mode", type=str, default="adaptive",
                        choices=["adaptive", "fixed"],
                        help="How to choose pairwise label margin epsilons")
    parser.add_argument("--long-risk-fixed-epsilon", type=float, default=0.02,
                        help="Fixed pairwise label margin if epsilon-mode=fixed")
    parser.add_argument("--long-risk-min-epsilon", type=float, default=0.01,
                        help="Lower bound for adaptive pairwise label margin")
    parser.add_argument("--long-risk-epsilon-quantile", type=float, default=0.25,
                        help="Quantile of nonzero pairwise diffs for adaptive epsilon")
    parser.add_argument("--long-risk-informative-min-pairs", type=int, default=1,
                        help="Min valid within-group pairs required for informative group")
    parser.add_argument("--long-risk-calibration-ratio", type=float, default=0.25,
                        help="Target calibration fraction among retained long-risk groups")
    parser.add_argument("--long-risk-risk-bins", type=str, default="0.35,0.70",
                        help="Risk score cut points for safe/medium/dangerous calibration bins")
    parser.add_argument("--long-risk-risk-weights", type=str, default="0.2,0.3,0.5",
                        help="Weights for group risk score: peak,cvar,terminal")
    parser.add_argument("--long-risk-filter-seed", type=int, default=42,
                        help="Seed for calibration group sampling")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--output-name", type=str, default="wm_train_data_fused.pt")
    parser.add_argument(
        "--require-lyapunov-l0-schema",
        type=str,
        default=None,
        help=(
            "Require every input to carry this Lyapunov collection schema. "
            "Use lyapunov_l1_collection_v3 for the current "
            "work+station+cumulative-arrival functional."
        ),
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument(
        "--split-unit",
        choices=("group", "source_seed"),
        default="group",
        help=(
            "group preserves the legacy within-run split; source_seed keeps "
            "each simulation seed wholly in train/val/test"
        ),
    )
    parser.add_argument(
        "--train-seeds",
        default="",
        help="Comma-separated source seeds for --split-unit source_seed",
    )
    parser.add_argument(
        "--val-seeds",
        default="",
        help="Comma-separated source seeds for --split-unit source_seed",
    )
    parser.add_argument(
        "--test-seeds",
        default="",
        help="Comma-separated source seeds for --split-unit source_seed",
    )
    args = parser.parse_args()

    if args.long_risk_inputs is not None and len(args.long_risk_inputs) != len(args.inputs):
        parser.error("--long-risk-inputs must have the same length as --inputs")
    if args.long_risk_filter_mode != "none" and args.long_risk_inputs is None:
        parser.error("--long-risk-filter-mode requires --long-risk-inputs")
    explicit_seed_lists = {
        "train": [value.strip() for value in args.train_seeds.split(",")
                  if value.strip()],
        "val": [value.strip() for value in args.val_seeds.split(",")
                if value.strip()],
        "test": [value.strip() for value in args.test_seeds.split(",")
                 if value.strip()],
    }
    if args.split_unit == "source_seed":
        if any(not values for values in explicit_seed_lists.values()):
            parser.error(
                "--split-unit source_seed requires --train-seeds, "
                "--val-seeds and --test-seeds"
            )
    elif any(explicit_seed_lists.values()):
        parser.error(
            "explicit seed lists require --split-unit source_seed"
        )
    if not 0.0 <= args.long_risk_calibration_ratio < 1.0:
        parser.error("--long-risk-calibration-ratio must be in [0, 1)")

    print("=" * 60)
    print("  Data Fusion & Stratified Split")
    print("=" * 60)

    long_risk_filter_config = None
    if args.long_risk_inputs is not None and args.long_risk_filter_mode != "none":
        lr_paths = [p for p in args.long_risk_inputs if not _is_skip_long_risk_path(p)]
        risk_bins = tuple(_parse_csv_floats(
            args.long_risk_risk_bins, 2, "--long-risk-risk-bins"))
        if risk_bins[0] >= risk_bins[1]:
            parser.error("--long-risk-risk-bins must be increasing")
        risk_weights_raw = _parse_csv_floats(
            args.long_risk_risk_weights, 3, "--long-risk-risk-weights")
        weight_sum = sum(risk_weights_raw)
        if weight_sum <= 0:
            parser.error("--long-risk-risk-weights must sum to a positive value")
        risk_weights = tuple(w / weight_sum for w in risk_weights_raw)
        epsilons = _compute_long_risk_epsilons(lr_paths, args)
        long_risk_filter_config = {
            "mode": args.long_risk_filter_mode,
            "epsilons": epsilons,
            "min_pairs": args.long_risk_informative_min_pairs,
            "calibration_ratio": args.long_risk_calibration_ratio,
            "risk_bins": risk_bins,
            "risk_weights": risk_weights,
            "seed": args.long_risk_filter_seed,
        }
        print("\n  Long-risk filter enabled:")
        print(f"    mode              : {args.long_risk_filter_mode}")
        print(f"    epsilons          : {epsilons}")
        print(f"    min_pairs         : {args.long_risk_informative_min_pairs}")
        print(f"    calibration_ratio : {args.long_risk_calibration_ratio}")
        print(f"    risk_bins         : {risk_bins}")
        print(f"    risk_weights      : {risk_weights}")

    # --- Step 1: Load & validate ---
    print(f"\n  Loading {len(args.inputs)} input file(s) ...")
    runs = []
    for idx, p in enumerate(args.inputs):
        print(f"    {p} ... ", end="", flush=True)
        run = load_and_validate(p)
        print(f"{len(run['samples'])} samples, run_id='{run['run_id']}'"
              f"{' (meta_missing)' if run['meta_missing'] else ''}")
        if args.long_risk_inputs is not None:
            lr_path = args.long_risk_inputs[idx]
            if not _is_skip_long_risk_path(lr_path):
                lr_stats = merge_long_risk_labels(
                    run["samples"],
                    lr_path,
                    filter_config=long_risk_filter_config,
                )
                run["long_risk_merge"] = lr_stats
                rate = lr_stats["matched_samples"] / max(lr_stats["total_samples"], 1)
                print(f"      long-risk: {lr_stats['matched_samples']}/"
                      f"{lr_stats['total_samples']} matched ({rate*100:.1f}%)")
                if lr_stats.get("filter"):
                    fs = lr_stats["filter"]
                    print(f"      filter: groups {fs['retained_groups']}/"
                          f"{fs['original_groups']} retained "
                          f"(info={fs['informative_groups']}, "
                          f"calib={fs['calibration_groups']})")
                if lr_stats["duplicate_keys"] > 0:
                    print(f"      WARNING: {lr_stats['duplicate_keys']} duplicate "
                          f"candidate_key entries in {lr_path}", file=sys.stderr)
            else:
                run["long_risk_merge"] = None
        runs.append(run)

    # A v1/v2 mix would silently reintroduce the historical all-zero traffic
    # target.  Reject it before deduplication or output creation.  Generic
    # non-Lyapunov datasets remain supported unless an explicit schema is
    # required by the caller.
    l0_schemas = {
        run["lyapunov_collection_schema"]
        for run in runs
        if run["lyapunov_collection_schema"] is not None
    }
    if len(l0_schemas) > 1:
        raise ValueError(
            "Refusing to fuse mixed Lyapunov collection schemas: "
            f"{sorted(l0_schemas)}. Recollect/fuse v2 in a separate directory."
        )
    if args.require_lyapunov_l0_schema is not None:
        mismatches = [
            (run["path"], run["lyapunov_collection_schema"])
            for run in runs
            if run["lyapunov_collection_schema"]
            != args.require_lyapunov_l0_schema
        ]
        if mismatches:
            details = ", ".join(
                f"{path}={schema!r}" for path, schema in mismatches
            )
            raise ValueError(
                "Lyapunov schema requirement failed: expected "
                f"{args.require_lyapunov_l0_schema!r}; {details}"
            )

    # --- Cross-file feature dimension consistency ---
    _feat_dim_keys = {
        "node_history": -1, "edge_features": -1, "demand_context": 0,
        "action_node": -1, "action_global": 0,
    }
    if len(runs) > 1:
        ref_sig = runs[0]["shape_signature"]
        ref_path = runs[0]["path"]
        for run in runs[1:]:
            for key, dim_idx in _feat_dim_keys.items():
                ref_shape = ref_sig.get(key)
                other_shape = run["shape_signature"].get(key)
                if ref_shape is None or other_shape is None:
                    continue
                if ref_shape[dim_idx] != other_shape[dim_idx]:
                    print(f"  WARNING: feature dim mismatch for '{key}': "
                          f"{ref_path} has {ref_shape}, "
                          f"{run['path']} has {other_shape}", file=sys.stderr)

    # --- Step 1b: Ensure unique run_ids ---
    runs = ensure_unique_run_ids(runs)

    # --- Step 2: Dedup (before namespace, using original group_id) ---
    print(f"\n  Detecting duplicates ...")
    included, excluded, warnings = detect_duplicates(runs)
    print(f"    included: {len(included)} files")
    print(f"    excluded (exact duplicate): {len(excluded)} files")
    print(f"    suspected duplicates: {len(warnings)} (kept)")
    for exc in excluded:
        print(f"      EXCLUDED: {exc['run']['path']}")
        print(f"        reason: {exc['reason']}, retained: {exc['retained_by']}")
    for w in warnings:
        print(f"      SUSPECTED: {w['run_a']} <-> {w['run_b']}"
              f" (overlap={w['overlap']}, size_diff={w['size_diff']})")

    # --- Step 3: Namespace included runs ---
    print(f"\n  Namespacing group_ids ...")
    all_samples = []
    per_run_info = {}
    for run in included:
        rid = run["run_id"]
        samples = namespace_samples(
            run["samples"], rid,
            run["source_seed"], run["source_load_level"],
        )
        groups = _group_by_gid(samples)
        pairs = sum(_count_pairs(m) for m in groups.values())
        per_run_info[rid] = {
            "path": run["path"],
            "samples": len(samples),
            "groups": len(groups),
            "pairs": pairs,
            "meta_missing": run["meta_missing"],
            "lyapunov_collection_schema": run[
                "lyapunov_collection_schema"
            ],
            "long_risk_merge": run.get("long_risk_merge"),
        }
        all_samples.extend(samples)
        print(f"    {rid}: {len(samples)} samples, {len(groups)} groups, {pairs} pairs")

    if not all_samples:
        print("\n  ERROR: No samples after dedup. Exiting.")
        sys.exit(1)

    # --- Step 3b: Compute B3 group baseline for tail-sensitive loss ---
    _compute_group_baselines(all_samples)

    # --- Step 4: Save fused data ---
    os.makedirs(args.output_dir, exist_ok=True)
    fused_path = os.path.join(args.output_dir, args.output_name)
    torch.save(all_samples, fused_path)
    print(f"\n  Saved fused data: {fused_path}")

    # --- Save manifest ---
    source_files_detail = []
    for r in included:
        source_files_detail.append({
            "path": r["path"],
            "sha256": _file_sha256(r["path"]),
            "run_id": r["run_id"],
            "samples": len(r["samples"]),
            "long_risk_merge": r.get("long_risk_merge"),
        })
    ref_keys = sorted(all_samples[0].keys()) if all_samples else []
    manifest = {
        "schema_version": "fuse_v1",
        "sample_schema": {
            "required_keys": sorted(REQUIRED_KEYS),
            "sample_keys_example": ref_keys,
            "label_schema_version": "wm_v4_local_pressure",
            "lyapunov_collection_schema": (
                next(iter(l0_schemas)) if l0_schemas else None
            ),
        },
        "long_risk_filter_config": long_risk_filter_config,
        "split_policy": {
            "unit": args.split_unit,
            "train_seeds": explicit_seed_lists["train"],
            "val_seeds": explicit_seed_lists["val"],
            "test_seeds": explicit_seed_lists["test"],
        },
        "source_files": source_files_detail,
        "excluded_duplicates": [
            {"path": e["run"]["path"], "reason": e["reason"],
             "retained_by": e["retained_by"]}
            for e in excluded
        ],
        "suspected_duplicates": warnings,
        "per_run": per_run_info,
    }
    manifest_path = os.path.join(args.output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    # --- Save duplicate report ---
    dup_report = {
        "exclude_duplicate": [
            {"path": e["run"]["path"], "reason": e["reason"],
             "retained_by": e["retained_by"]}
            for e in excluded
        ],
        "suspected_duplicate": warnings,
    }
    dup_path = os.path.join(args.output_dir, "duplicate_report.json")
    with open(dup_path, "w", encoding="utf-8") as f:
        json.dump(dup_report, f, indent=2, ensure_ascii=False)

    # --- Save quality report ---
    total_groups = _group_by_gid(all_samples)
    total_pairs = sum(_count_pairs(m) for m in total_groups.values())
    sizes = [len(m) for m in total_groups.values()]
    size_counter = dict(Counter(sizes))
    runs_with_meta_missing = [
        r["run_id"] for r in included if r["meta_missing"]
    ]
    quality = {
        "total_samples": len(all_samples),
        "total_long_risk_samples": sum(
            1 for s in all_samples if "long_risk_labels" in s
        ),
        "total_long_risk_groups": sum(
            1 for members in total_groups.values()
            if any("long_risk_labels" in s for s in members)
        ),
        "total_groups": len(total_groups),
        "total_pairs": total_pairs,
        "mean_group_size": round(sum(sizes) / max(len(sizes), 1), 2),
        "group_size_histogram": {str(k): v for k, v in sorted(size_counter.items())},
        "num_source_runs": len(included),
        "runs_with_meta_missing": runs_with_meta_missing,
        "meta_missing_count": len(runs_with_meta_missing),
    }
    quality_path = os.path.join(args.output_dir, "quality_report.json")
    with open(quality_path, "w", encoding="utf-8") as f:
        json.dump(quality, f, indent=2, ensure_ascii=False)

    # --- Step 5: Stratified split ---
    if args.split_unit == "source_seed":
        print(
            "\n  Splitting by whole source seed: "
            f"train={explicit_seed_lists['train']}, "
            f"val={explicit_seed_lists['val']}, "
            f"test={explicit_seed_lists['test']} ..."
        )
        train_gids, val_gids, test_gids, balance_report = (
            explicit_seed_group_split(
                all_samples,
                explicit_seed_lists["train"],
                explicit_seed_lists["val"],
                explicit_seed_lists["test"],
            )
        )
    else:
        print(f"\n  Splitting (train={args.train_ratio}, val={args.val_ratio}, "
              f"test={1 - args.train_ratio - args.val_ratio:.2f}, "
              f"seed={args.split_seed}) ...")
        train_gids, val_gids, test_gids, balance_report = (
            stratified_group_split(
                all_samples,
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
                seed=args.split_seed,
            )
        )

    splits = {
        "train": train_gids,
        "val": val_gids,
        "test": test_gids,
        "split_unit": args.split_unit,
        "explicit_seeds": explicit_seed_lists,
        "balance_report": balance_report,
    }
    splits_path = os.path.join(args.output_dir, "splits.json")
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2, ensure_ascii=False)

    # --- Step 6: Terminal summary ---
    t = balance_report["totals"]
    print(f"\n  {'=' * 40}")
    print(f"  {len(args.inputs)} candidate files")
    print(f"  included {len(included)} files")
    print(f"  excluded exact duplicates {len(excluded)} files")
    print(f"  suspected duplicates {len(warnings)} files")
    print(f"  fused samples {len(all_samples)}")
    print(f"  fused groups {len(total_groups)}")
    print(f"  fused pairs {total_pairs}")
    print(f"  split balance:")
    for split_name in ("train", "val", "test"):
        s = t[split_name]
        print(f"    {split_name:5s}: {s['samples']} samples, "
              f"{s['groups']} groups, {s['pairs']} pairs")
    print(f"  {'=' * 40}")
    print(f"\n  Output files:")
    print(f"    {fused_path}")
    print(f"    {splits_path}")
    print(f"    {manifest_path}")
    print(f"    {quality_path}")
    print(f"    {dup_path}")
    print(f"\n  Done.")


if __name__ == "__main__":
    main()
