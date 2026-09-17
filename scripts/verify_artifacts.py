#!/usr/bin/env python3
"""Verify evidence counts, protocol invariants, file indexes, and anonymity."""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_counts() -> None:
    main = read_rows(ROOT / "artifacts/raw/main_per_seed.csv")
    six = read_rows(ROOT / "artifacts/raw/station6_per_seed.csv")
    scale = read_rows(ROOT / "artifacts/raw/density_scale_per_seed.csv")
    events = read_rows(ROOT / "artifacts/raw/station_lock_events.csv")
    if len(main) != 750:
        raise AssertionError(f"main rows: expected 750, got {len(main)}")
    if {(r["load"], int(r["seed"]), r["arm"]) for r in main} != {
        (load, seed, arm)
        for load in ("low", "mid", "high")
        for seed in range(900, 950)
        for arm in ("Greedy", "Hungarian", "JSQ", "PhaseC", "ComboS1J1")
    }:
        raise AssertionError("main load/seed/arm grid is incomplete")
    if len(six) != 120:
        raise AssertionError(f"six-station rows: expected 120, got {len(six)}")
    if {(r["load"], int(r["seed"]), r["arm"]) for r in six} != {
        (load, seed, arm)
        for load in ("low", "mid", "high")
        for seed in range(721, 731)
        for arm in ("greedy", "hungarian", "phasec", "combo_s1_j1")
    }:
        raise AssertionError("six-station load/seed/arm grid is incomplete")
    if len(scale) != 1350:
        raise AssertionError(f"density-scale rows: expected 1350, got {len(scale)}")
    if {
        (
            int(r["map_rows"]),
            round(float(r["robot_density"]), 2),
            r["load"],
            int(r["seed"]),
            r["arm"],
        )
        for r in scale
    } != {
        (size, density, load, seed, arm)
        for size in (20, 30, 40)
        for density in (0.12, 0.15, 0.18)
        for load in ("low", "mid", "high")
        for seed in range(701, 711)
        for arm in ("greedy", "hungarian", "jsq", "phasec", "combo_s1_j1")
    }:
        raise AssertionError("density-scale factorial grid is incomplete")
    paired_arrivals = {}
    for row in scale:
        key = (row["load"], int(row["seed"]))
        paired_arrivals.setdefault(key, set()).add(int(row["order_arrival_count"]))
    if len(paired_arrivals) != 30 or any(len(values) != 1 for values in paired_arrivals.values()):
        raise AssertionError("density-scale arrival counts are not paired")
    for row in scale:
        size = int(row["map_rows"])
        robots = int(row["num_robots"])
        if int(row["map_cols"]) != size or int(row["num_stations"]) != 4:
            raise AssertionError("density-scale maps are not square four-station maps")
        if robots != round(size * size * float(row["robot_density"])):
            raise AssertionError("density-scale robot count is inconsistent with density")
        if row["variant"] != f"map{size}_r{robots}_s4":
            raise AssertionError("density-scale variant label is inconsistent")
    if len(events) != 100:
        raise AssertionError(f"station event rows: expected 100, got {len(events)}")


def verify_protocols() -> None:
    with (ROOT / "configs/evaluation/main_50seed.yaml").open(encoding="utf-8") as handle:
        main = yaml.safe_load(handle)
    if main["seeds"] != {"start": 900, "stop_exclusive": 950, "count": 50}:
        raise AssertionError("main seed contract changed")
    if main["bootstrap"]["iterations"] != 10_000 or main["bootstrap"]["seed"] != 20260905:
        raise AssertionError("bootstrap contract changed")
    with (ROOT / "configs/detection/collapse.yaml").open(encoding="utf-8") as handle:
        collapse = yaml.safe_load(handle)
    label = collapse["run_level_label"]
    if label["completed_orders"]["threshold"] != 300 or label["deadlock_ratio_mean"]["threshold"] != 0.4:
        raise AssertionError("collapse endpoint changed")
    if not label["conjunction"]:
        raise AssertionError("collapse endpoint must use conjunction")
    with (ROOT / "configs/evaluation/density_scale_10seed.yaml").open(encoding="utf-8") as handle:
        scale = yaml.safe_load(handle)
    if scale["expected_runs"] != 1350 or scale["stations"] != 4:
        raise AssertionError("density-scale protocol size or station count changed")
    if scale["analysis"]["bootstrap"] != {
        "type": "stratified_percentile",
        "strata": ["low", "mid", "high"],
        "iterations": 50_000,
        "seed": 20260916,
        "interval": 0.95,
    }:
        raise AssertionError("density-scale bootstrap contract changed")


def verify_manifest_metadata() -> None:
    rows = read_rows(ROOT / "manifests/metadata.csv")
    scale_rows = read_rows(ROOT / "artifacts/raw/density_scale_per_seed.csv")
    scale_counts = {
        (row["load"], int(row["seed"])): int(row["order_arrival_count"])
        for row in scale_rows
    }
    if len(rows) != 210:
        raise AssertionError(f"manifest metadata rows: expected 210, got {len(rows)}")
    if {row["campaign"] for row in rows} != {
        "main_50seed", "station6_10seed", "density_scale_10seed"
    }:
        raise AssertionError("arrival-manifest campaign set changed")
    for row in rows:
        target = ROOT / "manifests" / row["relative_path"]
        if not target.is_file():
            raise AssertionError(f"missing arrival manifest: {target}")
        payload = json.loads(target.read_text(encoding="utf-8"))
        if row["schema_version"] != payload["schema_version"]:
            raise AssertionError(f"manifest schema mismatch: {target}")
        if int(row["total_orders"]) != payload["total_orders"] or payload["total_orders"] != len(payload["orders"]):
            raise AssertionError(f"manifest order count mismatch: {target}")
        if row["campaign"] == "density_scale_10seed":
            match = re.fullmatch(r"orders_(low|mid|high)_seed(\d+)\.json", target.name)
            if match is None or scale_counts[(match.group(1), int(match.group(2)))] != payload["total_orders"]:
                raise AssertionError(f"density-scale result/manifest mismatch: {target}")


def verify_runtime_benchmark() -> None:
    figure_input = ROOT / "artifacts/raw/figure_inputs"
    labels = {"greedy", "jsq", "hungarian", "proposed_cpu", "proposed_cuda_0"}
    all_rows = []
    by_load = {}
    for load in ("low", "mid", "high"):
        rows = read_rows(figure_input / f"{load}_all_seeds_cpu_gpu_runs.csv")
        if len(rows) != 50:
            raise AssertionError(f"runtime {load} rows: expected 50, got {len(rows)}")
        if {(row["label"], int(row["seed"])) for row in rows} != {
            (label, seed) for label in labels for seed in range(721, 731)
        }:
            raise AssertionError(f"runtime {load} method/seed grid is incomplete")
        for row in rows:
            if row["load"] != load or int(row["ticks"]) != 1500:
                raise AssertionError(f"runtime {load} load/tick contract changed")
            if int(row["assignment_calls"]) != 1500:
                raise AssertionError(f"runtime {load} assignment-call count changed")
            if row["audit_passed"].lower() != "true":
                raise AssertionError(f"runtime {load} contains a failed run audit")
        all_rows.extend(rows)
        by_load[load] = rows
    if len(all_rows) != 150:
        raise AssertionError("runtime benchmark must contain 150 runs")

    summary = read_rows(figure_input / "station6_runtime_chart_summary.csv")
    if len(summary) != 15 or {(row["load"], row["label"]) for row in summary} != {
        (load, label) for load in ("low", "mid", "high") for label in labels
    }:
        raise AssertionError("runtime 3x5 summary is incomplete")
    if any(
        int(row["runs"]) != 10
        or int(row["assignment_calls"]) != 15_000
        or row["audit_passed"].lower() != "true"
        for row in summary
    ):
        raise AssertionError("runtime summary count/audit contract changed")

    mismatches = {}
    for load, rows in by_load.items():
        completed = {
            (row["label"], int(row["seed"])): int(float(row["completed_orders"]))
            for row in rows
        }
        mismatches[load] = {
            seed
            for seed in range(721, 731)
            if completed[("proposed_cpu", seed)]
            != completed[("proposed_cuda_0", seed)]
        }
    if mismatches != {"low": set(), "mid": set(), "high": {727, 728, 730}}:
        raise AssertionError("runtime CPU/GPU trajectory-mismatch audit changed")


def verify_artifact_index() -> int:
    root = ROOT / "artifacts"
    index = json.loads((root / "artifact_manifest.json").read_text(encoding="utf-8"))
    if index.get("schema_version") != "rmfs_artifact_index_v2":
        raise AssertionError("artifact index schema mismatch")
    entries = index["entries"]
    if index["entry_count"] != len(entries):
        raise AssertionError("artifact index entry count mismatch")
    expected = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file()
        and "generated" not in path.parts
        and path.name != "artifact_manifest.json"
    }
    indexed = {entry["path"]: entry for entry in entries}
    if set(indexed) != set(expected):
        raise AssertionError("artifact index paths do not match committed evidence")
    for relative in expected:
        expected_role = relative.split("/", 1)[0]
        if indexed[relative].get("role") != expected_role:
            raise AssertionError(f"artifact role mismatch: {relative}")
    return len(entries)


def anonymity_scan() -> None:
    # Split sensitive tokens so the scanner does not flag its own source.
    forbidden = [
        "Tian" + "Yuxuan",
        "yuxuan" + "_tian",
        "acct-" + "wanglin",
        "MAS_" + "RMFS_wm",
    ]
    text_suffixes = {".py", ".md", ".json", ".yaml", ".yml", ".toml", ".txt", ".csv", ".cff", ".sh"}
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts or "site" in path.parts:
            continue
        if path.suffix.lower() in text_suffixes or path.name in {"Makefile", "Dockerfile"}:
            content = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".pdf":
            raw = path.read_bytes().decode("latin-1", errors="ignore")
            # PDF page streams are compressed binary and can accidentally
            # contain byte sequences that resemble drive paths. Metadata and
            # embedded producer strings remain in printable ASCII runs.
            content = "\n".join(re.findall(r"[\x20-\x7e]{8,}", raw))
        else:
            continue
        for token in forbidden:
            if token.lower() in content.lower():
                offenders.append(f"{path.relative_to(ROOT)} contains forbidden identity token")
        cluster_home_pattern = r"/" + "lustre/home/" + r"[^/\s]+/"
        if re.search(cluster_home_pattern, content, flags=re.IGNORECASE):
            offenders.append(f"{path.relative_to(ROOT)} contains a cluster home path")
        workstation_pattern = r"[A-Za-z]:[/\\](?:" + "Users|note|MAS_" + r")[/\\]"
        if re.search(workstation_pattern, content, flags=re.IGNORECASE):
            offenders.append(f"{path.relative_to(ROOT)} contains an absolute workstation path")
        drive_path_pattern = r"(?<![A-Za-z0-9_])[A-Za-z]:[/\\][A-Za-z0-9_.-]"
        if re.search(drive_path_pattern, content):
            offenders.append(f"{path.relative_to(ROOT)} contains an absolute drive path")
        personal_email_pattern = (
            r"[A-Za-z0-9._%+-]+@"
            + r"(?:gmail|outlook|hotmail|qq|163|126|protonmail)\."
            + r"[A-Za-z]{2,}"
        )
        if re.search(personal_email_pattern, content, flags=re.IGNORECASE):
            offenders.append(f"{path.relative_to(ROOT)} contains a personal e-mail address")
        public_source_pattern = r"https?://" + r"github\.com/[A-Za-z0-9_.-]+/"
        if re.search(public_source_pattern, content, flags=re.IGNORECASE):
            offenders.append(f"{path.relative_to(ROOT)} contains a public GitHub owner URL")
        if re.search(r"(?<![0-9A-Fa-f])[0-9A-Fa-f]{64}(?![0-9A-Fa-f])", content):
            offenders.append(f"{path.relative_to(ROOT)} contains a static 64-hex fingerprint")
    if offenders:
        raise AssertionError("anonymity scan failed:\n" + "\n".join(sorted(set(offenders))))


def git_history_anonymity_scan() -> None:
    if not (ROOT / ".git").is_dir():
        return
    result = subprocess.run(
        ["git", "log", "--all", "--format=%an|%ae|%cn|%ce"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    allowed_name = "Anonymous Authors"
    allowed_email = "anonymous@invalid.example"
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if len(fields) != 4:
            raise AssertionError("unexpected Git identity record")
        author_name, author_email, committer_name, committer_email = fields
        if (author_name, committer_name) != (allowed_name, allowed_name):
            raise AssertionError("Git history contains a non-anonymous name")
        if (author_email, committer_email) != (allowed_email, allowed_email):
            raise AssertionError("Git history contains a non-anonymous e-mail address")


def main() -> None:
    verify_counts()
    verify_protocols()
    verify_manifest_metadata()
    verify_runtime_benchmark()
    indexed = verify_artifact_index()
    anonymity_scan()
    git_history_anonymity_scan()
    required = [
        ROOT / "src/Engine/simulation_engine.py",
        ROOT / "src/WorldModel/graph/graph_builder.py",
        ROOT / "src/Policies/TaskAssigner/JSQTaskAssigner/jsq_task_assigner.py",
        ROOT / "artifacts/statistics/paired_confidence_intervals.csv",
        ROOT / "artifacts/figures/results_overview.png",
        ROOT / "artifacts/figures/paired_effects_overview.png",
        ROOT / "artifacts/figures/fig05_station_lock_mechanism.pdf",
        ROOT / "artifacts/figures/fig06_runtime_benchmark.pdf",
        ROOT / "artifacts/figures/fig06s_runtime_wall_time.pdf",
        ROOT / "artifacts/figures/station6_runtime_assignment.pdf",
        ROOT / "artifacts/figures/station6_runtime_wall.pdf",
        ROOT / "artifacts/raw/figure_inputs/fig06_runtime_benchmark_runs.csv",
        ROOT / "artifacts/raw/figure_inputs/station6_runtime_chart_summary.csv",
        ROOT / "artifacts/figures/density_scale_completed_orders_factorial.png",
        ROOT / "artifacts/figures/density_scale_four_endpoint_summary.png",
        ROOT / "artifacts/statistics/density_scale_paired_effects.csv",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise AssertionError(f"missing required artifacts: {missing}")
    print(
        "PASS: main, six-station, 1,350-run scale, and 150-run runtime counts; "
        f"protocols; 210 arrival manifests; {indexed} indexed artifacts; "
        "and anonymity verified"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(1)
