"""Batch launcher for the resumable six-station data-collection cells.

The public CLI of ``run_phase_c_station6_adaptation`` intentionally exposes
``collect_*`` and ``build_*`` as single ``--load/--seed`` operations.  The
campaign module already contains the audited process-pool implementation for
running all cells, but its ``_parallel_cells`` helper is not a public CLI
mode.  This adapter exposes only that batching operation; it does not alter
the frozen campaign runner or its protocol hash.  For repair snapshots it can
also resolve an already-trained Phase-C checkpoint from its lineage inside
each spawned worker, avoiding an unnecessary dependency on the large fused
core-training tensor.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import WorldModel.evaluation.run_phase_c_station6_adaptation as campaign


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (_repo_root() / path).resolve()


def _resolve_checkpoint_reference(root: Path, reference: str) -> Path:
    """Map a lineage checkpoint path onto the current checkout.

    Normally the lineage stores a repository-relative path.  If training was
    launched with an absolute path on another host, preserve the relative
    suffix below ``training/phasec_core_v1`` so the same artifact can be
    consumed after synchronization to this checkout.
    """

    candidate = Path(reference)
    if not candidate.is_absolute():
        return _resolve_repo_path(candidate)
    if candidate.is_file():
        return candidate.resolve()
    parts = candidate.parts
    marker = ("training", "phasec_core_v1")
    for index in range(len(parts) - len(marker) + 1):
        if tuple(parts[index : index + len(marker)]) == marker:
            return (root / Path(*parts[index:])).resolve()
    return candidate.resolve()


def _resolve_phasec_checkpoint(
    root: Path, lineage_path: str | Path | None
) -> tuple[Path, str]:
    """Resolve and audit the final Phase-C checkpoint from its lineage.

    This deliberately does not call ``_train_phasec_core``.  The normal
    campaign helper performs fusion before checking an existing checkpoint,
    which would unnecessarily require the large core training tensor on the
    CPU collection host.
    """

    bundle, _ = campaign._load_protocol(root)
    path = (
        Path(lineage_path)
        if lineage_path is not None
        else root / "phasec_core_training_lineage.json"
    )
    if not path.is_absolute():
        # A caller-provided lineage is normally relative to the repository
        # root (as are paths emitted by the campaign), not to the output root.
        path = _resolve_repo_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Phase-C core lineage not found: {path}")
    try:
        lineage = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid Phase-C core lineage JSON: {path}") from exc
    if not isinstance(lineage, dict):
        raise ValueError(f"Phase-C core lineage must be an object: {path}")
    if lineage.get("schema_version") != "station6_phase_c_core_training_lineage_v1":
        raise ValueError(
            "unexpected Phase-C core lineage schema: "
            f"{lineage.get('schema_version')!r}"
        )
    if lineage.get("protocol_sha256") != bundle["protocol_sha256"]:
        raise ValueError("Phase-C core lineage belongs to a different protocol")
    audit = lineage.get("audit")
    if not isinstance(audit, dict) or not bool(audit.get("passed")):
        raise ValueError("Phase-C core lineage audit did not pass")

    checkpoint_ref = lineage.get("checkpoint")
    expected_sha = lineage.get("checkpoint_sha256")
    if not isinstance(checkpoint_ref, str) or not checkpoint_ref:
        raise ValueError("Phase-C core lineage has no checkpoint path")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("Phase-C core lineage has no valid checkpoint SHA256")
    checkpoint = _resolve_checkpoint_reference(root, checkpoint_ref)
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Phase-C core checkpoint from lineage is missing: {checkpoint}"
        )
    actual_sha = campaign._sha256_file(checkpoint)
    if actual_sha != expected_sha:
        raise ValueError(
            "Phase-C core checkpoint SHA256 mismatch: "
            f"expected {expected_sha}, got {actual_sha}"
        )

    # Load only the metadata needed to prevent accidentally using a four-
    # station or CPU-incompatible checkpoint.  This is a small one-time audit;
    # no training tensor is loaded into the process pool.
    import torch

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Phase-C core checkpoint payload is not a dictionary")
    config = payload.get("model_config")
    if not isinstance(config, dict):
        raise ValueError("Phase-C core checkpoint has no model_config")
    if (
        int(config.get("demand_dim", -1)) != 11
        or int(config.get("num_stations", -1)) != campaign.STATION_COUNT
    ):
        raise ValueError(
            "Phase-C core checkpoint is not six-station/demand_dim=11: "
            f"{config}"
        )
    return checkpoint, actual_sha


def _run_direct_repair_snapshot_cell(
    root: Path,
    load: str,
    seed: int,
    ticks: int,
    checkpoint: str,
    checkpoint_sha256: str,
) -> None:
    """Run one repair snapshot cell with a pre-trained checkpoint.

    ``ProcessPoolExecutor`` uses ``spawn``.  Therefore the bypass must be
    installed inside each child, rather than monkey-patched only in the
    parent.  The original runner function, audits, simulator, and snapshot
    writer are still used unchanged; only checkpoint resolution is replaced.
    """

    checkpoint_path = Path(checkpoint)
    if campaign._sha256_file(checkpoint_path) != checkpoint_sha256:
        raise ValueError(
            f"checkpoint changed before repair cell started: {checkpoint_path}"
        )
    original_contract = campaign._snapshot_stage_contract

    def direct_contract(root_arg: Path, stage: str):
        if stage == "repair":
            return (
                campaign.REPAIR_DATA_SEEDS,
                checkpoint_path,
                "Station6PhaseCRepairBehavior",
                "station6_phase_c_repair",
            )
        return original_contract(root_arg, stage)

    # _run_snapshot_cell resolves _snapshot_stage_contract through the
    # module's global dictionary, so this child-local replacement is enough
    # and cannot affect the parent or another Slurm job.
    campaign._snapshot_stage_contract = direct_contract
    campaign._run_snapshot_cell(root, "repair", load, seed, ticks)


def _parallel_direct_repair_snapshots(
    root: Path,
    checkpoint: Path,
    checkpoint_sha256: str,
    workers: int,
    ticks: int,
) -> None:
    failures: list[tuple[Any, str]] = []
    cells = campaign._cells("collect_repair_snapshots")
    with ProcessPoolExecutor(
        max_workers=max(1, int(workers)),
        mp_context=mp.get_context("spawn"),
    ) as pool:
        futures = {
            pool.submit(
                _run_direct_repair_snapshot_cell,
                root,
                load,
                seed,
                ticks,
                checkpoint.as_posix(),
                checkpoint_sha256,
            ): (load, seed)
            for load, seed in cells
        }
        for future in as_completed(futures):
            cell = futures[future]
            try:
                future.result()
            except Exception as exc:  # preserve all failed cells in summary
                failures.append((cell, repr(exc)))
    if failures:
        raise RuntimeError(
            "direct repair snapshot failures: " f"{failures[:10]}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "collect_core_snapshots",
            "build_core_data",
            "collect_repair_snapshots",
            "label_longrisk",
            "build_repair_data",
        ),
        required=True,
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=30)
    parser.add_argument("--ticks", type=int, default=1500)
    parser.add_argument(
        "--direct-repair-checkpoint",
        action="store_true",
        help=(
            "for collect_repair_snapshots, resolve the trained checkpoint "
            "from lineage without fusing/training core data"
        ),
    )
    parser.add_argument(
        "--phasec-core-lineage",
        type=Path,
        help="optional Phase-C core lineage path for direct repair collection",
    )
    args = parser.parse_args()

    root = args.output_root.resolve()
    # Validate the frozen protocol before starting any child process.  This
    # keeps the same source/config/checkpoint audits as the main campaign.
    campaign._load_protocol(root)
    if args.mode in ("collect_core_snapshots", "collect_repair_snapshots"):
        campaign._require_protocol_ticks(
            campaign._load_protocol(root)[1], args.ticks
        )

    if args.direct_repair_checkpoint:
        if args.mode != "collect_repair_snapshots":
            parser.error(
                "--direct-repair-checkpoint requires "
                "--mode collect_repair_snapshots"
            )
        checkpoint, checkpoint_sha256 = _resolve_phasec_checkpoint(
            root, args.phasec_core_lineage
        )
        _parallel_direct_repair_snapshots(
            root,
            checkpoint,
            checkpoint_sha256,
            int(args.workers),
            int(args.ticks),
        )
    else:
        if args.phasec_core_lineage is not None:
            parser.error(
                "--phasec-core-lineage is only valid with "
                "--direct-repair-checkpoint"
            )
        campaign._parallel_cells(
            root,
            args.mode,
            int(args.workers),
            int(args.ticks),
        )
    print(
        f"[complete] station6 collection mode={args.mode} "
        f"output={root} workers={args.workers}"
    )


if __name__ == "__main__":
    main()
