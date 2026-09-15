"""Run six-station campaign training stages on CUDA without changing the
frozen campaign runner.

The six-station protocol records the SHA-256 of
``run_phase_c_station6_adaptation.py``.  Changing that runner merely to
replace its CPU device flag would invalidate an already-collected protocol.
This small wrapper therefore leaves the runner untouched and intercepts only
the child training commands, changing the value following ``--device`` to a
user-selected CUDA device.  All normal checkpoint, lineage, and tensor audits
still execute in the original runner.

It intentionally does not collect simulator data or run online evaluation.
Those stages stay on the CPU server and are launched by
``run_phase_c_station6_data_collection_cpu.slurm``.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence


def _gpu_run_factory(campaign: Any, device: str):
    original_run = campaign._run

    def gpu_run(
        command: Sequence[str],
        *,
        cwd: Path,
        log: Path | None = None,
        env_overrides: Mapping[str, str] | None = None,
    ) -> None:
        rewritten = list(command)
        for index, value in enumerate(rewritten[:-1]):
            if value == "--device":
                rewritten[index + 1] = device
        original_run(
            rewritten,
            cwd=cwd,
            log=log,
            env_overrides=env_overrides,
        )

    return gpu_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("base", "core", "longrisk", "station_head"),
        required=True,
        help="one training stage; data collection remains CPU-only",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
            "phasec_station6_adaptation_711_740_v1"
        ),
    )
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available on this node")
    if not (args.device == "cuda" or args.device.startswith("cuda:")):
        raise ValueError(f"GPU device must be cuda or cuda:N, got {args.device!r}")

    import WorldModel.evaluation.run_phase_c_station6_adaptation as campaign

    # Keep the frozen campaign source and its protocol hash unchanged.
    campaign._run = _gpu_run_factory(campaign, args.device)
    root = args.output_root
    if args.stage == "base":
        output = campaign._train_base(root)
    elif args.stage == "core":
        output = campaign._train_phasec_core(root)
    elif args.stage == "longrisk":
        output = campaign._train_longrisk(root)
    else:
        output = campaign._train_station_head(root)
    print(f"[complete] GPU training stage={args.stage}: {output}")


if __name__ == "__main__":
    main()
