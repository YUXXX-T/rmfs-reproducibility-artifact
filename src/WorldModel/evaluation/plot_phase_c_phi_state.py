"""Plot fresh 531--540 phi_state validation outputs.

The plots are diagnostic only.  They do not refit the head, alter the frozen
validation JSON, or select online weights.  Figures are written below a new
``figures`` directory next to the validation report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


CHANNELS = ("service", "traffic")
CHANNEL_INDEX = {"traffic": 0, "service": 1}
LOADS = ("low", "mid", "high")
ARMS = ("greedy", "hungarian", "phasec", "phasec_s1")
ARM_LABELS = {
    "greedy": "Greedy",
    "hungarian": "Hungarian",
    "phasec": "Phase-C",
    "phasec_s1": "Phase-C+S1",
}
COLORS = {"service": "#2166ac", "traffic": "#b2182b"}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_original_outputs(root: Path) -> None:
    manifest = root / "validation_outputs.sha256"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    missing = []
    changed = []
    for raw in manifest.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        expected, relative = raw.split(None, 1)
        path = root / relative.strip()
        if not path.is_file():
            missing.append(path.name)
        elif _sha256(path) != expected:
            changed.append(path.name)
    if missing or changed:
        raise RuntimeError(
            f"validation artifact audit failed: missing={missing}, changed={changed}"
        )


def _rankdata(values: Sequence[float]) -> np.ndarray:
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


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size < 3 or left.size != right.size:
        return float("nan")
    if np.all(left == left[0]) or np.all(right == right[0]):
        return float("nan")
    left = _rankdata(left)
    right = _rankdata(right)
    left -= left.mean()
    right -= right.mean()
    denominator = math.sqrt(float(left @ left) * float(right @ right))
    return float(left @ right / denominator) if denominator > 0 else float("nan")


def _groups(keys: Iterable[Any]) -> dict[Any, list[int]]:
    result: dict[Any, list[int]] = defaultdict(list)
    for index, key in enumerate(keys):
        result[key].append(index)
    return result


def _load(root: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    report = _read_json(root / "phi_state_validation.json")
    data = np.load(root / "phi_state_predictions.npz", allow_pickle=False)
    required = {"prediction", "target", "run_id", "load", "arm", "seed", "tick"}
    if set(data.files) != required:
        raise ValueError(f"unexpected npz keys: {data.files}")
    arrays = {key: data[key] for key in data.files}
    count = int(arrays["prediction"].shape[0])
    if arrays["prediction"].shape != arrays["target"].shape or arrays["prediction"].shape[1] != 2:
        raise ValueError("prediction/target shape mismatch")
    for key in ("run_id", "load", "arm", "seed", "tick"):
        if len(arrays[key]) != count:
            raise ValueError(f"metadata length mismatch for {key}")
    return report, arrays


def _run_records(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    run_groups = _groups(arrays["run_id"].tolist())
    prediction = arrays["prediction"]
    target = arrays["target"]
    records = []
    for run_id, indices in sorted(run_groups.items(), key=lambda item: str(item[0])):
        index = np.asarray(indices, dtype=np.int64)
        arm = str(arrays["arm"][index[0]])
        load = str(arrays["load"][index[0]])
        seed = int(arrays["seed"][index[0]])
        row = {"run_id": str(run_id), "arm": arm, "load": load, "seed": seed}
        for channel, channel_index in CHANNEL_INDEX.items():
            pred = prediction[index, channel_index]
            true = target[index, channel_index]
            row[f"{channel}_rho"] = _spearman(pred, true)
            row[f"{channel}_mae"] = float(np.mean(np.abs(pred - true)))
        records.append(row)
    return records


def _within_tick_records(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    keys = list(zip(arrays["run_id"].tolist(), arrays["tick"].tolist()))
    grouped = _groups(keys)
    prediction = arrays["prediction"]
    target = arrays["target"]
    records = []
    for (run_id, tick), indices in grouped.items():
        index = np.asarray(indices, dtype=np.int64)
        row = {
            "run_id": str(run_id),
            "tick": int(tick),
            "load": str(arrays["load"][index[0]]),
            "arm": str(arrays["arm"][index[0]]),
        }
        for channel, channel_index in CHANNEL_INDEX.items():
            pred = prediction[index, channel_index]
            true = target[index, channel_index]
            row[f"{channel}_rho"] = _spearman(pred, true)
            pred_best = int(np.argmax(pred))
            row[f"{channel}_top1"] = bool(true[pred_best] >= np.max(true) - 1e-12)
        records.append(row)
    return records


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _figure_01_scatter(root: Path, arrays: Mapping[str, np.ndarray], report: Mapping[str, Any]) -> None:
    prediction = arrays["prediction"]
    target = arrays["target"]
    rng = np.random.default_rng(20260807)
    sample_size = min(60000, len(prediction))
    sample = rng.choice(len(prediction), size=sample_size, replace=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, channel, index in zip(axes, ("service", "traffic"), (1, 0)):
        x = target[sample, index]
        y = prediction[sample, index]
        hb = axis.hexbin(x, y, gridsize=55, mincnt=1, bins="log", cmap="viridis")
        axis.plot([0, 1], [0, 1], "k--", linewidth=1, label="identity")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        axis.set_xlabel("same-tick target")
        axis.set_ylabel("phi_state prediction")
        metrics = report["channels"][channel]
        axis.set_title(
            f"{channel}  rho={metrics['pooled_spearman']:.3f}\n"
            f"MAE={metrics['mae']:.3f}, pred/target std="
            f"{metrics['prediction_std'] / metrics['target_std']:.2f}"
        )
        fig.colorbar(hb, ax=axis, label="log10 count")
    _save(fig, root / "figure_01_true_vs_predicted.png")


def _figure_02_run_heatmaps(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharex=True, sharey=True)
    for row_index, channel in enumerate(CHANNELS):
        value_key = f"{channel}_rho"
        for column, load in enumerate(LOADS):
            matrix = np.full((10, len(ARMS)), np.nan, dtype=np.float64)
            for record in records:
                if record["load"] != load:
                    continue
                matrix[int(record["seed"]) - 531, ARMS.index(record["arm"])] = record[value_key]
            axis = axes[row_index, column]
            sns.heatmap(
                matrix,
                ax=axis,
                vmin=0.6,
                vmax=1.0,
                cmap="YlGnBu",
                annot=True,
                fmt=".2f",
                cbar=column == len(LOADS) - 1,
                cbar_kws={"label": "run Spearman"},
                xticklabels=[ARM_LABELS[a] for a in ARMS],
                yticklabels=list(range(531, 541)) if column == 0 else False,
            )
            axis.set_title(f"{channel} / {load}")
            axis.set_xlabel("arm")
            if column == 0:
                axis.set_ylabel("seed")
    _save(fig, root / "figure_02_run_rho_heatmaps.png")


def _figure_03_within_tick(root: Path, within: Sequence[Mapping[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for axis, channel in zip(axes, CHANNELS):
        values = []
        labels = []
        for load in LOADS:
            for arm in ARMS:
                group = [
                    row[f"{channel}_rho"] for row in within
                    if row["load"] == load and row["arm"] == arm
                    and math.isfinite(float(row[f"{channel}_rho"]))
                ]
                values.append(group)
                labels.append(f"{load}\n{ARM_LABELS[arm]}")
        axis.boxplot(values, patch_artist=True, showfliers=False,
                     boxprops={"facecolor": COLORS[channel], "alpha": 0.35},
                     medianprops={"color": "black"})
        axis.axhline(0.6, color="black", linestyle="--", linewidth=1,
                     label="0.60 service reference")
        axis.set_xticks(range(1, len(labels) + 1), labels, rotation=45, ha="right")
        axis.set_ylim(-0.05, 1.05)
        axis.set_ylabel("within-tick station Spearman")
        axis.set_title(f"{channel}: station ranking distribution")
    _save(fig, root / "figure_03_within_tick_station_ranking.png")


def _rolling(values: np.ndarray, window: int = 10) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(values, kernel, mode="same")


def _figure_04_temporal(root: Path, arrays: Mapping[str, np.ndarray], records: Sequence[Mapping[str, Any]]) -> None:
    # Include one difficult seed and one representative strong seed.  The
    # choice is fixed from the frozen report, not tuned on plotted curves.
    candidates = ["phasec_mid_seed539", "phasec_s1_mid_seed537"]
    available = set(str(value) for value in arrays["run_id"].tolist())
    selected = [value for value in candidates if value in available]
    if len(selected) < 2:
        selected = [str(records[0]["run_id"]), str(records[-1]["run_id"])]
    fig, axes = plt.subplots(len(selected), 2, figsize=(13, 4.5 * len(selected)), squeeze=False)
    for row_index, run_id in enumerate(selected):
        mask = arrays["run_id"] == run_id
        ticks = arrays["tick"][mask]
        order = np.argsort(ticks)
        ticks = ticks[order]
        for column, channel in enumerate(("service", "traffic")):
            channel_index = CHANNEL_INDEX[channel]
            pred = arrays["prediction"][mask, channel_index][order]
            true = arrays["target"][mask, channel_index][order]
            unique_ticks = np.unique(ticks)
            pred_mean = np.asarray([pred[ticks == tick].mean() for tick in unique_ticks])
            true_mean = np.asarray([true[ticks == tick].mean() for tick in unique_ticks])
            axis = axes[row_index, column]
            axis.plot(unique_ticks, true_mean, color="black", alpha=0.25, linewidth=0.8)
            axis.plot(unique_ticks, pred_mean, color=COLORS[channel], alpha=0.25, linewidth=0.8)
            axis.plot(unique_ticks, _rolling(true_mean), color="black", linewidth=2, label="target")
            axis.plot(unique_ticks, _rolling(pred_mean), color=COLORS[channel], linewidth=2, label="prediction")
            axis.set_ylim(0, 1)
            axis.set_xlabel("tick")
            axis.set_ylabel("station mean pressure")
            axis.set_title(f"{run_id} / {channel}")
            axis.legend(loc="upper left")
    _save(fig, root / "figure_04_temporal_pressure_curves.png")


def _figure_05_seed_summary(root: Path, records: Sequence[Mapping[str, Any]]) -> None:
    seed_values = {channel: [] for channel in CHANNELS}
    for seed in range(531, 541):
        rows = [record for record in records if int(record["seed"]) == seed]
        seed_values["service"].append(float(np.mean([r["service_rho"] for r in rows])))
        seed_values["traffic"].append(float(np.mean([r["traffic_rho"] for r in rows])))
    x = np.arange(10)
    width = 0.36
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(x - width / 2, seed_values["service"], width, label="service", color=COLORS["service"])
    axis.bar(x + width / 2, seed_values["traffic"], width, label="traffic", color=COLORS["traffic"])
    axis.axhline(0.8, color="black", linestyle="--", linewidth=1, label="0.80 reference")
    axis.set_xticks(x, [str(seed) for seed in range(531, 541)])
    axis.set_ylim(0.5, 1.0)
    axis.set_xlabel("seed")
    axis.set_ylabel("mean run Spearman across load/arm")
    axis.set_title("Seed difficulty summary (diagnostic, not a fitted threshold)")
    axis.legend()
    _save(fig, root / "figure_05_seed_difficulty.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(
            "WorldModel/checkpoints/phaseC_wm_onpolicy_round1_v1/"
            "phi_state_validate_531_540_v2/validation"
        ),
    )
    parser.add_argument("--sample-max", type=int, default=60000)
    args = parser.parse_args()
    root = args.input_root
    _verify_original_outputs(root)
    report, arrays = _load(root)
    records = _run_records(arrays)
    within = _within_tick_records(arrays)
    figures = root / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    _figure_01_scatter(figures, arrays, report)
    _figure_02_run_heatmaps(figures, records)
    _figure_03_within_tick(figures, within)
    _figure_04_temporal(figures, arrays, records)
    _figure_05_seed_summary(figures, records)
    summary = {
        "schema_version": "phase_c_phi_state_plot_summary_v1",
        "source_validation_sha256": _sha256(root / "phi_state_validation.json"),
        "source_predictions_sha256": _sha256(root / "phi_state_predictions.npz"),
        "figures": [path.name for path in sorted(figures.glob("figure_*.png"))],
        "runs": len(records),
        "within_tick_groups": len(within),
        "sample_rows": int(len(arrays["prediction"])),
    }
    (figures / "plot_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    manifest = figures / "figures.sha256"
    manifest.write_text(
        "\n".join(
            f"{_sha256(path)}  {path.name}"
            for path in sorted(figures.iterdir())
            if path.is_file() and path.name != manifest.name
        ) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"[complete] wrote figures to {figures}")


if __name__ == "__main__":
    main()
