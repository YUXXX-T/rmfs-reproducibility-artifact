"""Station-conditioned congestion targets and the J1 dispatch predictor.

This module is deliberately independent from the deployed assigner.  It
defines a state-level probe that was validated before being frozen for J1:

1. can the frozen node latent distinguish congestion at different stations;
2. can it distinguish traffic impairment from station service pressure.

``z`` is the node-latent matrix returned by :class:`STGNNEncoder`, not a
single global vector.  The shared head therefore indexes the station nodes
and returns one two-channel prediction per station.  No fixed station count
or global-pool fallback is allowed.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn as nn


SCHEMA_VERSION = "station_congestion_target_v1"
SCALE_SCHEMA_VERSION = "station_congestion_scale_contract_v1"
HEAD_SCHEMA_VERSION = "station_congestion_head_v1"

PRIMARY_REGION_HOPS = 3

TRAFFIC_COMPONENTS = (
    "stationary_ticks_max",
    "node_density_cvar90",
    "bottleneck_density_excess",
)
SERVICE_COMPONENTS = (
    "assigned_agent_capacity_ratio",
    "in_progress_pressure",
)
COMPONENT_NAMES = TRAFFIC_COMPONENTS + SERVICE_COMPONENTS
CHANNEL_NAMES = ("traffic", "service")


def canonical_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _finite_nonnegative(value: Any, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if number < -1e-8:
        raise ValueError(f"{name} must be non-negative, got {number}")
    return max(number, 0.0)


def extract_station_components(
    station_row: Mapping[str, Any],
    *,
    primary_region_hops: int = PRIMARY_REGION_HOPS,
) -> dict[str, float]:
    """Extract the frozen v1 physical components for one station and tick.

    ``bottleneck_density_excess`` removes the region-wide density baseline
    from bottleneck-weighted density.  This keeps the bottleneck component
    from simply counting the same density a second time.
    """

    regions = station_row.get("regions")
    if not isinstance(regions, Mapping):
        raise ValueError("station row lacks regions")
    region_key = f"h{int(primary_region_hops)}"
    region = regions.get(region_key)
    if not isinstance(region, Mapping):
        raise ValueError(f"station row lacks primary region {region_key}")

    density_mean = _finite_nonnegative(
        region.get("node_density_mean", 0.0),
        f"{region_key}.node_density_mean",
    )
    weighted_density = _finite_nonnegative(
        region.get("node_bottleneck_weighted_density", 0.0),
        f"{region_key}.node_bottleneck_weighted_density",
    )
    components = {
        "stationary_ticks_max": _finite_nonnegative(
            region.get("stationary_ticks_max", 0.0),
            f"{region_key}.stationary_ticks_max",
        ),
        "node_density_cvar90": _finite_nonnegative(
            region.get("node_density_cvar90", 0.0),
            f"{region_key}.node_density_cvar90",
        ),
        "bottleneck_density_excess": max(
            weighted_density - density_mean,
            0.0,
        ),
        "assigned_agent_capacity_ratio": _finite_nonnegative(
            station_row.get("assigned_agent_capacity_ratio", 0.0),
            "assigned_agent_capacity_ratio",
        ),
        "in_progress_pressure": _finite_nonnegative(
            station_row.get("in_progress_pressure", 0.0),
            "in_progress_pressure",
        ),
    }
    if tuple(components) != COMPONENT_NAMES:
        raise AssertionError("station congestion component order changed")
    return components


@dataclass(frozen=True)
class ComponentScale:
    lower: float
    upper: float
    constant: bool = False

    def normalise(self, value: float) -> float:
        value = float(value)
        if self.constant:
            return 0.0
        span = self.upper - self.lower
        if not math.isfinite(span) or span <= 0.0:
            raise ValueError("invalid component scale span")
        return min(max((value - self.lower) / span, 0.0), 1.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
            "constant": bool(self.constant),
        }


def fit_scale_contract(
    component_rows: Iterable[Mapping[str, float]],
    *,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    fitted_seeds: Sequence[int] = (),
    source_protocol_sha256: str | None = None,
) -> dict[str, Any]:
    """Fit a deterministic clipped-quantile scale on training seeds only."""

    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("quantiles must satisfy 0 <= lower < upper <= 1")
    values = {name: [] for name in COMPONENT_NAMES}
    row_count = 0
    for row in component_rows:
        row_count += 1
        for name in COMPONENT_NAMES:
            values[name].append(_finite_nonnegative(row[name], name))
    if row_count == 0:
        raise ValueError("cannot fit station congestion scales on zero rows")

    scales: dict[str, dict[str, Any]] = {}
    for name in COMPONENT_NAMES:
        tensor = torch.as_tensor(values[name], dtype=torch.float64)
        lower = float(torch.quantile(tensor, lower_quantile).item())
        upper = float(torch.quantile(tensor, upper_quantile).item())
        constant = not math.isfinite(upper - lower) or upper - lower <= 1e-12
        if constant:
            lower = float(tensor.min().item())
            upper = float(tensor.max().item())
            constant = upper - lower <= 1e-12
        scales[name] = ComponentScale(lower, upper, constant).to_dict()

    contract: dict[str, Any] = {
        "schema_version": SCALE_SCHEMA_VERSION,
        "target_schema_version": SCHEMA_VERSION,
        "primary_region_hops": PRIMARY_REGION_HOPS,
        "components": list(COMPONENT_NAMES),
        "traffic_components": list(TRAFFIC_COMPONENTS),
        "service_components": list(SERVICE_COMPONENTS),
        "channel_names": list(CHANNEL_NAMES),
        "aggregation": {
            "traffic": "equal_mean_after_component_normalisation",
            "service": "equal_mean_after_component_normalisation",
            "cross_channel_cancellation": False,
        },
        "normalisation": {
            "kind": "clipped_train_quantile_range",
            "lower_quantile": float(lower_quantile),
            "upper_quantile": float(upper_quantile),
        },
        "fit": {
            "rows": int(row_count),
            "seeds": sorted({int(value) for value in fitted_seeds}),
            "source_protocol_sha256": source_protocol_sha256,
        },
        "scales": scales,
    }
    contract["contract_sha256"] = canonical_sha256(contract)
    return contract


def verify_scale_contract(contract: Mapping[str, Any]) -> None:
    payload = dict(contract)
    claimed = str(payload.pop("contract_sha256", ""))
    if payload.get("schema_version") != SCALE_SCHEMA_VERSION:
        raise ValueError("wrong station congestion scale schema")
    if tuple(payload.get("components", ())) != COMPONENT_NAMES:
        raise ValueError("station congestion component schema mismatch")
    if tuple(payload.get("channel_names", ())) != CHANNEL_NAMES:
        raise ValueError("station congestion channel schema mismatch")
    if canonical_sha256(payload) != claimed:
        raise ValueError("station congestion scale contract hash mismatch")
    scales = payload.get("scales")
    if not isinstance(scales, Mapping):
        raise ValueError("station congestion scale contract lacks scales")
    for name in COMPONENT_NAMES:
        value = scales.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"station congestion scale missing {name}")
        ComponentScale(
            lower=float(value["lower"]),
            upper=float(value["upper"]),
            constant=bool(value.get("constant", False)),
        )


def normalise_components(
    components: Mapping[str, float],
    contract: Mapping[str, Any],
) -> dict[str, float]:
    verify_scale_contract(contract)
    result = {}
    scales = contract["scales"]
    for name in COMPONENT_NAMES:
        value = scales[name]
        scale = ComponentScale(
            lower=float(value["lower"]),
            upper=float(value["upper"]),
            constant=bool(value.get("constant", False)),
        )
        result[name] = scale.normalise(
            _finite_nonnegative(components[name], name)
        )
    return result


def build_station_targets(
    station_row: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, Any]:
    raw = extract_station_components(station_row)
    normalised = normalise_components(raw, contract)
    traffic = sum(normalised[name] for name in TRAFFIC_COMPONENTS) / len(
        TRAFFIC_COMPONENTS
    )
    service = sum(normalised[name] for name in SERVICE_COMPONENTS) / len(
        SERVICE_COMPONENTS
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "raw_components": raw,
        "normalised_components": normalised,
        "channels": {
            "traffic": float(traffic),
            "service": float(service),
        },
    }


class StationCongestionHead(nn.Module):
    """Shared affine-logistic probe over station node latents.

    The head has no fixed station dimension.  ``station_node_ids`` is a
    required station-selection contract; silently repeating a global latent
    would erase exactly the spatial distinction this probe is meant to test.
    """

    schema_version = HEAD_SCHEMA_VERSION
    channel_names = CHANNEL_NAMES

    def __init__(self, latent_dim: int = 64):
        super().__init__()
        if int(latent_dim) <= 0:
            raise ValueError("latent_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.proj = nn.Linear(self.latent_dim, len(CHANNEL_NAMES))

    @staticmethod
    def _normalise_station_ids(
        station_node_ids: Sequence[int] | torch.Tensor,
        *,
        num_nodes: int,
        device: torch.device,
    ) -> torch.Tensor:
        if station_node_ids is None:
            raise ValueError("station_node_ids is required")
        ids = torch.as_tensor(
            station_node_ids,
            dtype=torch.long,
            device=device,
        ).reshape(-1)
        if ids.numel() == 0:
            raise ValueError("station_node_ids cannot be empty")
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= int(num_nodes):
            raise ValueError("station_node_ids contains an invalid node index")
        if torch.unique(ids).numel() != ids.numel():
            raise ValueError("station_node_ids must be unique")
        return ids

    def forward(
        self,
        z: torch.Tensor,
        station_node_ids: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        if z.ndim != 2 or z.size(1) != self.latent_dim:
            raise ValueError(
                f"z must have shape (N, {self.latent_dim}), got {tuple(z.shape)}"
            )
        ids = self._normalise_station_ids(
            station_node_ids,
            num_nodes=z.size(0),
            device=z.device,
        )
        station_z = z.index_select(0, ids)
        return self.forward_station_latents(station_z)

    def forward_station_latents(self, station_z: torch.Tensor) -> torch.Tensor:
        """Evaluate already-selected station latents for offline probe training."""

        if station_z.ndim != 2 or station_z.size(1) != self.latent_dim:
            raise ValueError(
                "station_z must have shape "
                f"(S, {self.latent_dim}), got {tuple(station_z.shape)}"
            )
        return torch.sigmoid(self.proj(station_z))


__all__ = [
    "CHANNEL_NAMES",
    "COMPONENT_NAMES",
    "HEAD_SCHEMA_VERSION",
    "PRIMARY_REGION_HOPS",
    "SCALE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "SERVICE_COMPONENTS",
    "TRAFFIC_COMPONENTS",
    "ComponentScale",
    "StationCongestionHead",
    "build_station_targets",
    "canonical_sha256",
    "extract_station_components",
    "fit_scale_contract",
    "normalise_components",
    "verify_scale_contract",
]
