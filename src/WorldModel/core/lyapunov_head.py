"""Neural physical-component head with a fixed analytic L0 functional.

The network is allowed to predict auditable physical quantities only.  The
Lyapunov scalar itself is assembled by :class:`FixedLyapunovFunctional` from
non-negative, frozen coefficients and barrier terms; it is never a free MLP
output.

This module is intentionally separate from ``core.lyapunov``.  The latter is
torch-free and computes ground-truth labels directly from simulator state,
whereas this file provides the learned rollout-end estimator and its losses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, NamedTuple, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class LyapunovPrediction(NamedTuple):
    """Predicted raw physical quantities for one state.

    Shapes use optional leading batch dimensions:

    - ``station_work``: ``(..., S)``
    - ``station_queue_ratio``: ``(..., S)``
    - ``traffic_excess``: ``(..., 1)`` RMS bottleneck-barrier excess
    - ``stationary_excess``: ``(..., 1)`` RMS stationary-barrier excess
    - ``plan_fail_excess``: ``(..., 1)`` RMS plan-failure-barrier excess
    - ``arrival_bins``: ``(..., S, B)``
    """

    station_work: torch.Tensor
    station_queue_ratio: torch.Tensor
    traffic_excess: torch.Tensor
    stationary_excess: torch.Tensor
    plan_fail_excess: torch.Tensor
    arrival_bins: torch.Tensor


class EndpointDemandPredictor(nn.Module):
    """Predict an anchored rollout-end demand-context residual.

    ``RMFSWorldModel.rollout`` keeps the initial demand embedding fixed as a
    transition condition; it does not produce ``e_demand(t+H)``.  The safe
    baseline is therefore persistence, ``d_H = d_0``.  This module learns only
    a signed correction around that baseline::

        d_hat_H = clamp_min(d_0 + delta_d_theta, 0)

    Its final layer is zero-initialised, so an untrained or rejected predictor
    is exactly the persistence baseline rather than an arbitrary positive
    vector.  A TD V-tail or state-signature component head must either use this
    predictor (trained on ``future_demand_context``) or remain disabled for
    rollout endpoints.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        encoded_demand_dim: Optional[int] = None,
        raw_demand_dim: int = 9,
        hidden_dim: int = 128,
        residual_scale=1.0,
        residual_limit: float = 4.0,
    ):
        super().__init__()
        encoded_demand_dim = (
            latent_dim if encoded_demand_dim is None else encoded_demand_dim
        )
        if min(latent_dim, encoded_demand_dim, raw_demand_dim, hidden_dim) <= 0:
            raise ValueError("endpoint demand dimensions must be positive")
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")
        self.latent_dim = int(latent_dim)
        self.encoded_demand_dim = int(encoded_demand_dim)
        self.raw_demand_dim = int(raw_demand_dim)
        self.residual_limit = float(residual_limit)
        scale = torch.as_tensor(residual_scale, dtype=torch.float32).flatten()
        if scale.numel() == 1:
            scale = scale.expand(self.raw_demand_dim).clone()
        if scale.numel() != self.raw_demand_dim or bool((scale <= 0).any()):
            raise ValueError(
                "residual_scale must be positive and scalar or raw_demand_dim"
            )
        self.register_buffer("residual_scale", scale)
        self.register_buffer("enabled", torch.tensor(True, dtype=torch.bool))
        input_dim = 2 * self.latent_dim + self.encoded_demand_dim
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.raw_demand_dim),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def set_enabled(self, enabled: bool) -> None:
        """Freeze the held-out demand-improvement gate."""
        self.enabled.copy_(torch.tensor(
            bool(enabled), dtype=torch.bool, device=self.enabled.device
        ))

    def predict_residual(
        self,
        z_endpoint: torch.Tensor,
        initial_demand_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Return the bounded signed residual ``delta_d``."""
        if (z_endpoint.ndim != 2
                or z_endpoint.size(-1) != self.latent_dim):
            raise ValueError("z_endpoint has an invalid shape")
        if (initial_demand_embedding.ndim != 1
                or initial_demand_embedding.numel() != self.encoded_demand_dim):
            raise ValueError("initial_demand_embedding has an invalid shape")
        pooled = torch.cat((
            z_endpoint.mean(dim=0),
            z_endpoint.max(dim=0).values,
            initial_demand_embedding,
        ), dim=-1)
        return (
            self.residual_limit
            * self.residual_scale.to(dtype=pooled.dtype, device=pooled.device)
            * torch.tanh(self.net(pooled))
        )

    def forward(
        self,
        z_endpoint: torch.Tensor,
        initial_demand_embedding: torch.Tensor,
        initial_raw_demand: torch.Tensor,
        *,
        apply_gate: bool = True,
    ) -> torch.Tensor:
        anchor = torch.as_tensor(
            initial_raw_demand,
            dtype=z_endpoint.dtype,
            device=z_endpoint.device,
        ).flatten()
        if anchor.numel() != self.raw_demand_dim:
            raise ValueError("initial_raw_demand has an invalid shape")
        if not bool(torch.isfinite(anchor).all()):
            raise ValueError("initial_raw_demand contains NaN or Inf")
        candidate = torch.clamp_min(
            anchor + self.predict_residual(
                z_endpoint, initial_demand_embedding
            ),
            0.0,
        )
        if apply_gate and not bool(self.enabled):
            return anchor
        return candidate

    def predict_embedding(
        self,
        z_endpoint: torch.Tensor,
        initial_demand_embedding: torch.Tensor,
        initial_raw_demand: torch.Tensor,
        demand_encoder: nn.Module,
        *,
        apply_gate: bool = True,
    ):
        """Return ``(raw_context_hat, e_demand_endpoint_hat)``."""
        raw = self(
            z_endpoint, initial_demand_embedding, initial_raw_demand,
            apply_gate=apply_gate,
        )
        return raw, demand_encoder(raw)


class LyapunovComponentHead(nn.Module):
    """Predict endpoint physical-component residuals around analytic L0.

    The current state is never reconstructed by a network: callers provide the
    directly computed analytic component ledger as ``anchor``.  The head
    consumes the rollout-end latent/demand signature and predicts only a
    signed change.  The final physical quantities are clamped non-negative::

        x_hat_H = clamp_min(x_0 + delta_x_theta, 0)

    The residual output layers are zero-initialised, making a fresh head the
    exact analytic persistence baseline.  There is still no free scalar
    Lyapunov output; :class:`FixedLyapunovFunctional` assembles it from the
    resulting auditable components.
    """

    def __init__(
        self,
        latent_dim: int = 64,
        demand_dim: Optional[int] = None,
        hidden_dim: int = 128,
        num_stations: int = 4,
        num_arrival_bins: int = 4,
        bottleneck_fraction: float = 0.20,
        residual_scales: Optional[Mapping[str, float]] = None,
        residual_limit: float = 4.0,
    ):
        super().__init__()
        demand_dim = latent_dim if demand_dim is None else int(demand_dim)
        if min(latent_dim, demand_dim, hidden_dim, num_stations,
               num_arrival_bins) <= 0:
            raise ValueError("all dimensions/counts must be positive")
        if not 0.0 < bottleneck_fraction <= 1.0:
            raise ValueError("bottleneck_fraction must be in (0, 1]")
        if residual_limit <= 0.0:
            raise ValueError("residual_limit must be positive")

        self.latent_dim = int(latent_dim)
        self.demand_dim = demand_dim
        self.hidden_dim = int(hidden_dim)
        self.num_stations = int(num_stations)
        self.num_arrival_bins = int(num_arrival_bins)
        self.bottleneck_fraction = float(bottleneck_fraction)
        self.residual_limit = float(residual_limit)
        residual_scales = dict(residual_scales or {})
        for name in LyapunovPrediction._fields:
            scale = float(residual_scales.get(name, 1.0))
            if not torch.isfinite(torch.tensor(scale)) or scale <= 0.0:
                raise ValueError(f"residual scale for {name} must be positive")
            self.register_buffer(
                f"residual_scale_{name}", torch.tensor(scale)
            )
        self.register_buffer(
            "component_gates",
            torch.ones(len(LyapunovPrediction._fields), dtype=torch.bool),
        )

        # Per-station: W_j, physical queue ratio, and ETA-bin arrival masses.
        self.station_net = nn.Sequential(
            nn.LayerNorm(self.latent_dim + self.demand_dim),
            nn.Linear(self.latent_dim + self.demand_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(
                self.hidden_dim,
                2 + self.num_arrival_bins,
            ),
        )
        # Global mean/max + bottleneck mean/max + demand -> traffic/stall/fail.
        global_dim = 4 * self.latent_dim + self.demand_dim
        self.global_net = nn.Sequential(
            nn.LayerNorm(global_dim),
            nn.Linear(global_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, 3),
        )
        nn.init.zeros_(self.station_net[-1].weight)
        nn.init.zeros_(self.station_net[-1].bias)
        nn.init.zeros_(self.global_net[-1].weight)
        nn.init.zeros_(self.global_net[-1].bias)

    @staticmethod
    def _mean_max(z: torch.Tensor) -> torch.Tensor:
        return torch.cat((z.mean(dim=0), z.max(dim=0).values), dim=-1)

    def _bottleneck_pool(
        self,
        z: torch.Tensor,
        bottleneck_scores: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if bottleneck_scores is None:
            return torch.zeros(
                2 * self.latent_dim, dtype=z.dtype, device=z.device
            )
        scores = torch.as_tensor(
            bottleneck_scores, dtype=z.dtype, device=z.device
        ).flatten()
        if scores.numel() != z.size(0):
            raise ValueError(
                "bottleneck_scores length must equal the number of nodes"
            )
        count = max(1, int(round(z.size(0) * self.bottleneck_fraction)))
        ids = torch.topk(
            scores, k=min(count, z.size(0)), largest=True, sorted=False
        ).indices
        return self._mean_max(z.index_select(0, ids))

    def _scale(self, name: str, reference: torch.Tensor) -> torch.Tensor:
        return getattr(self, f"residual_scale_{name}").to(
            dtype=reference.dtype, device=reference.device
        )

    def set_component_gates(self, gates) -> None:
        """Freeze held-out improvement gates in prediction-field order."""
        if isinstance(gates, Mapping):
            values = [bool(gates[name]) for name in LyapunovPrediction._fields]
        else:
            values = list(gates)
        tensor = torch.as_tensor(
            values, dtype=torch.bool, device=self.component_gates.device
        ).flatten()
        if tensor.numel() != len(LyapunovPrediction._fields):
            raise ValueError("component gates have an invalid length")
        self.component_gates.copy_(tensor)

    @staticmethod
    def _validate_anchor(anchor: LyapunovPrediction) -> None:
        if not isinstance(anchor, LyapunovPrediction):
            raise ValueError("anchor must be a LyapunovPrediction")
        for value in anchor:
            if not bool(torch.isfinite(value).all()):
                raise ValueError("anchor contains NaN or Inf")
            if bool((value < 0.0).any()):
                raise ValueError("anchor physical quantities must be non-negative")

    def predict_residual(
        self,
        z: torch.Tensor,
        e_demand: torch.Tensor,
        station_node_ids: Sequence[int],
        bottleneck_scores: Optional[torch.Tensor] = None,
    ) -> LyapunovPrediction:
        """Predict signed component changes for one rollout endpoint.

        ``e_demand`` must describe the same endpoint as ``z``.  Callers that
        only have the initial demand embedding must not use this state-head
        signature for a rollout endpoint.
        """
        if z.ndim != 2 or z.size(-1) != self.latent_dim:
            raise ValueError(
                f"z must have shape (N, {self.latent_dim}), got {tuple(z.shape)}"
            )
        if e_demand.ndim != 1 or e_demand.numel() != self.demand_dim:
            raise ValueError(
                "e_demand must be one-dimensional and match demand_dim"
            )
        ids = torch.as_tensor(
            station_node_ids, dtype=torch.long, device=z.device
        ).flatten()
        if ids.numel() != self.num_stations:
            raise ValueError(
                f"expected {self.num_stations} station ids, got {ids.numel()}"
            )
        if bool(((ids < 0) | (ids >= z.size(0))).any()):
            raise ValueError("station_node_ids contains an invalid node index")

        station_z = z.index_select(0, ids)
        demand_station = e_demand.unsqueeze(0).expand(self.num_stations, -1)
        station_raw = self.station_net(
            torch.cat((station_z, demand_station), dim=-1)
        )

        global_features = torch.cat((
            self._mean_max(z),
            self._bottleneck_pool(z, bottleneck_scores),
            e_demand,
        ), dim=-1)
        global_raw = self.global_net(global_features)

        station_work = self.residual_limit * self._scale(
            "station_work", station_raw
        ) * torch.tanh(station_raw[:, 0])
        station_queue = self.residual_limit * self._scale(
            "station_queue_ratio", station_raw
        ) * torch.tanh(station_raw[:, 1])
        arrival = self.residual_limit * self._scale(
            "arrival_bins", station_raw
        ) * torch.tanh(station_raw[:, 2:]).reshape(
            self.num_stations, self.num_arrival_bins
        )
        traffic = self.residual_limit * self._scale(
            "traffic_excess", global_raw
        ) * torch.tanh(global_raw[0:1])
        stationary = self.residual_limit * self._scale(
            "stationary_excess", global_raw
        ) * torch.tanh(global_raw[1:2])
        plan_fail = self.residual_limit * self._scale(
            "plan_fail_excess", global_raw
        ) * torch.tanh(global_raw[2:3])

        return LyapunovPrediction(
            station_work=station_work,
            station_queue_ratio=station_queue,
            traffic_excess=traffic,
            stationary_excess=stationary,
            plan_fail_excess=plan_fail,
            arrival_bins=arrival,
        )

    def forward(
        self,
        z: torch.Tensor,
        e_demand: torch.Tensor,
        station_node_ids: Sequence[int],
        bottleneck_scores: Optional[torch.Tensor] = None,
        *,
        anchor: LyapunovPrediction,
        apply_component_gates: bool = True,
    ) -> LyapunovPrediction:
        """Return non-negative endpoint components around ``anchor``."""
        self._validate_anchor(anchor)
        residual = self.predict_residual(
            z, e_demand, station_node_ids, bottleneck_scores
        )
        values = []
        for index, (base, delta) in enumerate(zip(anchor, residual)):
            base = base.to(dtype=delta.dtype, device=delta.device)
            if base.shape != delta.shape:
                raise ValueError("anchor shape does not match residual output")
            if (apply_component_gates
                    and not bool(self.component_gates[index])):
                delta = torch.zeros_like(delta)
            values.append(torch.clamp_min(base + delta, 0.0))
        return LyapunovPrediction(*values)


def target_from_analytic_snapshot(
    snapshot,
    station_ids: Sequence[int],
    *,
    dtype: torch.dtype = torch.float32,
    device=None,
) -> LyapunovPrediction:
    """Convert a torch-free ``LyapunovSnapshot`` into head supervision.

    Station ordering is explicit so labels remain stable across maps and
    serialized dictionaries.  The traffic/stall fields are RMS barrier
    excesses; applying :class:`FixedLyapunovFunctional` reconstructs the same
    quadratic terms as the analytic ledger (given matching capacities and
    frozen weights).
    """
    station_ids = [int(value) for value in station_ids]
    if not station_ids:
        raise ValueError("station_ids must not be empty")
    def field_value(name):
        if isinstance(snapshot, Mapping):
            return snapshot[name]
        return getattr(snapshot, name)

    def station_value(mapping, station_id, default=0.0):
        return mapping.get(station_id, mapping.get(str(station_id), default))

    arrival_mapping = field_value("arrival_bins")
    arrival_rows = []
    expected_bins = None
    for station_id in station_ids:
        row = tuple(station_value(arrival_mapping, station_id, ()))
        if expected_bins is None:
            expected_bins = len(row)
        if len(row) != expected_bins:
            raise ValueError("all stations must use the same ETA-bin schema")
        arrival_rows.append(row)
    if not expected_bins:
        raise ValueError("snapshot contains no ETA arrival bins")

    def tensor(value):
        return torch.as_tensor(value, dtype=dtype, device=device)

    return LyapunovPrediction(
        station_work=tensor([
            station_value(field_value("station_work"), station_id, 0.0)
            for station_id in station_ids
        ]),
        station_queue_ratio=tensor([
            station_value(field_value("station_queue_ratio"), station_id, 0.0)
            for station_id in station_ids
        ]),
        traffic_excess=tensor([field_value("traffic_excess_rms")]),
        stationary_excess=tensor([field_value("stationary_excess_rms")]),
        plan_fail_excess=tensor([field_value("plan_fail_excess_rms")]),
        arrival_bins=tensor(arrival_rows),
    )


class FixedLyapunovFunctional(nn.Module):
    """Frozen, non-negative analytic L0 functional for predicted components."""

    COMPONENT_NAMES = ("work", "station", "traffic", "stall", "arrival")

    def __init__(
        self,
        work_capacity,
        arrival_capacity,
        *,
        work_weight: float = 1.0,
        station_weight: float = 1.0,
        traffic_weight: float = 1.0,
        stall_weight: float = 1.0,
        plan_fail_weight: float = 0.5,
        arrival_weight: float = 1.0,
        station_safe_ratio: float = 0.70,
    ):
        super().__init__()
        values = {
            "work_weight": work_weight,
            "station_weight": station_weight,
            "traffic_weight": traffic_weight,
            "stall_weight": stall_weight,
            "plan_fail_weight": plan_fail_weight,
            "arrival_weight": arrival_weight,
            "station_safe_ratio": station_safe_ratio,
        }
        for name, value in values.items():
            if float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")

        work_capacity = torch.as_tensor(work_capacity, dtype=torch.float32)
        arrival_capacity = torch.as_tensor(arrival_capacity, dtype=torch.float32)
        if work_capacity.ndim != 1 or bool((work_capacity <= 0).any()):
            raise ValueError("work_capacity must be a positive (S,) vector")
        if (arrival_capacity.ndim != 2
                or arrival_capacity.size(0) != work_capacity.numel()
                or bool((arrival_capacity < 0).any())):
            raise ValueError(
                "arrival_capacity must be a non-negative (S, B) matrix"
            )
        self.register_buffer("work_capacity", work_capacity)
        self.register_buffer("arrival_capacity", arrival_capacity)
        self.work_weight = float(work_weight)
        self.station_weight = float(station_weight)
        self.traffic_weight = float(traffic_weight)
        self.stall_weight = float(stall_weight)
        self.plan_fail_weight = float(plan_fail_weight)
        self.arrival_weight = float(arrival_weight)
        self.station_safe_ratio = float(station_safe_ratio)

    def forward(self, prediction: LyapunovPrediction) -> Dict[str, torch.Tensor]:
        if prediction.station_work.shape[-1:] != self.work_capacity.shape:
            raise ValueError("station_work shape does not match work_capacity")
        if prediction.station_queue_ratio.shape != prediction.station_work.shape:
            raise ValueError("station queue/work shapes must match")
        if prediction.arrival_bins.shape[-2:] != self.arrival_capacity.shape:
            raise ValueError("arrival_bins shape does not match arrival_capacity")

        work = 0.5 * self.work_weight * (
            prediction.station_work / self.work_capacity
        ).square().sum(dim=-1)
        station = 0.5 * self.station_weight * F.relu(
            prediction.station_queue_ratio - self.station_safe_ratio
        ).square().sum(dim=-1)
        traffic = 0.5 * self.traffic_weight * (
            prediction.traffic_excess.square().sum(dim=-1)
        )
        stall = (
            0.5 * self.stall_weight
            * prediction.stationary_excess.square().sum(dim=-1)
            + 0.5 * self.plan_fail_weight
            * prediction.plan_fail_excess.square().sum(dim=-1)
        )
        arrival = 0.5 * self.arrival_weight * F.relu(
            prediction.arrival_bins - self.arrival_capacity
        ).square().sum(dim=(-2, -1))
        total = work + station + traffic + stall + arrival
        return {
            "work": work,
            "station": station,
            "traffic": traffic,
            "stall": stall,
            "arrival": arrival,
            "total": total,
        }


@dataclass(frozen=True)
class LyapunovLossWeights:
    component: float = 1.0
    value: float = 1.0
    drift: float = 1.0
    balance: float = 1.0
    sign: float = 0.25

    def __post_init__(self):
        for name in ("component", "value", "drift", "balance", "sign"):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} loss weight must be non-negative")


def _component_huber(
    prediction: LyapunovPrediction,
    target: LyapunovPrediction,
    scales: Optional[Mapping[str, object]] = None,
) -> torch.Tensor:
    losses = []
    scales = dict(scales or {})
    for name, predicted, expected in zip(
            LyapunovPrediction._fields, prediction, target):
        scale = torch.as_tensor(
            scales.get(name, 1.0),
            dtype=predicted.dtype,
            device=predicted.device,
        )
        if bool((scale <= 0).any()) or not bool(torch.isfinite(scale).all()):
            raise ValueError(f"component scale for {name} must be positive")
        losses.append(F.smooth_l1_loss(
            (predicted - expected) / scale,
            torch.zeros_like(predicted),
        ))
    return torch.stack(losses).mean()


def compute_lyapunov_training_loss(
    current_prediction: LyapunovPrediction,
    future_prediction: LyapunovPrediction,
    current_target: LyapunovPrediction,
    future_target: LyapunovPrediction,
    functional: FixedLyapunovFunctional,
    *,
    balance_target: Optional[torch.Tensor] = None,
    weights: Optional[LyapunovLossWeights] = None,
    sign_margin: float = 0.0,
    component_scales: Optional[Mapping[str, object]] = None,
    value_scale: float = 1.0,
    drift_scale: float = 1.0,
    balance_scale: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Compute component/value/drift/balance/sign losses.

    ``balance_target`` is the registered physical ledger target for
    ``Delta W`` (normally ``A - P_prod`` plus any explicitly modelled reverse
    residual).  Passing it explicitly prevents new-order injection from being
    silently attributed to the scheduling action.
    """
    weights = weights or LyapunovLossWeights()
    if sign_margin < 0.0:
        raise ValueError("sign_margin must be non-negative")
    if min(float(value_scale), float(drift_scale), float(balance_scale)) <= 0.0:
        raise ValueError("value/drift/balance scales must be positive")

    current_values = functional(current_prediction)
    future_values = functional(future_prediction)
    current_true_values = functional(current_target)
    future_true_values = functional(future_target)

    component = 0.5 * (
        _component_huber(
            current_prediction, current_target, component_scales
        )
        + _component_huber(
            future_prediction, future_target, component_scales
        )
    )
    value = 0.5 * (
        F.smooth_l1_loss(
            (current_values["total"] - current_true_values["total"])
            / float(value_scale),
            torch.zeros_like(current_values["total"]),
        )
        + F.smooth_l1_loss(
            (future_values["total"] - future_true_values["total"])
            / float(value_scale),
            torch.zeros_like(future_values["total"]),
        )
    )
    predicted_drift = future_values["total"] - current_values["total"]
    target_drift = future_true_values["total"] - current_true_values["total"]
    drift = F.smooth_l1_loss(
        (predicted_drift - target_drift) / float(drift_scale),
        torch.zeros_like(predicted_drift),
    )

    predicted_delta_work = (
        future_prediction.station_work.sum(dim=-1)
        - current_prediction.station_work.sum(dim=-1)
    )
    if balance_target is None:
        balance = predicted_delta_work.new_zeros(())
    else:
        balance = F.smooth_l1_loss(
            (
                predicted_delta_work
                - balance_target.to(predicted_delta_work)
            ) / float(balance_scale),
            torch.zeros_like(predicted_delta_work),
        )

    direction = torch.sign(target_drift.detach())
    active = direction != 0
    if bool(active.any()):
        sign = F.relu(
            sign_margin - direction[active] * predicted_drift[active]
        ).mean()
    else:
        sign = predicted_drift.new_zeros(())

    total = (
        weights.component * component
        + weights.value * value
        + weights.drift * drift
        + weights.balance * balance
        + weights.sign * sign
    )
    return {
        "total": total,
        "component": component,
        "value": value,
        "drift": drift,
        "balance": balance,
        "sign": sign,
        "predicted_drift": predicted_drift,
        "target_drift": target_drift,
    }


__all__ = [
    "EndpointDemandPredictor",
    "FixedLyapunovFunctional",
    "LyapunovComponentHead",
    "LyapunovLossWeights",
    "LyapunovPrediction",
    "compute_lyapunov_training_loss",
    "target_from_analytic_snapshot",
]
