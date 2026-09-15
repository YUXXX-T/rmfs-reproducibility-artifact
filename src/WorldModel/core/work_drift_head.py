"""Finite-horizon work-drift estimator on top of a frozen World Model.

The estimator is deliberately narrower than ``LyapunovComponentHead``.  It
uses the current analytic station-work ledger together with the frozen World
Model's action-conditioned ``z_0``/``z_H`` representation and predicts only
the endpoint station-work residual.  ``Delta L_work`` is then reconstructed by
the fixed analytic quadratic functional; it is never a free scalar output.

Unknown future orders, a continuation policy, TD tails and load/gap gates are
outside this module's contract.  Candidate-group range normalisation is a
separate comparison transform and does not redefine the raw physical ledger.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


SCHEMA_VERSION = "work_drift_group_range_head_v1"


@dataclass(frozen=True)
class WorkDriftFeatures:
    """Compact frozen-WM features for one candidate action.

    ``global_context`` has shape ``(..., 5D)`` and contains mean/max pooled
    ``z_0``, ``z_H`` plus the current known demand embedding.  The candidate
    action already conditions ``z_H`` through the frozen World Model rollout;
    exposing it again would let the head bypass the dynamics being tested.
    ``station_context`` has shape ``(..., S, 2D+1)`` and
    contains station-local ``z_0``, ``z_H`` and normalised current work.
    """

    global_context: torch.Tensor
    station_context: torch.Tensor


@dataclass(frozen=True)
class WorkDriftPrediction:
    endpoint_station_work: torch.Tensor
    raw_work_drift: torch.Tensor


def group_range_normalise(
    values: torch.Tensor,
    *,
    dim: int = -1,
) -> torch.Tensor:
    """Return ``(x - mean(x)) / (max(x) - min(x))``.

    There is no practical-gap threshold.  An exactly tied group maps to zero;
    every non-zero range, however small, uses its exact observed denominator.
    This matches the frozen Layer-3 transform instead of silently introducing
    a dtype-dependent gap.
    """

    if not torch.is_floating_point(values):
        raise TypeError("group-range normalisation requires a floating tensor")
    if values.size(dim) < 1:
        raise ValueError("cannot normalise an empty candidate group")
    centred = values - values.mean(dim=dim, keepdim=True)
    value_range = (
        values.amax(dim=dim, keepdim=True)
        - values.amin(dim=dim, keepdim=True)
    )
    tied = value_range == 0.0
    safe_range = torch.where(tied, torch.ones_like(value_range), value_range)
    normalised = centred / safe_range
    return torch.where(tied, torch.zeros_like(normalised), normalised)


def all_non_tie_pairwise_logistic_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float = 0.25,
) -> tuple[torch.Tensor, int]:
    """Tie-safe ranking loss over every non-tied target pair.

    No raw target gap is used.  Exact analytic ties are excluded because they
    carry no direction.  Lower and higher values retain their natural order.
    """

    if predicted.ndim != 1 or target.ndim != 1 or predicted.shape != target.shape:
        raise ValueError("pairwise inputs must be aligned one-dimensional tensors")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if predicted.numel() < 2:
        return predicted.new_zeros(()), 0
    left, right = torch.triu_indices(
        predicted.numel(), predicted.numel(), offset=1, device=predicted.device
    )
    target_gap = target[left] - target[right]
    non_tie = target_gap != 0.0
    if not bool(non_tie.any()):
        return predicted.new_zeros(()), 0
    direction = target_gap[non_tie].sign()
    predicted_gap = predicted[left[non_tie]] - predicted[right[non_tie]]
    loss = F.softplus(-direction * predicted_gap / float(temperature)).mean()
    return loss, int(non_tie.sum().item())


class WorkDriftHead(nn.Module):
    """Predict endpoint station work and reconstruct analytic ``Delta L_work``.

    The frozen World Model supplies all latent/action features.  The only
    learned output is a bounded per-station residual around the current
    simulator ledger::

        W_hat_j(t+H) = clamp_min(W_j(t) + delta_W_j, 0)
        Delta L_hat_work = L_work(W_hat(t+H)) - L_work(W(t))

    The class accepts arbitrary candidate-group sizes because candidates are
    scored independently; group-range normalisation is applied afterwards.
    """

    def __init__(
        self,
        *,
        latent_dim: int = 64,
        hidden_dim: int = 128,
        num_stations: int = 4,
        work_capacity: Sequence[float] | float = 1.0,
        work_weight: float = 1.0,
        residual_scale: Sequence[float] | float = 1.0,
        residual_limit: float = 4.0,
    ) -> None:
        super().__init__()
        if min(int(latent_dim), int(hidden_dim), int(num_stations)) <= 0:
            raise ValueError("latent/hidden/station dimensions must be positive")
        if float(work_weight) < 0.0:
            raise ValueError("work_weight must be non-negative")
        if float(residual_limit) <= 0.0:
            raise ValueError("residual_limit must be positive")
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_stations = int(num_stations)
        self.work_weight = float(work_weight)
        self.residual_limit = float(residual_limit)

        capacity = torch.as_tensor(work_capacity, dtype=torch.float32).flatten()
        if capacity.numel() == 1:
            capacity = capacity.expand(self.num_stations).clone()
        if capacity.numel() != self.num_stations or bool((capacity <= 0.0).any()):
            raise ValueError("work_capacity must be positive and scalar or length S")
        scale = torch.as_tensor(residual_scale, dtype=torch.float32).flatten()
        if scale.numel() == 1:
            scale = scale.expand(self.num_stations).clone()
        if scale.numel() != self.num_stations or bool((scale <= 0.0).any()):
            raise ValueError("residual_scale must be positive and scalar or length S")
        self.register_buffer("work_capacity", capacity)
        self.register_buffer("residual_scale", scale)

        self.global_dim = 5 * self.latent_dim
        self.station_dim = 2 * self.latent_dim + 1
        self.global_net = nn.Sequential(
            nn.LayerNorm(self.global_dim),
            nn.Linear(self.global_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
        )
        self.station_net = nn.Sequential(
            nn.LayerNorm(self.station_dim + self.hidden_dim),
            nn.Linear(self.station_dim + self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, 1),
        )
        # A fresh estimator is exactly the analytic persistence baseline.
        # The raw-drift and pairwise losses still backpropagate through this
        # zero-initialised output layer on the first optimisation step.
        nn.init.zeros_(self.station_net[-1].weight)
        nn.init.zeros_(self.station_net[-1].bias)

    @staticmethod
    def _mean_max(value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 2:
            raise ValueError("node latent/action tensors must have a node axis")
        return torch.cat((value.mean(dim=-2), value.amax(dim=-2)), dim=-1)

    def build_features(
        self,
        z_start: torch.Tensor,
        z_endpoint: torch.Tensor,
        current_demand_embedding: torch.Tensor,
        station_node_ids: Sequence[int] | torch.Tensor,
        current_station_work: torch.Tensor,
    ) -> WorkDriftFeatures:
        """Build compact features without reconstructing future demand/state."""

        expected_latent = self.latent_dim
        for name, value in (
            ("z_start", z_start),
            ("z_endpoint", z_endpoint),
        ):
            if value.ndim != 2 or value.size(-1) != expected_latent:
                raise ValueError(f"{name} must have shape (N, latent_dim)")
        if z_start.shape != z_endpoint.shape:
            raise ValueError("z_start and z_endpoint must align")
        if (
            current_demand_embedding.ndim != 1
            or current_demand_embedding.numel() != expected_latent
        ):
            raise ValueError("current_demand_embedding must have shape (latent_dim,)")
        indices = torch.as_tensor(
            station_node_ids, dtype=torch.long, device=z_start.device
        ).flatten()
        if indices.numel() != self.num_stations:
            raise ValueError("station_node_ids must contain exactly S indices")
        if bool((indices < 0).any()) or bool((indices >= z_start.size(0)).any()):
            raise ValueError("station_node_ids contains an out-of-range index")
        current = torch.as_tensor(
            current_station_work, dtype=z_start.dtype, device=z_start.device
        ).flatten()
        if current.numel() != self.num_stations or bool((current < 0.0).any()):
            raise ValueError("current_station_work must be non-negative length S")

        global_context = torch.cat((
            self._mean_max(z_start),
            self._mean_max(z_endpoint),
            current_demand_embedding,
        ), dim=-1)
        current_ratio = current / self.work_capacity.to(
            dtype=current.dtype, device=current.device
        )
        station_context = torch.cat((
            z_start.index_select(0, indices),
            z_endpoint.index_select(0, indices),
            current_ratio.unsqueeze(-1),
        ), dim=-1)
        return WorkDriftFeatures(global_context, station_context)

    def work_potential(self, station_work: torch.Tensor) -> torch.Tensor:
        capacity = self.work_capacity.to(
            dtype=station_work.dtype, device=station_work.device
        )
        if station_work.shape[-1:] != capacity.shape:
            raise ValueError("station_work must end with the station dimension S")
        return 0.5 * self.work_weight * (station_work / capacity).square().sum(dim=-1)

    def predict_from_features(
        self,
        global_context: torch.Tensor,
        station_context: torch.Tensor,
        current_station_work: torch.Tensor,
    ) -> WorkDriftPrediction:
        """Predict one candidate or a leading batch of candidates."""

        if global_context.shape[-1:] != (self.global_dim,):
            raise ValueError("global_context has an invalid final dimension")
        if station_context.shape[-2:] != (self.num_stations, self.station_dim):
            raise ValueError("station_context has an invalid station shape")
        if current_station_work.shape[-1:] != (self.num_stations,):
            raise ValueError("current_station_work has an invalid station shape")
        if station_context.shape[:-2] != global_context.shape[:-1]:
            raise ValueError("global and station feature batch dimensions differ")
        if current_station_work.shape[:-1] != global_context.shape[:-1]:
            raise ValueError("current-work and feature batch dimensions differ")

        global_hidden = self.global_net(global_context)
        expanded = global_hidden.unsqueeze(-2).expand(
            *global_hidden.shape[:-1], self.num_stations, self.hidden_dim
        )
        residual_unit = torch.tanh(
            self.station_net(torch.cat((station_context, expanded), dim=-1))
            .squeeze(-1)
        )
        scale = self.residual_scale.to(
            dtype=residual_unit.dtype, device=residual_unit.device
        )
        endpoint = torch.clamp_min(
            current_station_work
            + self.residual_limit * scale * residual_unit,
            0.0,
        )
        drift = self.work_potential(endpoint) - self.work_potential(
            current_station_work
        )
        return WorkDriftPrediction(endpoint, drift)

    def forward(
        self,
        z_start: torch.Tensor,
        z_endpoint: torch.Tensor,
        current_demand_embedding: torch.Tensor,
        station_node_ids: Sequence[int] | torch.Tensor,
        current_station_work: torch.Tensor,
    ) -> WorkDriftPrediction:
        features = self.build_features(
            z_start,
            z_endpoint,
            current_demand_embedding,
            station_node_ids,
            current_station_work,
        )
        return self.predict_from_features(
            features.global_context,
            features.station_context,
            current_station_work,
        )

    def checkpoint_config(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "num_stations": self.num_stations,
            "work_capacity": self.work_capacity.detach().cpu().tolist(),
            "work_weight": self.work_weight,
            "residual_scale": self.residual_scale.detach().cpu().tolist(),
            "residual_limit": self.residual_limit,
        }

    @classmethod
    def from_checkpoint(
        cls,
        path: str,
        *,
        map_location: str | torch.device = "cpu",
    ) -> tuple["WorkDriftHead", Mapping]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"expected {SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
            )
        head = cls(**dict(payload["head_config"]))
        head.load_state_dict(payload["head_state_dict"], strict=True)
        head.to(map_location).eval()
        return head, payload
