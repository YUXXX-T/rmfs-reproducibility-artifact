"""Residual correction around the analytic free-flow work-relief baseline.

The fixed Lyapunov object remains the quadratic station-work potential.  This
head does **not** learn a replacement potential and it does not emit a free
scalar score.  Given a nominal endpoint produced from the analytic work
ledger, it learns only the signed station-wise relief residual caused by
blocking, reservations, queues and multi-robot interaction::

    mu_hat = clamp(mu_nominal + residual_hat, 0, mu_available_pipeline)
    W_hat(t+H) = W_post_action - mu_hat
    Delta L_hat = L_work(W_hat(t+H)) - L_work(W_start)

The final residual layer is zero-initialised.  Consequently a fresh head is
exactly the parameter-free analytic baseline, which makes the A/B/C ablation
auditable and prevents the neural network from silently redefining
``L_work``.  Unknown future orders, continuation policies and TD tails are
outside this module's contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch
import torch.nn as nn


SCHEMA_VERSION = "analytic_work_relief_residual_head_v1"


@dataclass(frozen=True)
class AnalyticWorkResidualFeatures:
    """Compact frozen-WM features for one candidate action.

    ``station_context`` ends with four physical ratios, in this order:
    ``W_start / capacity``, ``W_post_action / capacity``,
    ``mu_nominal / capacity`` and ``mu_available / capacity``.  Keeping these
    values inside the immutable feature tensor lets evaluation use arbitrary
    candidate-group sizes while
    retaining an explicit, reconstructible analytic baseline.
    """

    global_context: torch.Tensor
    station_context: torch.Tensor


@dataclass(frozen=True)
class AnalyticWorkResidualPrediction:
    nominal_station_relief: torch.Tensor
    available_station_relief: torch.Tensor
    residual_station_relief: torch.Tensor
    applied_residual_station_relief: torch.Tensor
    unconstrained_station_relief: torch.Tensor
    predicted_station_relief: torch.Tensor
    nominal_endpoint_station_work: torch.Tensor
    endpoint_station_work: torch.Tensor
    nominal_raw_work_drift: torch.Tensor
    raw_work_drift: torch.Tensor


class AnalyticWorkResidualHead(nn.Module):
    """Learn station-wise relief residuals around a fixed analytic baseline."""

    _START_RATIO_OFFSET = -4
    _POST_RATIO_OFFSET = -3
    _NOMINAL_RELIEF_RATIO_OFFSET = -2
    _AVAILABLE_RELIEF_RATIO_OFFSET = -1

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
        self.station_dim = 2 * self.latent_dim + 4
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
        # Exact analytic fallback at initialisation.  The first optimisation
        # step still receives gradients through this output layer.
        nn.init.zeros_(self.station_net[-1].weight)
        nn.init.zeros_(self.station_net[-1].bias)

    @staticmethod
    def _mean_max(value: torch.Tensor) -> torch.Tensor:
        if value.ndim < 2:
            raise ValueError("node latent tensors must have a node axis")
        return torch.cat((value.mean(dim=-2), value.amax(dim=-2)), dim=-1)

    def _physical_vector(
        self,
        value: torch.Tensor | Sequence[float],
        *,
        reference: torch.Tensor,
        name: str,
        non_negative: bool = True,
    ) -> torch.Tensor:
        result = torch.as_tensor(
            value, dtype=reference.dtype, device=reference.device
        ).flatten()
        if result.numel() != self.num_stations:
            raise ValueError(f"{name} must contain exactly S values")
        if non_negative and bool((result < 0.0).any()):
            raise ValueError(f"{name} must be non-negative")
        return result

    def build_features(
        self,
        z_start: torch.Tensor,
        z_endpoint: torch.Tensor,
        current_demand_embedding: torch.Tensor,
        station_node_ids: Sequence[int] | torch.Tensor,
        start_station_work: torch.Tensor | Sequence[float],
        post_action_station_work: torch.Tensor | Sequence[float],
        nominal_station_relief: torch.Tensor | Sequence[float],
        available_station_relief: torch.Tensor | Sequence[float],
    ) -> AnalyticWorkResidualFeatures:
        """Build features from current-known state and the analytic baseline."""

        for name, value in (("z_start", z_start), ("z_endpoint", z_endpoint)):
            if value.ndim != 2 or value.size(-1) != self.latent_dim:
                raise ValueError(f"{name} must have shape (N, latent_dim)")
        if z_start.shape != z_endpoint.shape:
            raise ValueError("z_start and z_endpoint must align")
        if (
            current_demand_embedding.ndim != 1
            or current_demand_embedding.numel() != self.latent_dim
        ):
            raise ValueError("current_demand_embedding must have shape (latent_dim,)")
        indices = torch.as_tensor(
            station_node_ids, dtype=torch.long, device=z_start.device
        ).flatten()
        if indices.numel() != self.num_stations:
            raise ValueError("station_node_ids must contain exactly S indices")
        if bool((indices < 0).any()) or bool((indices >= z_start.size(0)).any()):
            raise ValueError("station_node_ids contains an out-of-range index")

        start = self._physical_vector(
            start_station_work, reference=z_start, name="start_station_work"
        )
        post = self._physical_vector(
            post_action_station_work,
            reference=z_start,
            name="post_action_station_work",
        )
        nominal = self._physical_vector(
            nominal_station_relief,
            reference=z_start,
            name="nominal_station_relief",
        )
        available = self._physical_vector(
            available_station_relief,
            reference=z_start,
            name="available_station_relief",
        )
        tolerance = 1e-5 * torch.maximum(torch.ones_like(available), available)
        if bool((nominal > available + tolerance).any()):
            raise ValueError("nominal relief exceeds available pipeline relief")
        if bool((available > post + tolerance).any()):
            raise ValueError("available pipeline relief exceeds post-action work")
        capacity = self.work_capacity.to(dtype=z_start.dtype, device=z_start.device)
        global_context = torch.cat((
            self._mean_max(z_start),
            self._mean_max(z_endpoint),
            current_demand_embedding,
        ), dim=-1)
        physical = torch.stack((
            start / capacity,
            post / capacity,
            nominal / capacity,
            available / capacity,
        ), dim=-1)
        station_context = torch.cat((
            z_start.index_select(0, indices),
            z_endpoint.index_select(0, indices),
            physical,
        ), dim=-1)
        return AnalyticWorkResidualFeatures(global_context, station_context)

    def work_potential(self, station_work: torch.Tensor) -> torch.Tensor:
        capacity = self.work_capacity.to(
            dtype=station_work.dtype, device=station_work.device
        )
        if station_work.shape[-1:] != capacity.shape:
            raise ValueError("station_work must end with the station dimension S")
        return 0.5 * self.work_weight * (station_work / capacity).square().sum(dim=-1)

    def _baseline_from_station_context(
        self, station_context: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        capacity = self.work_capacity.to(
            dtype=station_context.dtype, device=station_context.device
        )
        start = station_context[..., self._START_RATIO_OFFSET] * capacity
        post = station_context[..., self._POST_RATIO_OFFSET] * capacity
        nominal = (
            station_context[..., self._NOMINAL_RELIEF_RATIO_OFFSET] * capacity
        )
        available = (
            station_context[..., self._AVAILABLE_RELIEF_RATIO_OFFSET] * capacity
        )
        return start, post, nominal, available

    def predict_from_features(
        self,
        global_context: torch.Tensor,
        station_context: torch.Tensor,
        start_station_work: torch.Tensor,
    ) -> AnalyticWorkResidualPrediction:
        """Predict a candidate batch while retaining an analytic zero-head path."""

        if global_context.shape[-1:] != (self.global_dim,):
            raise ValueError("global_context has an invalid final dimension")
        if station_context.shape[-2:] != (self.num_stations, self.station_dim):
            raise ValueError("station_context has an invalid station shape")
        if start_station_work.shape[-1:] != (self.num_stations,):
            raise ValueError("start_station_work has an invalid station shape")
        if station_context.shape[:-2] != global_context.shape[:-1]:
            raise ValueError("global and station feature batch dimensions differ")
        if start_station_work.shape[:-1] != global_context.shape[:-1]:
            raise ValueError("start-work and feature batch dimensions differ")

        encoded_start, post, nominal, available = self._baseline_from_station_context(
            station_context
        )
        # The explicit argument is used by the potential reconstruction.  The
        # encoded copy is checked so compact feature files cannot silently
        # pair a candidate with another context's physical ledger.
        tolerance = 1e-5 * torch.maximum(
            torch.ones_like(start_station_work), start_station_work.abs()
        )
        if bool(((encoded_start - start_station_work).abs() > tolerance).any()):
            raise ValueError("encoded and explicit start_station_work differ")

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
        residual = self.residual_limit * scale * residual_unit
        unconstrained_relief = nominal + residual
        # Positive relief may never exceed the work already present in active
        # pipeline chains, so the network cannot dissipate pending work.
        # The lower side intentionally remains signed: a detour or reverse
        # physical move can increase the current remaining-work ledger over a
        # short horizon and must not be hidden by a zero clamp.
        predicted_relief = torch.minimum(unconstrained_relief, available)
        applied_residual = predicted_relief - nominal
        nominal_endpoint = torch.clamp_min(post - nominal, 0.0)
        endpoint = torch.clamp_min(post - predicted_relief, 0.0)
        baseline_potential = self.work_potential(start_station_work)
        nominal_drift = self.work_potential(nominal_endpoint) - baseline_potential
        drift = self.work_potential(endpoint) - baseline_potential
        return AnalyticWorkResidualPrediction(
            nominal_station_relief=nominal,
            available_station_relief=available,
            residual_station_relief=residual,
            applied_residual_station_relief=applied_residual,
            unconstrained_station_relief=unconstrained_relief,
            predicted_station_relief=predicted_relief,
            nominal_endpoint_station_work=nominal_endpoint,
            endpoint_station_work=endpoint,
            nominal_raw_work_drift=nominal_drift,
            raw_work_drift=drift,
        )

    def forward(
        self,
        z_start: torch.Tensor,
        z_endpoint: torch.Tensor,
        current_demand_embedding: torch.Tensor,
        station_node_ids: Sequence[int] | torch.Tensor,
        start_station_work: torch.Tensor,
        post_action_station_work: torch.Tensor,
        nominal_station_relief: torch.Tensor,
        available_station_relief: torch.Tensor,
    ) -> AnalyticWorkResidualPrediction:
        features = self.build_features(
            z_start,
            z_endpoint,
            current_demand_embedding,
            station_node_ids,
            start_station_work,
            post_action_station_work,
            nominal_station_relief,
            available_station_relief,
        )
        return self.predict_from_features(
            features.global_context, features.station_context, start_station_work
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
    ) -> tuple["AnalyticWorkResidualHead", Mapping]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"expected {SCHEMA_VERSION}, got {payload.get('schema_version')!r}"
            )
        head = cls(**dict(payload["head_config"]))
        head.load_state_dict(payload["head_state_dict"], strict=True)
        head.to(map_location).eval()
        return head, payload


__all__ = [
    "SCHEMA_VERSION",
    "AnalyticWorkResidualFeatures",
    "AnalyticWorkResidualHead",
    "AnalyticWorkResidualPrediction",
]
