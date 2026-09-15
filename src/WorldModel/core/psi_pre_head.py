"""Small behavior-aligned readout used only by the ``psi_pre`` kill test.

The head is intentionally independent from the deployed station decoder and
from :mod:`WorldModel.core.station_congestion_head`.  It consumes a variable
number of station rows represented by a shared station/region latent and
returns a fixed set of *named physical channels* per row.  No station-count
dependent parameter is learned.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


PSI_PRE_HEAD_SCHEMA_VERSION = "psi_pre_head_v1"


class PsiPreHead(nn.Module):
    """Shared affine-logistic station-region probe.

    A linear readout is used for the first kill test on purpose.  If this
    simple, spatially shared probe cannot recover endpoint congestion, the
    result is easy to interpret and does not hide a large MLP's memorisation.
    A future experiment may add capacity under a new frozen schema.
    """

    schema_version = PSI_PRE_HEAD_SCHEMA_VERSION

    def __init__(self, latent_dim: int, output_dim: int):
        super().__init__()
        if int(latent_dim) <= 0 or int(output_dim) <= 0:
            raise ValueError("latent_dim and output_dim must be positive")
        self.latent_dim = int(latent_dim)
        self.output_dim = int(output_dim)
        self.proj = nn.Linear(self.latent_dim, self.output_dim)

    def forward(self, station_region_latents: torch.Tensor) -> torch.Tensor:
        if station_region_latents.ndim != 2:
            raise ValueError(
                "station_region_latents must have shape (rows, latent_dim)"
            )
        if station_region_latents.size(1) != self.latent_dim:
            raise ValueError(
                f"expected latent dimension {self.latent_dim}, got "
                f"{station_region_latents.size(1)}"
            )
        return torch.sigmoid(self.proj(station_region_latents))

    @staticmethod
    def concat_station_region(
        z: torch.Tensor,
        station_node_ids: Sequence[int] | torch.Tensor,
        station_region_node_ids: Sequence[Sequence[int]],
    ) -> torch.Tensor:
        """Build ``[station_z, region_mean_z, region_max_z]`` rows."""

        if z.ndim != 2:
            raise ValueError("z must have shape (nodes, latent_dim)")
        ids = torch.as_tensor(
            station_node_ids, dtype=torch.long, device=z.device
        ).reshape(-1)
        if ids.numel() == 0:
            raise ValueError("station_node_ids cannot be empty")
        if len(station_region_node_ids) != int(ids.numel()):
            raise ValueError("station region count differs from station count")
        if int(ids.min().item()) < 0 or int(ids.max().item()) >= z.size(0):
            raise ValueError("station node id is outside latent node range")
        rows = []
        for station_id, region in zip(ids.tolist(), station_region_node_ids):
            region_ids = torch.as_tensor(
                list(region), dtype=torch.long, device=z.device
            ).reshape(-1)
            if region_ids.numel() == 0:
                raise ValueError("station region cannot be empty")
            if int(region_ids.min().item()) < 0 or int(region_ids.max().item()) >= z.size(0):
                raise ValueError("station region node id is outside latent range")
            region_z = z.index_select(0, region_ids)
            rows.append(torch.cat([
                z[station_id],
                region_z.mean(dim=0),
                region_z.max(dim=0).values,
            ], dim=-1))
        return torch.stack(rows, dim=0)


__all__ = ["PSI_PRE_HEAD_SCHEMA_VERSION", "PsiPreHead"]
