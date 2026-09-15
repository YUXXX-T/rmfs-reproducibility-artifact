"""
Realized Cost Computation
=========================
Shared cost formula used by both data generation (realized_cost)
and the model's CostHead. Ensures label-model consistency.
"""

import torch


DEFAULT_LAMBDAS = [
    1.0,   # total_wait_time
    0.5,   # avg_excess_delay
    0.5,   # station_queue_delta
    0.5,   # station_load_imbalance
    1.0,   # bottleneck_CVaR
    -1.0,  # completed_orders_delta (reward)
    2.0,   # deadlock_risk
]

CONGESTION_LAMBDAS = [
    1.0,   # total_wait_time
    0.5,   # avg_excess_delay
    0.5,   # station_queue_delta
    0.5,   # station_load_imbalance
    1.0,   # bottleneck_CVaR
    0.0,   # completed_orders_delta (excluded)
    2.0,   # deadlock_risk
]


def compute_realized_cost(
    future_system_labels: torch.Tensor,
    gamma: float = 0.95,
    lambdas: list = None,
    risk_weight: float = None,
) -> float:
    """Weighted discounted sum of 7-dim system labels.

    Deadlock risk (dim 6) uses max across steps instead of discounted sum.

    Parameters
    ----------
    future_system_labels : (H, 7)
    gamma : discount factor
    lambdas : per-dim weights, length 7
    risk_weight : if not None, overrides lambdas[6] (set 0 for ablation)

    Returns
    -------
    cost : float
    """
    if lambdas is None:
        lambdas = list(DEFAULT_LAMBDAS)
    else:
        lambdas = list(lambdas)

    if risk_weight is not None:
        lambdas[6] = risk_weight

    lam = torch.tensor(lambdas, dtype=torch.float32)
    H = future_system_labels.shape[0]
    discounts = torch.tensor([gamma ** k for k in range(H)], dtype=torch.float32)

    weighted = future_system_labels * lam.unsqueeze(0)
    risk_max = weighted[:, 6].max().item()
    non_risk = weighted[:, :6].sum(dim=-1)
    return float((discounts * non_risk).sum().item()) + risk_max
