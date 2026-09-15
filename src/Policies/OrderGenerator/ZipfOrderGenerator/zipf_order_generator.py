"""
Zipf Order Generator
====================
Generates orders using a Zipf (power-law) distribution over SKUs.

This models real-world warehouse behavior where a small number of
popular products (SKUs) account for a disproportionately large share
of orders — the "80/20 rule".
"""

import random
import numpy as np
from typing import List, Dict

from Policies.OrderGenerator.base_order_generator import BaseOrderGenerator
from WorldState.order_state import Order


class ZipfOrderGenerator(BaseOrderGenerator):
    """
    Generates orders at regular intervals, selecting SKUs according
    to a Zipf (power-law) distribution.

    SKUs are ranked by their name. Lower-ranked SKUs are ordered
    exponentially more often than higher-ranked ones, controlled
    by the ``zipf_param`` exponent.

    参数
    ----------
    order_interval : int
        Generate a new order every N ticks.
    max_items_per_order : int
        Maximum number of distinct SKU types per order.
    zipf_param : float
        Zipf exponent (a > 1).  Higher values make the distribution
        more skewed toward popular SKUs.  Typical values: 1.2–2.0.
        Default is 1.5.
    max_items_per_sku : int
        每种 SKU 需求的物品数量上限。
    """

    def __init__(
        self,
        order_interval: int = 5,
        max_items_per_order: int = 2,
        zipf_param: float = 1.5,
        fixed_order_size: bool = False,
        max_items_per_sku: int = 5,
    ):
        self.order_interval = order_interval
        self.max_items_per_order = max_items_per_order
        self.zipf_param = zipf_param
        self.fixed_order_size = fixed_order_size
        self.max_items_per_sku = max_items_per_sku

    def _zipf_weights(self, n: int) -> np.ndarray:
        """
        Compute Zipf probability weights for *n* items.

        返回值
        -------
        np.ndarray
            Normalised probability array of length *n*.
        """
        ranks = np.arange(1, n + 1, dtype=float)
        weights = 1.0 / np.power(ranks, self.zipf_param)
        return weights / weights.sum()

    def _create_one_order(self, world_state, created_at: int):
        """Core order creation logic shared by generate() and generate_one()."""
        available_pods = world_state.pod_state.get_available_pods()
        if not available_pods:
            return None

        all_skus = set()
        for pod in available_pods:
            for sku, qty in pod.sku_inventory.items():
                if qty > 0:
                    all_skus.add(sku)
        all_skus_list = sorted(all_skus)
        if not all_skus_list:
            return None

        station_ids = list(world_state.map_state.station_positions.keys())
        if not station_ids:
            return None

        if self.fixed_order_size:
            num_sku_types = min(self.max_items_per_order, len(all_skus_list))
        else:
            num_sku_types = min(
                random.randint(1, self.max_items_per_order),
                len(all_skus_list),
            )

        weights = self._zipf_weights(len(all_skus_list))
        indices = np.random.choice(
            len(all_skus_list),
            size=num_sku_types,
            replace=False,
            p=weights,
        )
        chosen_skus = [all_skus_list[i] for i in indices]

        sku_demands: Dict[str, int] = {}
        for sku in chosen_skus:
            sku_demands[sku] = random.randint(1, self.max_items_per_sku)

        station_id = random.choice(station_ids)

        return Order(
            sku_demands=sku_demands,
            station_id=station_id,
            created_at=created_at,
        )

    def generate(self, world_state) -> List[Order]:
        """Generate orders with SKU selection biased by Zipf distribution."""
        if world_state.tick == 0 or world_state.tick % self.order_interval != 0:
            return []

        order = self._create_one_order(world_state, world_state.tick)
        return [order] if order else []

    def generate_one(self, world_state, created_at: int = 0):
        """Generate a single order unconditionally (ignores interval)."""
        return self._create_one_order(world_state, created_at)
