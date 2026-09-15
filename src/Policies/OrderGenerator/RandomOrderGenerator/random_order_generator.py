"""
Random Order Generator
======================
Default implementation: generates random orders with SKU demands
at a configurable interval.

随机订单生成器：按配置间隔生成随机 SKU 需求的订单。
"""

import random
from typing import List, Dict

from Policies.OrderGenerator.base_order_generator import BaseOrderGenerator
from WorldState.order_state import Order


class RandomOrderGenerator(BaseOrderGenerator):
    """
    Generates random orders at regular intervals.

    Each order picks 1..max_items random SKUs from available pods
    and assigns random demand quantities.

    参数
    ----------
    order_interval : int
        Generate a new order every N ticks.
    max_items_per_order : int
        Maximum number of distinct SKU types per order.
    max_items_per_sku : int
        每种 SKU 需求的物品数量上限。
    """

    def __init__(self, order_interval: int = 5, max_items_per_order: int = 2,
                 fixed_order_size: bool = False, max_items_per_sku: int = 5):
        self.order_interval = order_interval
        self.max_items_per_order = max_items_per_order
        self.fixed_order_size = fixed_order_size
        self.max_items_per_sku = max_items_per_sku

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
        chosen_skus = random.sample(all_skus_list, num_sku_types)

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
        """Generate random orders every `order_interval` ticks."""
        if world_state.tick == 0 or world_state.tick % self.order_interval != 0:
            return []

        order = self._create_one_order(world_state, world_state.tick)
        return [order] if order else []

    def generate_one(self, world_state, created_at: int = 0):
        """Generate a single order unconditionally (ignores interval)."""
        return self._create_one_order(world_state, created_at)
