"""
Recorded Order Generator
========================
从预先录制的 JSON 文件中回放订单。

两种模式：
- 默认：按录制时的 tick 回放订单。
- immediate_dispatch=True：忽略 tick，在首个 tick 一次性投放全部订单。
"""

import json
from typing import List, Dict
from collections import defaultdict

from Policies.OrderGenerator.base_order_generator import BaseOrderGenerator
from WorldState.order_state import Order


class RecordedOrderGenerator(BaseOrderGenerator):
    """
    Replays pre-recorded orders from a JSON file.

    参数
    ----------
    recorded_orders_path : str
        Path to the JSON file produced by OrderGenerateRecord.recorder.
    immediate_dispatch : bool
        If True, all orders are emitted at tick 1 (ignoring recorded ticks).
    """

    def __init__(self, recorded_orders_path: str = "",
                 immediate_dispatch: bool = False, **_kwargs):
        self._immediate = immediate_dispatch
        self._orders_by_tick: Dict[int, list] = defaultdict(list)
        self._all_orders: list = []
        self._dispatched: bool = False

        if not recorded_orders_path:
            return

        with open(recorded_orders_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for entry in data.get("orders", []):
            if self._immediate:
                self._all_orders.append(entry)
            else:
                self._orders_by_tick[entry["tick"]].append(entry)

    def generate(self, world_state) -> List[Order]:
        if self._immediate:
            if self._dispatched or world_state.tick < 1:
                return []
            self._dispatched = True
            entries = self._all_orders
        else:
            entries = self._orders_by_tick.get(world_state.tick, [])

        orders = []
        for entry in entries:
            order = Order(
                sku_demands=entry["sku_demands"],
                station_id=entry["station_id"],
                created_at=world_state.tick,
            )
            orders.append(order)
        return orders
