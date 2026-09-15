"""
OrderState Module
=================
Represents customer orders in the fulfillment system.

订单状态模块
=================
表示履行系统中的客户订单。
"""

from enum import Enum, auto
from typing import List, Dict, Optional


class OrderStatus(Enum):
    """Lifecycle states of an order."""
    PENDING = auto()      # Waiting to be assigned
    IN_PROGRESS = auto()  # Tasks created, being fulfilled
    COMPLETED = auto()    # All items delivered


class Order:
    """
    A customer order requesting items (SKUs) to be delivered to a station.

    订单模型：记录 SKU 需求（需要哪些 SKU 各多少件），
    由 PodRetriever 策略将 SKU 需求转化为具体的 pod 列表。

    属性
    ----------
    order_id : int
        Unique order identifier.
    sku_demands : dict[str, int]
        SKU 需求映射，键为 SKU 标识符，值为所需数量。
        例如 {"A_SKU_0": 3, "B_SKU_2": 1}
    pod_ids : list[int]
        由 PodRetriever 填充的 pod ID 列表。
    station_id : int
        Target station where pods should be delivered.
    status : OrderStatus
        Current order lifecycle status.
    created_at : int
        Tick when the order was created.
    completed_at : int or None
        Tick when the order was completed.
    """

    _next_id: int = 0

    def __init__(self, sku_demands: Dict[str, int], station_id: int,
                 created_at: int = 0):
        self.order_id = Order._next_id
        Order._next_id += 1
        self.sku_demands: Dict[str, int] = sku_demands
        self.pod_ids: List[int] = []  # 由 PodRetriever 填充
        self.station_id: int = station_id
        self.status: OrderStatus = OrderStatus.PENDING
        self.created_at: int = created_at
        self.completed_at: Optional[int] = None
        # Track which pod_ids have been delivered
        self.delivered_pod_ids: List[int] = []

    @property
    def is_fully_delivered(self) -> bool:
        """Check if all pods have been delivered."""
        return len(self.pod_ids) > 0 and set(self.delivered_pod_ids) >= set(self.pod_ids)

    def mark_pod_delivered(self, pod_id: int):
        """Record that a pod has been delivered for this order."""
        if pod_id in self.pod_ids and pod_id not in self.delivered_pod_ids:
            self.delivered_pod_ids.append(pod_id)

    def __repr__(self) -> str:
        return (
            f"Order(id={self.order_id}, sku_demands={self.sku_demands}, "
            f"pods={self.pod_ids}, station={self.station_id}, "
            f"status={self.status.name})"
        )


class OrderState:
    """
    Container managing all orders in the system.

    属性
    ----------
    orders : dict[int, Order]
        All orders keyed by order_id.
    """

    def __init__(self):
        self.orders: dict[int, Order] = {}

    def add_order(self, order: Order):
        """Register a new order."""
        self.orders[order.order_id] = order

    def get_pending_orders(self) -> List[Order]:
        """Return all orders that are still pending assignment."""
        return [o for o in self.orders.values() if o.status == OrderStatus.PENDING]

    def get_in_progress_orders(self) -> List[Order]:
        """Return all orders currently in progress."""
        return [o for o in self.orders.values() if o.status == OrderStatus.IN_PROGRESS]

    def get_completed_orders(self) -> List[Order]:
        """Return all completed orders."""
        return [o for o in self.orders.values() if o.status == OrderStatus.COMPLETED]

    @property
    def total_orders(self) -> int:
        return len(self.orders)

    @property
    def total_completed(self) -> int:
        return len(self.get_completed_orders())

    def __repr__(self) -> str:
        return (
            f"OrderState(total={self.total_orders}, "
            f"pending={len(self.get_pending_orders())}, "
            f"in_progress={len(self.get_in_progress_orders())}, "
            f"completed={self.total_completed})"
        )
