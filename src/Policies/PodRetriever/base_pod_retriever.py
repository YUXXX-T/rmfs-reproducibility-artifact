"""
Base Pod Retriever
==================
Abstract base class (interface) for pod retrieval policies.

Pod 检索策略抽象基类
==================
定义将订单 SKU 需求映射为具体 Pod 列表的策略接口。
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from WorldState.order_state import Order
    from WorldState.world import WorldState


class BasePodRetriever(ABC):
    """
    Interface for pod retrieval policies.

    Implementations decide how to map an order's SKU demands
    to specific pod IDs that should be fetched.

    Pod 检索策略接口。
    实现类负责将订单的 SKU 需求映射为需要取回的 Pod 列表。
    """

    @abstractmethod
    def retrieve(self, order: "Order", world_state: "WorldState") -> List[int]:
        """
        根据订单的 SKU 需求，从可用 Pod 中选取能满足需求的 Pod 列表。

        Map an order's SKU demands to a list of pod IDs to retrieve.

        参数
        ----------
        order : Order
            带有 sku_demands 的订单。
        world_state : WorldState
            当前仿真状态（包含 pod_state 等）。

        返回值
        -------
        list[int]
            应取回的 pod_id 列表。
        """
        ...
