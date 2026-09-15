"""
Base Pod Initializer
====================
Abstract base class (interface) for pod initialization policies.

Pod 初始化策略抽象基类
====================
定义 Pod 初始化策略的接口。
"""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from WorldState.world import WorldState


class BasePodInitializer(ABC):
    """
    Interface for pod initialization policies.

    Implementations decide how to assign pod_type and skus to each pod,
    ensuring that same-type pods are placed in the same pod_zone and
    same-type SKUs only appear on same-type pods.

    Pod 初始化策略接口。
    实现类负责为每个 Pod 分配 pod_type 和 skus，
    保证同类型 Pod 在同一个 pod_zone 内，
    同类型 SKU 只出现在同类型 Pod 上。
    """

    @abstractmethod
    def initialize_pods(self, world_state: "WorldState", *args, **kwargs) -> None:
        """
        根据 map_state 中的 pod_zones 和 config 中的参数，
        初始化所有 Pod 的 pod_type 和 sku_inventory 属性，
        并将 Pod 对象注册到 world_state.pod_state 中。

        该方法在仿真开始前调用一次。

        参数
        ----------
        world_state : WorldState
            当前仿真状态（包含 map_state, pod_state, config 等）。
        *args, **kwargs :
            额外参数，可由外部调用者传入以注入自定义属性到 Pod 中。
        """
        ...
