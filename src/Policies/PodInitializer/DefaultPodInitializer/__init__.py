"""
Default Pod Initializer
=======================
默认实现：按轮询方式将 pod_zone 分配给各 pod_type，
为每种 pod_type 生成专属 SKU 池，并按配置分配 SKU 库存。

Default implementation: assigns pod_zones to pod_types in round-robin,
generates a dedicated SKU pool per pod_type, and assigns SKU inventory per config.
"""

import random
from typing import List, Dict

from Policies.PodInitializer.base_pod_initializer import BasePodInitializer
from WorldState.map_state import CellType
from WorldState.pod_state import Pod


class DefaultPodInitializer(BasePodInitializer):
    """
    默认 Pod 初始化策略。

    - 将 pod_zones 按轮询方式均匀分配给各 pod_type
    - 为每种 pod_type 生成专属 SKU 池（大小由 sku_pool_size_per_type 控制）
    - 每个 Pod 从对应类型的 SKU 池中随机抽取 skus_per_pod 个 SKU
    - 每个 SKU 的初始数量由 items_per_sku 控制
    - 支持通过 **kwargs 注入自定义属性到 Pod 中
    """

    def initialize_pods(self, world_state, *args, **kwargs) -> None:
        """
        初始化所有 Pod 的 pod_type 和 sku_inventory，并注册到 world_state.pod_state。

        逻辑：
        1. 从 config.pods 读取 pod_types, skus_per_pod, sku_pool_size_per_type, items_per_sku
        2. 当 pods 配置为空（default 类型）时，skus_per_pod=1, sku_pool_size=1
        3. 将 map_state.pod_zones 按轮询分配 pod_type
        4. 为每个 zone 内的每个位置创建 Pod，设置 pod_type 和 sku_inventory
        5. 透传 **kwargs 到 Pod 构造函数以注入自定义属性
        """
        config = world_state.config
        pods_cfg = config.pods

        pod_types: List[str] = pods_cfg.pod_types
        skus_per_pod: int = pods_cfg.skus_per_pod
        sku_pool_size: int = pods_cfg.sku_pool_size_per_type
        items_per_sku: int = pods_cfg.items_per_sku

        # 需求1：当 pod_types 为空时使用 default 类型，
        # 且 skus_per_pod 和 sku_pool_size 使用最小默认值
        if not pod_types:
            pod_types = ["default"]
            skus_per_pod = 1
            sku_pool_size = 1

        # 为每种 pod_type 生成专属 SKU 池
        # 例如 type "A" -> ["A_SKU_0", "A_SKU_1", ..., "A_SKU_9"]
        sku_pools: Dict[str, List[str]] = {}
        for pt in pod_types:
            sku_pools[pt] = [f"{pt}_SKU_{i}" for i in range(sku_pool_size)]

        # 获取 pod_zones 配置（来自 map_state）
        pod_zones = config.map.pod_zones

        # 轮询分配 pod_type 给各 zone
        zone_types = []
        for idx, zone in enumerate(pod_zones):
            assigned_type = pod_types[idx % len(pod_types)]
            zone_types.append(assigned_type)

        # 遍历每个 zone，展开位置并创建 Pod
        pod_id_counter = 0
        for zone_idx, zone in enumerate(pod_zones):
            pod_type = zone_types[zone_idx]
            pool = sku_pools[pod_type]

            for r in range(zone.origin_row, zone.origin_row + zone.num_rows):
                for c in range(zone.origin_col, zone.origin_col + zone.num_cols):
                    if world_state.map_state.grid[r][c] != CellType.POD_HOME:
                        continue
                    # 从该类型的 SKU 池中随机抽取 skus_per_pod 个
                    actual_n = min(skus_per_pod, len(pool))
                    chosen_skus = random.sample(pool, actual_n)

                    # 构建 sku_inventory：每个 SKU 的初始数量为 items_per_sku
                    sku_inventory = {sku: items_per_sku for sku in chosen_skus}

                    pod = Pod(
                        pod_id=pod_id_counter,
                        home_position=(r, c),
                        pod_type=pod_type,
                        sku_inventory=sku_inventory,
                        **kwargs,
                    )
                    world_state.pod_state.add_pod(pod)
                    pod_id_counter += 1
