"""
Default Pod Retriever
=====================
默认实现：贪婪地匹配订单的 SKU 需求，选择能满足最多需求的 Pod。

Default implementation: greedily matches an order's SKU demands
by selecting pods that cover the most unfulfilled demands.
"""

from typing import List, Dict

from Policies.PodRetriever.base_pod_retriever import BasePodRetriever
from WorldState.task_state import TaskStatus


class DefaultPodRetriever(BasePodRetriever):
    """
    默认 Pod 检索策略。

    贪婪算法：反复从可用 Pod 中选出能覆盖最多未满足 SKU 需求的 Pod，
    直到所有需求被满足或没有更多可用 Pod。

    Greedy algorithm: repeatedly pick the available pod that covers
    the most unfulfilled SKU demands until all demands are met
    or no more pods are available.
    """

    def retrieve(self, order, world_state) -> List[int]:
        """
        贪婪匹配 SKU 需求到 Pod 列表。

        Greedily match SKU demands to a list of pod IDs.
        """
        # 剩余未满足的 SKU 需求（副本）
        remaining: Dict[str, int] = dict(order.sku_demands)
        selected_pod_ids: List[int] = []

        # 获取所有可用 Pod（在原位且未被搬运），排除已被任务预定的 Pod
        reserved_pod_ids = {
            t.pod_id
            for t in world_state.task_state.tasks.values()
            if t.status in (TaskStatus.ASSIGNED, TaskStatus.IN_PROGRESS)
        }
        delivered_pod_ids = {
            int(pod_id)
            for pod_id in getattr(order, "delivered_pod_ids", ())
        }
        available_pods = [
            p for p in world_state.pod_state.get_available_pods()
            if p.pod_id not in reserved_pod_ids
            and int(p.pod_id) not in delivered_pod_ids
        ]

        while any(v > 0 for v in remaining.values()) and available_pods:
            best_pod = None
            best_score = 0  # 该 Pod 能覆盖的 SKU 种类数

            for pod in available_pods:
                if pod.pod_id in selected_pod_ids:
                    continue
                # 计算该 pod 能覆盖多少未满足的 SKU 需求
                score = 0
                for sku, demand in remaining.items():
                    if demand > 0 and sku in pod.sku_inventory and pod.sku_inventory[sku] > 0:
                        score += min(demand, pod.sku_inventory[sku])
                if score > best_score:
                    best_score = score
                    best_pod = pod

            if best_pod is None:
                break  # 没有 Pod 能满足任何剩余需求

            selected_pod_ids.append(best_pod.pod_id)

            # 扣减剩余需求（注意不修改 pod 实际库存，仅用于匹配计算）
            for sku in list(remaining.keys()):
                if remaining[sku] > 0 and sku in best_pod.sku_inventory:
                    can_provide = best_pod.sku_inventory[sku]
                    fulfilled = min(remaining[sku], can_provide)
                    remaining[sku] -= fulfilled

            # 移除已选中的 Pod，避免重复选取
            available_pods = [p for p in available_pods if p.pod_id != best_pod.pod_id]

        return selected_pod_ids
