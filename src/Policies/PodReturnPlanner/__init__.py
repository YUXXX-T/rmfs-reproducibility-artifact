from .base_pod_return_planner import BasePodReturnPlanner
from .HomeReturnPlanner import HomeReturnPlanner
from .NearestSlotPlanner import NearestSlotPlanner

from Policies.policy_registry import register
register("pod_return_planner", "HomeReturnPlanner", HomeReturnPlanner)
register("pod_return_planner", "NearestSlotPlanner", NearestSlotPlanner)

__all__ = ["BasePodReturnPlanner", "HomeReturnPlanner", "NearestSlotPlanner"]
