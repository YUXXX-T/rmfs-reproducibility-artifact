from .base_task_assigner import BaseTaskAssigner
from .GreedyTaskAssigner import GreedyTaskAssigner
from .HungarianTaskAssigner import HungarianTaskAssigner
from .WorldModelTaskAssigner import WorldModelTaskAssigner
from .JSQTaskAssigner import JSQTaskAssigner

from Policies.policy_registry import register
register("task_assigner", "GreedyTaskAssigner", GreedyTaskAssigner)
register("task_assigner", "HungarianTaskAssigner", HungarianTaskAssigner)
register("task_assigner", "WorldModelTaskAssigner", WorldModelTaskAssigner)
register("task_assigner", "JSQTaskAssigner", JSQTaskAssigner)

__all__ = ["BaseTaskAssigner", "GreedyTaskAssigner", "HungarianTaskAssigner",
           "WorldModelTaskAssigner", "JSQTaskAssigner"]
