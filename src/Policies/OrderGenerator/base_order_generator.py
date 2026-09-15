"""
Base Order Generator
====================
Abstract base class (interface) for order generation policies.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from WorldState.world import WorldState
    from WorldState.order_state import Order


class BaseOrderGenerator(ABC):
    """
    Interface for order generation policies.

    Implementations decide when and how to create new customer orders.
    """

    @abstractmethod
    def generate(self, world_state: "WorldState") -> List["Order"]:
        """
        Generate new orders based on the current world state.

        参数
        ----------
        world_state : WorldState
            Current simulation state including tick counter.

        返回值
        -------
        list[Order]
            Newly generated orders (may be empty).
        """
        ...

    def generate_one(self, world_state: "WorldState", created_at: int = 0) -> Optional["Order"]:
        """Generate a single order unconditionally (ignores interval).

        Used by backlog refill and initial order pool seeding.
        Subclasses should override for efficiency; default calls generate()
        with a spoofed tick to force production.
        """
        return None
