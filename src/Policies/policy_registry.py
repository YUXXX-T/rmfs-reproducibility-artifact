"""
策略注册中心
===============
将算法名称（字符串）映射到其类的集中注册中心。

每个算法在其子包被导入时自动注册。
仿真从 JSON 配置中读取所需的算法名称，
并使用此注册中心实例化正确的类 — 无需
修改源代码即可切换算法。

Usage
-----
注册（在每个算法的 __init__.py 中完成）：：

    from Policies.policy_registry import register
    register("order_generator", "RandomOrderGenerator", RandomOrderGenerator)

查找（在 main.py 中完成）：：

    from Policies.policy_registry import get_policy
    cls = get_policy("order_generator", "RandomOrderGenerator")
    instance = cls(order_interval=5)
"""

from typing import Dict, Type

# 类别 -> {名称 -> 类}
_registry: Dict[str, Dict[str, Type]] = {}


def register(category: str, name: str, cls: Type) -> None:
    """
    在给定的类别和名称下注册策略类。

    参数
    ----------
    category : str
        "order_generator", "task_assigner", "path_planner" 之一。
    name : str
        可读的算法名称（如 "AStarPathPlanner"）。
    cls : type
        实现该算法的具体类。
    """
    _registry.setdefault(category, {})[name] = cls


def get_policy(category: str, name: str) -> Type:
    """
    按类别和名称查找已注册的策略类。

    参数
    ----------
    category : str
        策略类别。
    name : str
        已注册的算法名称。

    返回值
    -------
    type
        策略类。

    异常
    ------
    ValueError
        如果未找到类别或名称，附带有用的错误信息
        列出可用选项。
    """
    if category not in _registry:
        available = ", ".join(sorted(_registry.keys())) or "(none)"
        raise ValueError(
            f"Unknown policy category '{category}'. "
            f"Available categories: {available}"
        )

    policies = _registry[category]
    if name not in policies:
        available = ", ".join(sorted(policies.keys())) or "(none)"
        raise ValueError(
            f"Unknown {category} algorithm '{name}'. "
            f"Available options: {available}"
        )

    return policies[name]


def list_policies(category: str = None) -> Dict[str, list]:
    """
    列出已注册的策略，可按类别过滤。

    返回值
    -------
    dict[str, list[str]]
        类别 -> 已注册算法名称列表的映射。
    """
    if category:
        return {category: sorted(_registry.get(category, {}).keys())}
    return {cat: sorted(names.keys()) for cat, names in _registry.items()}
