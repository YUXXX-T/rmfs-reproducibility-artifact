"""
Config Loader Module
====================
Loads simulation configuration from a JSON file and provides it as typed dataclasses.

配置加载模块
====================
从 JSON 文件加载仿真配置，并将其提供为类型化的数据类。

"""

import json
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any


@dataclass
class StationQueueConfig:
    """Configuration for a station's queue zone layout."""
    service: Optional[Tuple[int, int]] = None
    queue: Optional[List[Tuple[int, int]]] = None
    buffer: Optional[List[Tuple[int, int]]] = None
    entry: Optional[Tuple[int, int]] = None
    exit: Optional[Tuple[int, int]] = None
    queue_length: int = 3
    buffer_length: int = 1
    direction: str = "auto"


@dataclass
class StationConfig:
    """Configuration for a single workstation."""
    id: int
    row: int
    col: int
    queue: Optional[StationQueueConfig] = None


@dataclass
class PodZoneConfig:
    """Configuration for a rectangular block of pod storage positions."""
    origin_row: int
    origin_col: int
    num_rows: int
    num_cols: int


@dataclass
class PodLayoutConfig:
    """Compact configuration that auto-generates pod_zones."""
    num_rows: int = 10
    num_cols: int = 2
    row_step: int = 12
    col_step: int = 6
    margin: int = 3
    max_pods: int = 0


@dataclass
class MapConfig:
    """Configuration for the warehouse map/grid."""
    rows: int
    cols: int
    obstacles: List[Tuple[int, int]]
    stations: List[StationConfig]
    pod_zones: List[PodZoneConfig]


@dataclass
class RobotConfig:
    """Configuration for robot agents."""
    num_robots: int
    starts: List[Tuple[int, int]]
    speed: int = 1
    random_starts: bool = False    # True = 忽略 starts，运行时从 free cell 中按全局 seed 采样


@dataclass
class PodsConfig:
    """Pod 初始化相关配置。"""
    pod_types: List[str] = field(default_factory=lambda: ["A", "B", "C"])
    skus_per_pod: int = 3
    sku_pool_size_per_type: int = 10
    items_per_sku: int = 20  # 每个 Pod 初始化时每种 SKU 的物品数量


@dataclass
class PolicyConfig:
    """
    Configuration for which algorithm to use for each policy.

    Each field stores a (name, params) tuple.  In JSON the value can be
    either a plain string (name only, no extra params) or an object
    ``{"name": "...", "params": {...}}``.

    用于指定各策略所使用算法的配置。 

    每个字段均存储一个 (name, params) 元组。在 JSON 格式中，其取值可以是
    一个纯字符串（仅包含名称，无额外参数），也可以是一个对象
    ``{"name": "...", "params": {...}}``。
    """
    order_generator: Tuple[str, Dict[str, Any]] = ("RandomOrderGenerator", {})
    task_assigner: Tuple[str, Dict[str, Any]] = ("GreedyTaskAssigner", {})
    path_planner: Tuple[str, Dict[str, Any]] = ("AStarPathPlanner", {})
    pod_return_planner: Tuple[str, Dict[str, Any]] = ("HomeReturnPlanner", {})
    pod_initializer: Tuple[str, Dict[str, Any]] = ("DefaultPodInitializer", {})
    pod_retriever: Tuple[str, Dict[str, Any]] = ("DefaultPodRetriever", {})


@dataclass
class BenchmarkConfig:
    """MAPF benchmark 模式配置。"""
    enabled: bool = False
    map_path: str = ""           # MovingAI .map 文件路径
    scen_path: str = ""          # .scen 文件路径（空则随机生成 start/goal）
    num_agents: int = 10         # agent 数量（scen 模式下取前 N 个）
    max_ticks: int = 500         # 超时 tick 数
    random_seed: int = 42        # 随机生成 start/goal 的种子


@dataclass
class SnapshotConfig:
    """训练数据快照采集配置。"""
    enabled: bool = False
    output_dir: str = "DataGen/snapshots"
    episode_id: str = "{map}_{planner}_{assigner}_r{robots}"


@dataclass
class RobotModelConfig:
    """3D 机器人模型参数。"""
    use_model: bool = True
    body_offset_z: float = 0.008
    body_hpr: Tuple[float, float, float] = (90.0, 90.0, 90.0)
    wheel_radius: float = 0.08
    wheel_positions: List[Tuple[float, float, float]] = field(
        default_factory=lambda: [
            (0.18, 0.14, 0.08),
            (0.18, -0.14, 0.08),
            (-0.18, 0.14, 0.08),
            (-0.18, -0.14, 0.08),
        ]
    )


@dataclass
class SimulationParams:
    """仿真级参数。"""
    order_interval: int = 5           # Generate new order every N ticks
    max_items_per_order: int = 2      # Max pods per order
    pickup_duration: int = 2          # Ticks to pause when picking up a pod
    dropoff_duration: int = 2         # Ticks to pause when dropping off a pod
    station_process_duration: int = 5 # Ticks to process at workstation
    tick_delay: float = 0.0           # Seconds to sleep between ticks (0 = no delay)
    p3d_view_mode: str = "2d"         # Panda3D camera: "2d" (ortho) or "3d" (perspective)
    p3d_use_gpu: bool = False         # Enable GPU batching/instancing in Panda3D
    night_mode: bool = True           # True = dark theme, False = light/white theme
    log_level: str = "INFO"
    log_file: Optional[str] = None
    fixed_order_size: bool = False    # True = 每个订单固定 max_items_per_order 个 pod
    task_execution_mode: str = "parallel"  # "parallel" 或 "serial"
    max_items_per_sku: int = 5        # 订单中每种 SKU 需求的物品数量上限
    robot_label_scale: float = 0.25   # Panda3D 机器人编号标签大小（3D 默认值；2D 自动取 88%）
    show_selection_panel: bool = True  # 是否显示选中信息面板
    show_robot_paths_panel: bool = True  # 是否显示机器人路径面板
    use_recorded_orders: bool = False  # True = 从文件读取预录制订单
    recorded_orders_path: str = ""    # 预录制订单 JSON 文件路径
    immediate_dispatch: bool = False  # True = 忽略录制 tick，立即投放所有订单（强制 serial 模式）
    seed: Optional[int] = None        # 全局随机种子；None = 不 seed（每次跑结果不同）
    max_ticks: int = 0                # 引擎自动停机的 tick 上限；0 = 不限（仅响应 Ctrl+C / 外部信号）
    initial_order_pool_size: int = 0  # tick 0 前预填充的订单数；0 = 不预填
    backlog_floor: int = 0            # pre-assignment 时 pending 不足该值则补单；0 = 不补
    backlog_refill_mode: str = "none" # none | on_pre_assignment | on_completed
    warmup_ticks: int = 0             # 预热期（不采集数据）tick 数
    evaluation_ticks: int = 0         # 评估期 tick 数（0 = 使用全部）


@dataclass
class SimulationConfig:
    """Top-level simulation configuration aggregating all sub-configs."""
    map: MapConfig = field(default_factory=lambda: MapConfig(10, 10, [], [], []))
    robots: RobotConfig = field(default_factory=lambda: RobotConfig(1, [[0, 0]]))
    simulation: SimulationParams = field(default_factory=SimulationParams)
    policies: PolicyConfig = field(default_factory=PolicyConfig)
    pods: PodsConfig = field(default_factory=PodsConfig)
    robot_model: RobotModelConfig = field(default_factory=RobotModelConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    snapshot: SnapshotConfig = field(default_factory=SnapshotConfig)


def load_config(path: str) -> SimulationConfig:
    """
    Load a SimulationConfig from a JSON file.

    参数
    ----------
    path : str
        JSON 配置文件路径。

    返回值
    -------
    SimulationConfig
        Parsed and validated configuration.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # --- Parse map ---
    map_raw = raw.get("map", {})
    obstacles = [tuple(o) for o in map_raw.get("obstacles", [])]

    # Parse top-level station_queue defaults
    sq_defaults_raw = raw.get("station_queue", {})
    sq_defaults = StationQueueConfig(
        queue_length=sq_defaults_raw.get("queue_length", 3),
        buffer_length=sq_defaults_raw.get("buffer_length", 1),
        direction=sq_defaults_raw.get("direction", "auto"),
    )

    stations = []
    for s in map_raw.get("stations", []):
        sq_raw = s.get("queue", None)
        if sq_raw is not None:
            sq = StationQueueConfig(
                service=tuple(sq_raw["service"]) if "service" in sq_raw else None,
                queue=[tuple(q) for q in sq_raw["queue"]] if "queue" in sq_raw else None,
                buffer=[tuple(b) for b in sq_raw["buffer"]] if "buffer" in sq_raw else None,
                entry=tuple(sq_raw["entry"]) if "entry" in sq_raw else None,
                exit=tuple(sq_raw["exit"]) if "exit" in sq_raw else None,
                queue_length=sq_raw.get("queue_length", sq_defaults.queue_length),
                buffer_length=sq_raw.get("buffer_length", sq_defaults.buffer_length),
                direction=sq_raw.get("direction", sq_defaults.direction),
            )
        else:
            sq = StationQueueConfig(
                queue_length=sq_defaults.queue_length,
                buffer_length=sq_defaults.buffer_length,
                direction=sq_defaults.direction,
            )
        stations.append(StationConfig(
            id=s["id"], row=s["row"], col=s["col"], queue=sq,
        ))

    map_rows = map_raw.get("rows", 10)
    map_cols = map_raw.get("cols", 10)

    if "pod_layout" in map_raw:
        pl = map_raw["pod_layout"]
        layout = PodLayoutConfig(
            num_rows=pl.get("num_rows", 10),
            num_cols=pl.get("num_cols", 2),
            row_step=pl.get("row_step", 12),
            col_step=pl.get("col_step", 6),
            margin=pl.get("margin", 3),
            max_pods=pl.get("max_pods", 0),
        )
        pod_zones = []
        pods_per_zone = layout.num_rows * layout.num_cols
        total_pods = 0
        for r in range(layout.margin, map_rows - layout.num_rows + 1, layout.row_step):
            for c in range(layout.margin, map_cols - layout.num_cols + 1, layout.col_step):
                pod_zones.append(PodZoneConfig(
                    origin_row=r, origin_col=c,
                    num_rows=layout.num_rows, num_cols=layout.num_cols,
                ))
                total_pods += pods_per_zone
                if layout.max_pods > 0 and total_pods >= layout.max_pods:
                    break
            if layout.max_pods > 0 and total_pods >= layout.max_pods:
                break
    else:
        pod_zones = [
            PodZoneConfig(
                origin_row=pz["origin_row"],
                origin_col=pz["origin_col"],
                num_rows=pz["num_rows"],
                num_cols=pz["num_cols"],
            )
            for pz in map_raw.get("pod_zones", [])
        ]

    map_config = MapConfig(
        rows=map_rows,
        cols=map_cols,
        obstacles=obstacles,
        stations=stations,
        pod_zones=pod_zones,
    )

    # --- Parse robots ---
    robot_raw = raw.get("robots", {})
    starts = [tuple(s) for s in robot_raw.get("starts", [[0, 0]])]
    robot_config = RobotConfig(
        num_robots=robot_raw.get("num_robots", len(starts)),
        starts=starts,
        speed=robot_raw.get("speed", 1),
        random_starts=robot_raw.get("random_starts", False),
    )

    # --- Parse simulation params ---
    sim_raw = raw.get("simulation", {})
    sim_params = SimulationParams(
        order_interval=sim_raw.get("order_interval", 5),
        max_items_per_order=sim_raw.get("max_items_per_order", 2),
        pickup_duration=sim_raw.get("pickup_duration", 2),
        dropoff_duration=sim_raw.get("dropoff_duration", 2),
        station_process_duration=sim_raw.get("station_process_duration", 5),
        tick_delay=sim_raw.get("tick_delay", 0.0),
        p3d_view_mode=sim_raw.get("p3d_view_mode", "2d"),
        p3d_use_gpu=sim_raw.get("p3d_use_gpu", False),
        night_mode=sim_raw.get("night_mode", True),
        log_level=sim_raw.get("log_level", "INFO"),
        log_file=sim_raw.get("log_file", None),
        fixed_order_size=sim_raw.get("fixed_order_size", False),
        task_execution_mode=sim_raw.get("task_execution_mode", "parallel"),
        max_items_per_sku=sim_raw.get("max_items_per_sku", 5),
        robot_label_scale=sim_raw.get("robot_label_scale", 0.25),
        show_selection_panel=sim_raw.get("show_selection_panel", True),
        show_robot_paths_panel=sim_raw.get("show_robot_paths_panel", True),
        use_recorded_orders=sim_raw.get("use_recorded_orders", False),
        recorded_orders_path=sim_raw.get("recorded_orders_path", ""),
        immediate_dispatch=sim_raw.get("immediate_dispatch", False),
        seed=sim_raw.get("seed", None),
        max_ticks=sim_raw.get("max_ticks", 0),
        initial_order_pool_size=sim_raw.get("initial_order_pool_size", 0),
        backlog_floor=sim_raw.get("backlog_floor", 0),
        backlog_refill_mode=sim_raw.get("backlog_refill_mode", "none"),
        warmup_ticks=sim_raw.get("warmup_ticks", 0),
        evaluation_ticks=sim_raw.get("evaluation_ticks", 0),
    )

    # --- Parse policies ---
    pol_raw = raw.get("policies", {})

    def _parse_policy_entry(val, default_name: str):
        """Accept either a string or {"name": ..., "params": {...}}."""
        if val is None:
            return (default_name, {})
        if isinstance(val, str):
            return (val, {})
        return (val.get("name", default_name), val.get("params", {}))

    policy_config = PolicyConfig(
        order_generator=_parse_policy_entry(
            pol_raw.get("order_generator"), "RandomOrderGenerator"),
        task_assigner=_parse_policy_entry(
            pol_raw.get("task_assigner"), "GreedyTaskAssigner"),
        path_planner=_parse_policy_entry(
            pol_raw.get("path_planner"), "AStarPathPlanner"),
        pod_return_planner=_parse_policy_entry(
            pol_raw.get("pod_return_planner"), "HomeReturnPlanner"),
        pod_initializer=_parse_policy_entry(
            pol_raw.get("pod_initializer"), "DefaultPodInitializer"),
        pod_retriever=_parse_policy_entry(
            pol_raw.get("pod_retriever"), "DefaultPodRetriever"),
    )

    # --- Parse pods config ---
    pods_raw = raw.get("pods", {})
    pods_config = PodsConfig(
        pod_types=pods_raw.get("pod_types", ["A", "B", "C"]),
        skus_per_pod=pods_raw.get("skus_per_pod", 3),
        sku_pool_size_per_type=pods_raw.get("sku_pool_size_per_type", 10),
        items_per_sku=pods_raw.get("items_per_sku", 20),
    )

    # --- Parse robot model config ---
    rm_raw = raw.get("robot_model", {})
    _default_wp = RobotModelConfig().wheel_positions
    robot_model_config = RobotModelConfig(
        use_model=rm_raw.get("use_model", True),
        body_offset_z=rm_raw.get("body_offset_z", 0.008),
        body_hpr=tuple(rm_raw.get("body_hpr", [90, 90, 90])),
        wheel_radius=rm_raw.get("wheel_radius", 0.08),
        wheel_positions=[tuple(p) for p in rm_raw.get("wheel_positions", _default_wp)],
    )

    # --- Parse benchmark config ---
    bm_raw = raw.get("benchmark", {})
    benchmark_config = BenchmarkConfig(
        enabled=bm_raw.get("enabled", False),
        map_path=bm_raw.get("map_path", ""),
        scen_path=bm_raw.get("scen_path", ""),
        num_agents=bm_raw.get("num_agents", 10),
        max_ticks=bm_raw.get("max_ticks", 500),
        random_seed=bm_raw.get("random_seed", 42),
    )

    # --- Parse snapshot config ---
    snap_raw = raw.get("snapshot", {})
    snapshot_config = SnapshotConfig(
        enabled=snap_raw.get("enabled", False),
        output_dir=snap_raw.get("output_dir", "DataGen/snapshots"),
        episode_id=snap_raw.get("episode_id", "{map}_{planner}_{assigner}_r{robots}"),
    )

    return SimulationConfig(
        map=map_config, robots=robot_config,
        simulation=sim_params, policies=policy_config,
        pods=pods_config, robot_model=robot_model_config,
        benchmark=benchmark_config, snapshot=snapshot_config,
    )
