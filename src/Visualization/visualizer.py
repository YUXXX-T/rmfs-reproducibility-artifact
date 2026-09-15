"""
Visualizer Module
=================
Visualization backends for the MAS-RMFS simulation.

可视化模块
=================
MAS-RMFS 仿真的可视化后端。

Includes:
    - TerminalVisualizer  — ASCII grid to stdout
    - MatplotlibVisualizer — animated 2×2 dashboard
"""

from abc import ABC, abstractmethod
from typing import List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from WorldState.world import WorldState


class BaseVisualizer(ABC):
    """Interface for simulation visualization."""

    @abstractmethod
    def render(self, world_state: "WorldState"):
        """Render the current world state."""
        ...


class TerminalVisualizer(BaseVisualizer):
    """
    Simple text-based visualizer that prints a grid to stdout.
    一个简单的文本可视化工具，将网格输出至标准输出。
    
    Legend:
        .  = free cell
        #  = obstacle
        S  = station
        P  = pod (at home)
        R  = robot (idle)
        *  = robot carrying pod
    """

    def render(self, world_state):
        """Print a text grid of the current world state."""
        ms = world_state.map_state
        grid = [["." for _ in range(ms.cols)] for _ in range(ms.rows)]

        # Mark cell types
        for r in range(ms.rows):
            for c in range(ms.cols):
                from WorldState.map_state import CellType
                cell = ms.grid[r][c]
                if cell == CellType.OBSTACLE:
                    grid[r][c] = "#"
                elif cell == CellType.STATION:
                    grid[r][c] = "S"
                elif cell == CellType.POD_HOME:
                    grid[r][c] = "P"
                elif cell == CellType.STATION_QUEUE:
                    grid[r][c] = "Q"
                elif cell == CellType.STATION_BUFFER:
                    grid[r][c] = "B"
                elif cell == CellType.STATION_SERVICE:
                    grid[r][c] = "V"
                elif cell == CellType.STATION_EXIT:
                    grid[r][c] = "X"
                elif cell == CellType.STATION_ENTRY:
                    grid[r][c] = "E"

        # Mark pods not at home (being carried or displaced)
        for pod in world_state.pod_state.pods.values():
            if not pod.is_carried and not pod.is_at_home:
                pr, pc = pod.current_position
                grid[pr][pc] = "p"

        # Mark agents
        for agent in world_state.agents:
            r, c = agent.position
            if agent.carried_pod_id is not None:
                grid[r][c] = "*"
            else:
                grid[r][c] = "R"

        # Print
        header = f"=== Tick {world_state.tick} ==="
        print(header)
        print("  " + " ".join(str(c) for c in range(ms.cols)))
        for r in range(ms.rows):
            print(f"{r} " + " ".join(grid[r]))
        print()


# ---------------------------------------------------------------------------
# Matplotlib animated dashboard
# ---------------------------------------------------------------------------

class MatplotlibVisualizer(BaseVisualizer):
    """
    每个仿真 tick 更新的 2×2 matplotlib 动画仪表盘。

    Panels
    ------
    左上：实时仓库网格（障碍物、工作站、货架、机器人）。
    右上：累积路径密度热力图 (YlOrRd)。
    左下：智能体状态时间线（智能体 × tick 颜色编码）。
    右下：累积已完成订单吞吐量曲线。
    """

    # 状态 → 时间线颜色映射的整数编码
    _STATUS_CODES = {
        "IDLE": 0,
        "MOVING_TO_POD": 1,
        "CARRYING": 2,
        "DELIVERING": 3,
        "RETURNING": 4,
        "MOVING": 5,
        "QUEUING": 6,
        "EXITING": 7,
        "WAITING_ASSIGNED": 8,
    }

    def __init__(self, night_mode: bool = True):
        import matplotlib
        matplotlib.use("TkAgg")          # 确保交互式后端
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap

        self._plt = plt
        self._fig = None
        self._axes = None
        self._initialised = False
        self._night_mode = night_mode

        # 主题颜色
        if night_mode:
            self._bg = "#0f0f1a"
            self._ax_bg = "#16162a"
            self._tick_clr = "#aaaaaa"
            self._spine_clr = "#333355"
            self._title_clr = "#e0e0e0"
            self._grid_base = [0.09, 0.09, 0.16]
            self._grid_obs = [0.25, 0.25, 0.30]
            self._grid_pod_home = [0.10, 0.18, 0.20]
            self._grid_zone_queue = [0.55, 0.35, 0.12]
            self._grid_zone_buffer = [0.45, 0.25, 0.50]
            self._grid_zone_service = [0.70, 0.18, 0.22]
            self._grid_zone_exit = [0.20, 0.55, 0.35]
            self._grid_zone_entry = [0.18, 0.35, 0.60]
            self._grid_line_clr = "#333355"
            self._label_bg = "#16162a"
            self._idle_clr = "#1a1a2e"
        else:
            self._bg = "#f5f5f8"
            self._ax_bg = "#ffffff"
            self._tick_clr = "#333333"
            self._spine_clr = "#bbbbcc"
            self._title_clr = "#222222"
            self._grid_base = [0.92, 0.92, 0.95]
            self._grid_obs = [0.60, 0.60, 0.65]
            self._grid_pod_home = [0.80, 0.90, 0.92]
            self._grid_zone_queue = [0.85, 0.65, 0.30]
            self._grid_zone_buffer = [0.72, 0.45, 0.75]
            self._grid_zone_service = [0.90, 0.30, 0.35]
            self._grid_zone_exit = [0.30, 0.75, 0.45]
            self._grid_zone_entry = [0.30, 0.55, 0.85]
            self._grid_line_clr = "#bbbbcc"
            self._label_bg = "#ffffff"
            self._idle_clr = "#dddde8"

        # 跨 tick 累积存储的数据
        self._density: np.ndarray | None = None     # (rows, cols) float
        self._status_history: List[List[int]] = []   # tick → [status_code per agent]
        self._throughput: List[int] = []             # completed orders per tick

        # 不同机器人颜色（最多 10 个智能体）
        self._robot_cmap = plt.cm.get_cmap("tab10")

        # 时间线面板的自定义离散颜色映射
        # 顺序：IDLE, MOVING_TO_POD, CARRYING, DELIVERING, RETURNING, MOVING
        self._timeline_cmap = ListedColormap(
            [self._idle_clr,  # IDLE
             "#4361ee",       # MOVING_TO_POD – blue
             "#f0a500",       # CARRYING    – gold
             "#e07c24",       # DELIVERING  – orange
             "#7b2cbf",       # RETURNING   – purple
             "#2ec4b6",       # MOVING      – teal
             "#f59e0b",       # QUEUING     – amber
             "#6b7280",       # EXITING     – gray
             "#38bdf8"]       # WAITING_ASSIGNED – admission wait
        )
        self._timeline_labels = list(self._STATUS_CODES.keys())

    # ---- 公共 API --------------------------------------------------------

    def render(self, world_state: "WorldState"):
        """每个 tick 由仿真引擎调用一次。"""
        plt = self._plt

        if not self._initialised:
            self._setup(world_state)

        # 累积数据
        self._accumulate(world_state)

        # 重绘每个面板
        for ax in self._axes.flat:
            ax.clear()

        self._draw_grid(self._axes[0, 0], world_state)
        self._draw_density(self._axes[0, 1], world_state)
        self._draw_timeline(self._axes[1, 0], world_state)
        self._draw_throughput(self._axes[1, 1], world_state)

        self._fig.suptitle(
            f"MAS-RMFS  ·  Tick {world_state.tick}",
            fontsize=14, fontweight="bold", color=self._title_clr,
        )
        self._fig.canvas.draw_idle()
        self._fig.canvas.flush_events()
        plt.pause(0.05)

    # ---- initialisation ----------------------------------------------------

    def _setup(self, world_state):
        plt = self._plt
        plt.ion()
        self._fig, self._axes = plt.subplots(
            2, 2, figsize=(14, 10),
            facecolor=self._bg,
        )
        for ax in self._axes.flat:
            ax.set_facecolor(self._ax_bg)
            ax.tick_params(colors=self._tick_clr, labelsize=8)
            for spine in ax.spines.values():
                spine.set_color(self._spine_clr)

        rows = world_state.map_state.rows
        cols = world_state.map_state.cols
        self._density = np.zeros((rows, cols), dtype=float)
        self._initialised = True

    # ---- data accumulation -------------------------------------------------

    def _accumulate(self, world_state):
        # 路径密度：本 tick 每个机器人占据的格子 +1
        for agent in world_state.agents:
            r, c = agent.position
            self._density[r, c] += 1.0

        # 状态历史：每个 tick 一行
        codes = []
        for agent in world_state.agents:
            codes.append(self._STATUS_CODES.get(agent.status.name, 0))
        self._status_history.append(codes)

        # 吞吐量：累积已完成订单
        self._throughput.append(world_state.order_state.total_completed)

    # ---- panel drawers -----------------------------------------------------

    def _draw_grid(self, ax, world_state):
        """左上：实时仓库网格。"""
        from WorldState.map_state import CellType

        ms = world_state.map_state
        rows, cols = ms.rows, ms.cols

        # 背景单元格颜色矩阵
        grid = np.ones((rows, cols, 3)) * np.array(self._grid_base)

        for r in range(rows):
            for c in range(cols):
                cell = ms.grid[r][c]
                if cell == CellType.OBSTACLE:
                    grid[r, c] = self._grid_obs
                elif cell == CellType.POD_HOME:
                    grid[r, c] = self._grid_pod_home
                elif cell == CellType.STATION_QUEUE:
                    grid[r, c] = self._grid_zone_queue
                elif cell == CellType.STATION_BUFFER:
                    grid[r, c] = self._grid_zone_buffer
                elif cell == CellType.STATION_SERVICE:
                    grid[r, c] = self._grid_zone_service
                elif cell == CellType.STATION_EXIT:
                    grid[r, c] = self._grid_zone_exit
                elif cell == CellType.STATION_ENTRY:
                    grid[r, c] = self._grid_zone_entry

        ax.imshow(grid, origin="upper", aspect="equal")

        # 网格线
        for i in range(rows + 1):
            ax.axhline(i - 0.5, color=self._grid_line_clr, linewidth=0.5)
        for j in range(cols + 1):
            ax.axvline(j - 0.5, color=self._grid_line_clr, linewidth=0.5)

        # Stations (★)
        for sid, (sr, sc) in ms.station_positions.items():
            ax.plot(sc, sr, marker="*", markersize=18, color="#ff4757",
                    markeredgecolor="#ff6b81", markeredgewidth=0.8)
            ax.text(sc, sr - 0.38, f"S{sid}", ha="center", va="bottom",
                    fontsize=7, color="#ff6b81", fontweight="bold")

        # Pods at home (▲)
        for pod in world_state.pod_state.pods.values():
            if not pod.is_carried:
                pr, pc = pod.current_position
                ax.plot(pc, pr, marker="^", markersize=9, color="#2ec4b6",
                        markeredgecolor="#3dd6c8", markeredgewidth=0.6)

        # Robots (●) with ID labels
        for agent in world_state.agents:
            r, c = agent.position
            colour = self._robot_cmap(agent.agent_id % 10)

            if agent.carried_pod_id is not None:
                # Glow ring when carrying a pod
                ax.plot(c, r, "o", markersize=18, color=colour, alpha=0.25)
                ax.plot(c, r, "o", markersize=12, color=colour,
                        markeredgecolor="white", markeredgewidth=1.2)
            else:
                ax.plot(c, r, "o", markersize=12, color=colour,
                        markeredgecolor="white", markeredgewidth=0.8)

            ax.text(c + 0.32, r - 0.32, str(agent.agent_id),
                    fontsize=7, color="white", fontweight="bold")

        ax.set_xlim(-0.5, cols - 0.5)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_title("Warehouse Grid", color=self._title_clr, fontsize=11, pad=8)

    def _draw_density(self, ax, world_state):
        """右上：累积路径密度热力图。"""
        im = ax.imshow(
            self._density, cmap="YlOrRd", origin="upper", aspect="equal",
            interpolation="nearest",
        )
        rows, cols = self._density.shape
        ax.set_xticks(range(cols))
        ax.set_yticks(range(rows))
        ax.set_title("Path Density (cumulative)", color=self._title_clr,
                      fontsize=11, pad=8)

        # 轻量颜色条（首次调用时创建，更新值范围）
        if not hasattr(self, "_density_cb"):
            self._density_cb = self._fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            self._density_cb.ax.tick_params(colors=self._tick_clr, labelsize=7)
        else:
            self._density_cb.update_normal(im)

    def _draw_timeline(self, ax, world_state):
        """左下：智能体状态时间线（智能体 × tick）。"""
        if not self._status_history:
            return

        n_agents = len(world_state.agents)
        n_ticks = len(self._status_history)

        # 构建矩阵：行 = 智能体，列 = ticks
        mat = np.zeros((n_agents, n_ticks), dtype=int)
        for t, row in enumerate(self._status_history):
            for a, code in enumerate(row):
                mat[a, t] = code

        ax.imshow(
            mat, cmap=self._timeline_cmap, aspect="auto", origin="upper",
            vmin=0, vmax=len(self._STATUS_CODES) - 1,
            interpolation="nearest",
        )

        ax.set_yticks(range(n_agents))
        ax.set_yticklabels([f"R{i}" for i in range(n_agents)])
        ax.set_xlabel("Tick", color=self._tick_clr, fontsize=9)
        ax.set_title("Agent Status Timeline", color=self._title_clr,
                      fontsize=11, pad=8)

        # 图例（状态标签）
        if not hasattr(self, "_timeline_legend_drawn"):
            from matplotlib.patches import Patch
            patches = [
                Patch(facecolor=self._timeline_cmap.colors[i], label=lbl)
                for i, lbl in enumerate(self._timeline_labels)
            ]
            ax.legend(
                handles=patches, loc="lower left", fontsize=6,
                ncol=3, framealpha=0.6, facecolor=self._label_bg,
                edgecolor=self._spine_clr, labelcolor=self._tick_clr,
            )
            self._timeline_legend_drawn = True

    def _draw_throughput(self, ax, world_state):
        """右下：累积已完成订单。"""
        ticks = list(range(len(self._throughput)))

        ax.fill_between(ticks, self._throughput, alpha=0.15, color="#2ecc71")
        ax.plot(ticks, self._throughput, color="#2ecc71", linewidth=2)

        # 数字标注
        current = self._throughput[-1] if self._throughput else 0
        rate = current / max(world_state.tick, 1)
        ax.text(
            0.98, 0.92,
            f"Completed: {current}\nRate: {rate:.2f} orders/tick",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=9, color="#2ecc71", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=self._ax_bg,
                      edgecolor="#2ecc71", alpha=0.8),
        )

        ax.set_xlabel("Tick", color=self._tick_clr, fontsize=9)
        ax.set_ylabel("Completed Orders", color=self._tick_clr, fontsize=9)
        ax.set_title("Throughput", color=self._title_clr, fontsize=11, pad=8)
