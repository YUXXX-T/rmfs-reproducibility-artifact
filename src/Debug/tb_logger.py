"""
TensorBoard 日志器
====================
封装 torch.utils.tensorboard.SummaryWriter 的 RL 训练指标日志器。

TensorBoard Logger
==================
Lightweight wrapper around SummaryWriter for logging RL training metrics.

Usage
-----
    from Debug.tb_logger import TBLogger

    tb = TBLogger(log_dir="runs/experiment_1")
    tb.log_episode(episode=0, metrics={"reward/mean": 12.5, "episode/length": 200})
    tb.log_scalars("reward/per_agent", {"agent_0": 10.0, "agent_1": 15.0}, step=0)
    tb.close()
"""

import os
from typing import Dict, Optional, Union, Sequence

import numpy as np
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


class TBLogger:
    """
    TensorBoard 日志器，用于记录 RL 训练指标。

    参数
    ----------
    log_dir : str
        TensorBoard 事件文件存储目录。默认 ``runs/rmfs``。
    comment : str
        附加到 log_dir 后缀的注释字符串（方便区分实验）。
    flush_secs : int
        自动 flush 间隔（秒），默认 30。
    """

    def __init__(
        self,
        log_dir: str = "runs/rmfs",
        comment: str = "",
        flush_secs: int = 30,
    ):
        self._log_dir = log_dir
        self._writer = SummaryWriter(
            log_dir=log_dir,
            comment=comment,
            flush_secs=flush_secs,
        )

    # ── Episode-level logging ────────────────────────────────────

    def log_episode(self, episode: int, metrics: Dict[str, float]) -> None:
        """
        记录一整个 episode 的指标。

        参数
        ----------
        episode : int
            当前 episode 编号（作为 global_step）。
        metrics : dict[str, float]
            标量指标字典，如::

                {
                    "reward/mean": 12.5,
                    "reward/min": -1.0,
                    "reward/max": 25.0,
                    "throughput/completed_orders": 8,
                    "episode/length": 200,
                }
        """
        for tag, value in metrics.items():
            self._writer.add_scalar(tag, value, global_step=episode)

    # ── Per-step logging ─────────────────────────────────────────

    def log_step(self, global_step: int, metrics: Dict[str, float]) -> None:
        """
        记录单步指标（用于更细粒度的训练过程追踪）。

        参数
        ----------
        global_step : int
            全局步数计数器。
        metrics : dict[str, float]
            标量指标字典。
        """
        for tag, value in metrics.items():
            self._writer.add_scalar(tag, value, global_step=global_step)

    # ── Single scalar ────────────────────────────────────────────

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """
        记录单个标量值。

        参数
        ----------
        tag : str
            指标名称，如 ``"loss/policy"``。
        value : float
            指标值。
        step : int
            步数。
        """
        self._writer.add_scalar(tag, value, global_step=step)

    # ── Grouped scalars (e.g. per-agent) ─────────────────────────

    def log_scalars(
        self, main_tag: str, tag_dict: Dict[str, float], step: int
    ) -> None:
        """
        将多个标量以同一组名记录（同一图表的多条曲线）。

        参数
        ----------
        main_tag : str
            组名，如 ``"reward/per_agent"``。
        tag_dict : dict[str, float]
            子标签 → 值，如 ``{"agent_0": 10.0, "agent_1": 15.0}``。
        step : int
            步数。
        """
        self._writer.add_scalars(main_tag, tag_dict, global_step=step)

    # ── Histogram ────────────────────────────────────────────────

    def log_histogram(
        self,
        tag: str,
        values: Union[np.ndarray, Sequence[float]],
        step: int,
    ) -> None:
        """
        记录直方图（如动作分布、Q 值分布）。

        参数
        ----------
        tag : str
            指标名称，如 ``"actions/distribution"``。
        values : array-like
            数据数组。
        step : int
            步数。
        """
        if not isinstance(values, np.ndarray):
            values = np.array(values, dtype=np.float32)
        self._writer.add_histogram(tag, values, global_step=step)

    # ── Text ─────────────────────────────────────────────────────

    def log_text(self, tag: str, text: str, step: int) -> None:
        """
        记录文本（如超参数摘要、配置 JSON）。

        参数
        ----------
        tag : str
            标签名。
        text : str
            Markdown 格式的文本内容。
        step : int
            步数。
        """
        self._writer.add_text(tag, text, global_step=step)

    # ── Hyperparameters ──────────────────────────────────────────

    def log_hparams(
        self,
        hparam_dict: Dict[str, Union[str, float, int, bool]],
        metric_dict: Dict[str, float],
    ) -> None:
        """
        记录超参数及其对应的最终指标（用于 HParams 面板）。

        参数
        ----------
        hparam_dict : dict
            超参数字典，如 ``{"lr": 0.001, "gamma": 0.99}``。
        metric_dict : dict
            最终评价指标，如 ``{"final/reward_mean": 50.0}``。
        """
        self._writer.add_hparams(hparam_dict, metric_dict)

    # ── Utilities ────────────────────────────────────────────────

    @property
    def log_dir(self) -> str:
        """返回实际的日志目录路径。"""
        return self._writer.log_dir

    def flush(self) -> None:
        """立即将缓冲的事件写入磁盘。"""
        self._writer.flush()

    def close(self) -> None:
        """关闭 writer，释放资源。"""
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
