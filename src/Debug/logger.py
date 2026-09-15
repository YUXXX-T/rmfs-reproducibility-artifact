"""
日志模块
=============
封装 Python logging 模块的可配置仿真日志器。
"""

import logging
import sys
from typing import Optional


class SimLogger:
    """
    提供按模块日志记录、可配置级别的仿真日志器。

    Usage
    -----
    >>> logger = SimLogger("Engine", level="DEBUG")
    >>> logger.info("Simulation started")
    >>> logger.debug("Tick 42: agents moved")
    """

    _loggers: dict = {}

    def __init__(
        self,
        name: str = "MAS_RMFS",
        level: str = "INFO",
        log_file: Optional[str] = None,
    ):
        # 如果此名称的日志器已创建则复用
        if name in SimLogger._loggers:
            self._logger = SimLogger._loggers[name]
            return

        self._logger = logging.getLogger(f"MAS_RMFS.{name}")
        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        self._logger.propagate = False

        # 控制台处理器
        if not self._logger.handlers:
            fmt = logging.Formatter(
                fmt="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
                datefmt="%H:%M:%S",
            )
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(fmt)
            self._logger.addHandler(console_handler)

            # 可选的文件处理器
            if log_file:
                file_handler = logging.FileHandler(log_file, encoding="utf-8")
                file_handler.setFormatter(fmt)
                self._logger.addHandler(file_handler)

        SimLogger._loggers[name] = self._logger

    def debug(self, msg: str, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)
