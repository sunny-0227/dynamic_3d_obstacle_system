"""
日志工具模块
提供统一的日志初始化与获取接口，支持同时输出到控制台和文件。
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "dynamic_3d",
    level: str = "DEBUG",
    to_file: bool = True,
    log_dir: str = "outputs/logs",
    filename: str = "app.log",
    project_root: Optional[Path] = None,
) -> logging.Logger:
    """
    初始化并返回根日志器。
    应在程序启动时调用一次，后续通过 get_logger() 获取子日志器。

    参数：
        name:         根日志器名称
        level:        日志级别字符串（DEBUG/INFO/WARNING/ERROR）
        to_file:      是否同时写入文件
        log_dir:      日志目录（相对项目根目录）
        filename:     日志文件名
        project_root: 项目根目录路径，默认使用此文件上溯两级
    """
    # 确定项目根目录
    if project_root is None:
        # 此文件位于 app/utils/logger.py，上溯两级即项目根
        project_root = Path(__file__).resolve().parent.parent.parent

    logger = logging.getLogger(name)

    # 直接检查 handlers 是否已存在，防止重复添加（对 importlib.reload 也安全）
    if logger.handlers:
        return logger

    # 解析日志级别
    numeric_level = getattr(logging, level.upper(), logging.DEBUG)
    logger.setLevel(numeric_level)

    # 统一格式：时间 | 级别 | 模块 | 消息
    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台 handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(numeric_level)
    logger.addHandler(console_handler)

    # 文件 handler
    if to_file:
        log_path = project_root / log_dir
        log_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_path / filename, encoding="utf-8", mode="a"
        )
        file_handler.setFormatter(fmt)
        file_handler.setLevel(numeric_level)
        logger.addHandler(file_handler)

    # 避免日志向 root logger 传播导致重复打印
    logger.propagate = False

    logger.info("日志系统初始化完成 | 根目录: %s", project_root)
    return logger


def get_logger(module_name: str) -> logging.Logger:
    """
    获取子日志器，命名格式为 dynamic_3d.<module_name>。
    必须在 setup_logger() 调用之后使用。

    参数：
        module_name: 子模块名称（如 'io.loader'、'core.detector'）
    """
    return logging.getLogger(f"dynamic_3d.{module_name}")
