"""
面向动态场景的三维障碍物检测与分割系统研究 — 程序入口

启动流程：
  1. 解析 config/settings.yaml
  2. 初始化日志系统
  3. 启动 PyQt5 GUI 主窗口（MainWindow）
"""

import sys
from pathlib import Path

import yaml
from PyQt5.QtWidgets import QApplication

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.utils.logger import setup_logger, get_logger
from app.ui.main_window import MainWindow


def load_config(config_path: Path) -> dict:
    """
    读取 YAML 配置文件，返回配置字典。
    若文件不存在则返回空字典，系统将使用代码内的默认值。
    """
    if not config_path.exists():
        print(f"[警告] 配置文件不存在: {config_path}，使用默认配置")
        return {}

    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    return cfg


def main() -> None:
    """程序主函数"""
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    config = load_config(config_path)

    log_cfg = config.get("logging", {})
    setup_logger(
        name="dynamic_3d",
        level=log_cfg.get("level", "DEBUG"),
        to_file=log_cfg.get("to_file", True),
        log_dir=log_cfg.get("log_dir", "outputs/logs"),
        filename=log_cfg.get("filename", "app.log"),
        project_root=PROJECT_ROOT,
    )

    logger = get_logger("main")
    app_info = config.get("app", {})
    _default_display_name = "面向动态场景的三维障碍物检测与分割系统研究"
    logger.info(
        "启动 %s v%s | 阶段: %s",
        app_info.get("name") or app_info.get("main_title") or _default_display_name,
        app_info.get("version", "1.0.0"),
        app_info.get("stage", "phase5_fusion_fullrun"),
    )

    # 3. 创建 PyQt5 应用（高 DPI 须在 QApplication 创建前设置）
    from PyQt5.QtCore import Qt as _Qt

    try:
        QApplication.setAttribute(_Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(_Qt.AA_UseHighDpiPixmaps, True)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName(
        app_info.get("name")
        or app_info.get("main_title")
        or "面向动态场景的三维障碍物检测与分割系统研究"
    )
    app.setApplicationVersion(app_info.get("version", "1.0.0"))

    window = MainWindow(config=config)
    window.show()

    logger.info("主窗口已显示，进入事件循环")
    exit_code = app.exec_()
    logger.info("程序退出，退出码=%d", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
