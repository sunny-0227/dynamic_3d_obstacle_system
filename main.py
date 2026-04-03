"""
动态3D障碍物感知系统 - 程序入口
负责：
  1. 解析配置文件（config/settings.yaml）
  2. 初始化日志系统
  3. 启动 PyQt5 GUI 主窗口

里程碑2：nuScenes mini 适配在 GUI 中完成；数据集逻辑位于 app.datasets，
点云加载仍由 app.io.pointcloud_loader 完成，与算法层解耦。

里程碑3：真实/占位检测接口接入在 app.core.detector + app.core.pipeline 中完成，
GUI 提供“执行检测”按钮调用统一检测 pipeline。

里程碑4：真实/占位语义分割接口接入在 app.core.segmentor + app.core.pipeline 中完成，
GUI 提供“执行分割”按钮调用统一分割 pipeline。

里程碑5：检测框与分割结果的统一坐标融合：
  - app.core.geometry.transform 提供 SE(3) 变换（旋转/平移/齐次矩阵）
  - app.core.fusion.result_fusion 负责输出对齐到同一坐标系的 FusedScene
  - app.core.pipeline.full_pipeline 提供一键分析入口
  - app.visualization.scene_renderer 在同一 Open3D 视窗渲染点云+分割+检测框

里程碑6：GUI 完善（科研系统主界面）：
  - app.ui.controller：控制器 + 后台线程；workflow 与 nuScenes 导航默认逻辑
  - app.ui.widgets.*：左侧控制 / 顶栏 / 摘要卡片 / 分区状态栏 / 系统日志
  - app.ui.main_window：左右分栏 + 顶栏课题主副标题；严格按钮与数据源单选
"""

import sys
from pathlib import Path

import yaml
from PyQt5.QtWidgets import QApplication

# 将项目根目录加入模块搜索路径，保证 Windows 打包/直接运行均可 import
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
    # 1. 读取配置
    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    config = load_config(config_path)

    # 2. 初始化日志系统
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

    # 3. 创建 PyQt5 应用
    # 高 DPI（Windows 多显示器等）；须在 QApplication 创建前设置；旧版/精简 Qt 可能不支持
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

    # 4. 创建并显示主窗口
    window = MainWindow(config=config)
    window.show()

    logger.info("PyQt5 主窗口已显示，进入事件循环")
    exit_code = app.exec_()

    logger.info("程序退出，退出码: %d", exit_code)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
