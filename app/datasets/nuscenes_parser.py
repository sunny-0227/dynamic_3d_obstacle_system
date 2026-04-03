"""
nuScenes 元数据解析（纯逻辑，不持有 NuScenes 长生命周期对象）
根据官方表格构建 sample_token 序列，并解析每帧 LiDAR 文件路径。

说明：
    本模块仅依赖传入的 NuScenes 实例（duck typing：需具备 .get / .dataroot / .scene / .sample），
    便于单元测试与算法层解耦；具体 SDK 由 nuscenes_loader 负责导入。

与 GUI 的衔接：
    数据集连接、导航模式、帧索引与 LiDAR 路径均由 NuScenesMiniLoader + NuScenesFrameRecord
    统一输出；界面层应区分「单文件点云」与「nuScenes 数据集」两种工作流，避免混用入口。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Literal, Optional

from app.utils.logger import get_logger

logger = get_logger("datasets.nuscenes_parser")

# 导航模式：全表样本顺序 / 单场景链式顺序
NavigationMode = Literal["global", "scene"]


@dataclass(frozen=True)
class NuScenesFrameRecord:
    """
    单帧 LiDAR 的统一描述（数据集层 → 算法/可视化层的契约结构）。

    算法模块应只读取本数据类字段，不直接访问 nuScenes 内部表；
    点云几何由 app.io.pointcloud_loader.load_pointcloud(lidar_path) 加载。

    里程碑2 仅提供最小字段；接入真实检测/分割时建议扩展（可另建子类或并行 DTO）：
        - ego_pose_token / 平移四元数（lidar 与全局/自车坐标对齐）
        - calibrated_sensor / 外参矩阵（lidar2ego、ego2global）
        - 相机 sample_data 路径或多视图路径列表
        - sample_annotation 或 3D box 列表（训练与评测）
        - 非关键帧 LIDAR（sweeps）路径序列（若模型需要时序）
    """

    sample_token: str
    scene_token: str
    scene_name: str
    frame_index: int
    frame_count: int
    lidar_path: Path
    timestamp_us: Optional[float]
    navigation_mode: NavigationMode
    version: str


def get_lidar_path_for_sample(nusc: Any, sample_token: str) -> Path:
    """
    根据 sample_token 解析该关键帧对应的 LIDAR_TOP 二进制路径。

    参数：
        nusc:   NuScenes 实例
        sample_token: sample 表主键

    返回：
        dataroot 下的绝对 pathlib.Path（文件通常为 samples/LIDAR_TOP/*.bin）
    """
    sample_rec = nusc.get("sample", sample_token)
    lidar_sd_token = sample_rec["data"]["LIDAR_TOP"]
    sd_rec = nusc.get("sample_data", lidar_sd_token)
    rel = sd_rec["filename"]
    path = Path(nusc.dataroot) / rel
    logger.debug("sample=%s -> lidar=%s", sample_token, path)
    return path


def list_scene_summaries(nusc: Any) -> List[dict]:
    """
    列出数据集中所有场景摘要，按场景名称排序。

    返回：
        每项包含 token、name、sample_count
    """
    summaries: List[dict] = []
    for scene in nusc.scene:
        stok = scene["token"]
        cnt = count_samples_in_scene(nusc, stok)
        summaries.append(
            {
                "token": stok,
                "name": scene["name"],
                "sample_count": cnt,
            }
        )
    summaries.sort(key=lambda x: x["name"])
    return summaries


def count_samples_in_scene(nusc: Any, scene_token: str) -> int:
    """统计某场景下关键帧 sample 数量（沿 next 链遍历）。"""
    scene = nusc.get("scene", scene_token)
    tok: Optional[str] = scene["first_sample_token"]
    n = 0
    while tok:
        n += 1
        s = nusc.get("sample", tok)
        nxt = s.get("next")
        tok = nxt if nxt else None
    return n


def build_sample_token_list_global(nusc: Any) -> List[str]:
    """
    按官方 sample 表顺序（与 v1.0*.json 中条目顺序一致）构建全数据集 sample_token 列表。

    注意：此顺序一般不等于「全数据集采集时间线」或「按 scene 连续」；
    时序实验请在 loader 层增加按 timestamp 排序等策略，勿默认等同于时间序。
    """
    tokens = [rec["token"] for rec in nusc.sample]
    logger.info("全局样本序列构建完成，共 %d 帧", len(tokens))
    return tokens


def build_sample_token_list_for_scene(nusc: Any, scene_token: str) -> List[str]:
    """按场景内 first_sample -> next 链构建 sample_token 列表。"""
    scene = nusc.get("scene", scene_token)
    tokens: List[str] = []
    tok: Optional[str] = scene["first_sample_token"]
    while tok:
        tokens.append(tok)
        s = nusc.get("sample", tok)
        nxt = s.get("next")
        tok = nxt if nxt else None
    logger.info("场景 %s 样本序列构建完成，共 %d 帧", scene_token, len(tokens))
    return tokens


def build_frame_record(
    nusc: Any,
    sample_token: str,
    frame_index: int,
    frame_count: int,
    navigation_mode: NavigationMode,
    version: str,
) -> NuScenesFrameRecord:
    """
    将 sample_token 与导航上下文打包为 NuScenesFrameRecord。
    """
    sample_rec = nusc.get("sample", sample_token)
    scene_token = sample_rec["scene_token"]
    scene_rec = nusc.get("scene", scene_token)
    scene_name = scene_rec["name"]
    lidar_path = get_lidar_path_for_sample(nusc, sample_token)
    ts = sample_rec.get("timestamp")
    try:
        ts_val = float(ts) if ts is not None else None
    except (TypeError, ValueError):
        ts_val = None

    return NuScenesFrameRecord(
        sample_token=sample_token,
        scene_token=scene_token,
        scene_name=scene_name,
        frame_index=frame_index,
        frame_count=frame_count,
        lidar_path=lidar_path,
        timestamp_us=ts_val,
        navigation_mode=navigation_mode,
        version=version,
    )
