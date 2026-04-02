"""
nuScenes mini 数据集适配器
封装 nuscenes-devkit，提供连接数据集、切换导航模式、按索引取帧的统一入口。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from app.datasets.nuscenes_parser import (
    NavigationMode,
    NuScenesFrameRecord,
    build_frame_record,
    build_sample_token_list_for_scene,
    build_sample_token_list_global,
    list_scene_summaries,
)
from app.utils.logger import get_logger

logger = get_logger("datasets.nuscenes_loader")


class NuScenesMiniLoader:
    """
    nuScenes（含 mini）数据根目录适配器。

    使用流程：
        1. 构造时传入 dataroot、version（通常为 v1.0-mini）
        2. connect() 加载 NuScenes 对象
        3. set_navigation(mode, scene_token=None) 构建当前 token 列表
        4. get_frame_record(index) 获取统一帧描述
    """

    def __init__(self, dataroot: Path, version: str = "v1.0-mini") -> None:
        self.dataroot = Path(dataroot)
        self.version = version
        self._nusc = None
        self._navigation_mode: NavigationMode = "global"
        self._tokens: List[str] = []
        self._scene_summaries: List[dict] = []

    @property
    def is_connected(self) -> bool:
        return self._nusc is not None

    @property
    def frame_count(self) -> int:
        return len(self._tokens)

    @property
    def navigation_mode(self) -> NavigationMode:
        return self._navigation_mode

    def connect(self) -> None:
        """
        实例化 NuScenes 并缓存场景列表。
        若未安装 nuscenes-devkit，抛出 ImportError。
        """
        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError as exc:
            logger.error("未安装 nuscenes-devkit，请执行: pip install nuscenes-devkit")
            raise ImportError(
                "缺少依赖 nuscenes-devkit，无法加载 nuScenes 数据集"
            ) from exc

        meta_dir = self.dataroot / self.version
        if not meta_dir.is_dir():
            raise FileNotFoundError(
                f"未找到元数据目录: {meta_dir}，请确认 dataroot 与 version 配置正确"
            )

        logger.info("正在连接 nuScenes | dataroot=%s | version=%s", self.dataroot, self.version)
        self._nusc = NuScenes(version=self.version, dataroot=str(self.dataroot), verbose=False)
        self._scene_summaries = list_scene_summaries(self._nusc)
        logger.info("连接成功，场景数: %d", len(self._scene_summaries))

    def get_scene_summaries(self) -> List[dict]:
        """返回场景摘要列表（connect 之后可用）。"""
        return list(self._scene_summaries)

    def set_navigation(self, mode: NavigationMode, scene_token: Optional[str] = None) -> None:
        """
        设置导航模式并重建 sample_token 列表。

        参数：
            mode:        "global" 全表顺序；"scene" 单场景链式顺序
            scene_token: mode 为 scene 时必填
        """
        if self._nusc is None:
            raise RuntimeError("请先调用 connect()")

        self._navigation_mode = mode
        if mode == "global":
            self._tokens = build_sample_token_list_global(self._nusc)
        else:
            if not scene_token:
                raise ValueError("按场景导航时必须提供 scene_token")
            self._tokens = build_sample_token_list_for_scene(self._nusc, scene_token)

        if not self._tokens:
            logger.warning("当前导航下列表为空，请检查数据完整性")

    def get_frame_record(self, index: int) -> NuScenesFrameRecord:
        """
        按当前导航列表下标获取 NuScenesFrameRecord。

        异常：
            IndexError: 索引越界或列表为空
        """
        if self._nusc is None:
            raise RuntimeError("请先调用 connect()")
        if not self._tokens:
            raise IndexError("当前无可用帧（token 列表为空）")
        if index < 0 or index >= len(self._tokens):
            raise IndexError(f"帧索引越界: {index}，有效范围 0..{len(self._tokens)-1}")

        tok = self._tokens[index]
        return build_frame_record(
            self._nusc,
            sample_token=tok,
            frame_index=index,
            frame_count=len(self._tokens),
            navigation_mode=self._navigation_mode,
            version=self.version,
        )
