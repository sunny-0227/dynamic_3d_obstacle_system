"""
MMDetection3D 分割器封装（里程碑 4）

说明：
  - 当前实现提供“可运行的占位分割”，保证工程闭环与 GUI 演示可用。
  - 若环境中已安装 mmdet3d + mmengine 等依赖，可在 _segment_real_impl 中接入真实推理逻辑。
  - 替换点已在代码中标注（不要在 GUI 层写框架耦合逻辑）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from app.core.segmentor.base_segmentor import BaseSegmentor, SegmentationResult
from app.utils.logger import get_logger

logger = get_logger("core.segmentor.mmdet3d_segmentor")


@dataclass(frozen=True)
class MMDet3DSegmentorConfig:
    # 真实推理相关（占位实现不强依赖）
    config_file: str = ""
    checkpoint_file: str = ""
    device: str = "cpu"  # "cpu" / "cuda"

    # 占位分割相关
    num_classes: int = 4
    class_names: Optional[List[str]] = None
    palette: Optional[Dict[int, List[float]]] = None


class MMDet3DSegmentor(BaseSegmentor):
    """
    MMDetection3D 分割器封装类。

    当前策略：
      - 若 config/checkpoint 均非空，尝试初始化真实后端；失败则降级为占位分割。
      - 若为空，则直接走占位分割。
    """

    def __init__(self, cfg: MMDet3DSegmentorConfig):
        self._cfg = cfg
        self._use_real_backend = False
        self._real_model = None

        self._id_to_name = self._build_id_to_name(cfg.num_classes, cfg.class_names)
        self._id_to_color = self._build_id_to_color(cfg.num_classes, cfg.palette)

        if cfg.config_file and cfg.checkpoint_file:
            self._use_real_backend = self._try_init_real_backend()

        if self._use_real_backend:
            logger.info("MMDet3D 分割后端：真实模式（已初始化）")
        else:
            logger.info("MMDet3D 分割后端：占位模式（可运行演示）")

    def _segment_impl(self, points: np.ndarray) -> SegmentationResult:
        if self._use_real_backend:
            labels = self._segment_real_impl(points)
        else:
            labels = self._segment_fake_impl(points)

        labels = np.asarray(labels, dtype=np.int32)
        if labels.ndim != 1 or labels.shape[0] != points.shape[0]:
            raise ValueError(
                f"labels 形状不匹配：期望 ({points.shape[0]},)，实际 {labels.shape}"
            )

        return SegmentationResult(
            labels=labels,
            id_to_name=self._id_to_name,
            id_to_color=self._id_to_color,
        )

    def _try_init_real_backend(self) -> bool:
        """
        尝试初始化真实 MMDetection3D 后端。

        注意：为了保证在未安装依赖时也能运行，这里只做“温和探测”，失败直接降级。
        """
        try:
            # 替换点：如果你要接真实模型，这里应使用 mmengine / mmdet3d 的初始化 API
            # 例如：from mmengine.config import Config
            #      from mmdet3d.apis import init_model
            #      self._real_model = init_model(cfg, ckpt, device=...)
            import importlib

            importlib.import_module("mmengine")
            importlib.import_module("mmdet3d")
        except Exception as e:
            logger.warning("MMDet3D 依赖未就绪，降级占位分割：%s", e)
            return False

        # 当前仍不默认启用真实推理，避免依赖/API 变化导致工程不可运行
        logger.warning(
            "检测到 mmdet3d 依赖，但真实推理逻辑尚未接入，仍使用占位分割。"
        )
        self._real_model = None
        return False

    def _segment_real_impl(self, points: np.ndarray) -> np.ndarray:
        """
        真实推理替换点：把 points 送入 MMDet3D 分割模型，输出 (N,) labels。
        """
        raise NotImplementedError(
            "真实 MMDet3D 分割尚未接入。请在 _segment_real_impl 中实现推理逻辑。"
        )

    def _segment_fake_impl(self, points: np.ndarray) -> np.ndarray:
        """
        占位分割：基于简单几何规则生成逐点类别。
        目标：可解释、稳定、可复现、无需额外依赖。
        """
        xyz = points[:, :3]
        x = xyz[:, 0]
        y = xyz[:, 1]
        z = xyz[:, 2]

        labels = np.zeros((xyz.shape[0],), dtype=np.int32)
        if xyz.shape[0] == 0:
            return labels

        # 类别定义（默认 4 类）：
        # 0: background
        # 1: ground（低处）
        # 2: obstacle（较高且靠近中心）
        # 3: vegetation（其余）
        z_lo = np.percentile(z, 20)
        z_hi = np.percentile(z, 75)
        r = np.sqrt(x * x + y * y)

        ground = z <= z_lo
        obstacle = (z >= z_hi) & (r <= np.percentile(r, 45))
        vegetation = (~ground) & (~obstacle)

        labels[ground] = 1 if self._cfg.num_classes > 1 else 0
        if self._cfg.num_classes > 2:
            labels[obstacle] = 2
        if self._cfg.num_classes > 3:
            labels[vegetation] = 3

        # 若 num_classes < 4，进行裁剪
        labels = np.minimum(labels, self._cfg.num_classes - 1)
        return labels

    @staticmethod
    def _build_id_to_name(num_classes: int, class_names: Optional[List[str]]) -> Dict[int, str]:
        if class_names and len(class_names) >= num_classes:
            return {i: str(class_names[i]) for i in range(num_classes)}
        # 默认名
        defaults = ["background", "ground", "obstacle", "vegetation"]
        id_to_name: Dict[int, str] = {}
        for i in range(num_classes):
            id_to_name[i] = defaults[i] if i < len(defaults) else f"class_{i}"
        return id_to_name

    @staticmethod
    def _build_id_to_color(
        num_classes: int, palette: Optional[Dict[int, List[float]]]
    ) -> Dict[int, List[float]]:
        # 默认 palette（归一化）
        defaults: Dict[int, List[float]] = {
            0: [0.55, 0.55, 0.55],  # background
            1: [0.20, 0.55, 0.95],  # ground
            2: [0.90, 0.25, 0.25],  # obstacle
            3: [0.20, 0.85, 0.25],  # vegetation
        }
        id_to_color: Dict[int, List[float]] = {}
        for i in range(num_classes):
            if palette and i in palette:
                c = palette[i]
                id_to_color[i] = [float(c[0]), float(c[1]), float(c[2])]
            else:
                id_to_color[i] = defaults.get(i, [0.8, 0.8, 0.2])
        return id_to_color

