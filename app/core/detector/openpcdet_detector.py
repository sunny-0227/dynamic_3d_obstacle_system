"""
OpenPCDet 检测器封装（降级/备用模式）

当 OpenPCDetJsonDetector（WSL 真实推理）未启用时，由此类提供模拟检测结果，
输出标准 DetectionBox 格式，保证 pipeline 正常运行。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import List, Optional

import numpy as np

from app.core.detector.base_detector import BaseDetector, DetectionBox
from app.utils.logger import get_logger

logger = get_logger("core.detector.openpcdet_detector")


DEFAULT_CLASS_NAMES = ["car", "pedestrian", "cyclist"]
DEFAULT_CLASS_SIZES = {
    "car": (4.5, 2.0, 1.6),  # [l, w, h]
    "pedestrian": (0.8, 0.8, 1.8),
    "cyclist": (1.8, 0.8, 1.6),
}


class OpenPCDetDetector(BaseDetector):
    """OpenPCDet 检测器封装类（enable_wsl=false 时的降级检测器）。"""

    def __init__(
        self,
        model_cfg_path: Optional[Path] = None,
        checkpoint_path: Optional[Path] = None,
        device: str = "cpu",
        score_threshold: float = 0.1,
        num_boxes_fake: int = 3,
        class_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.model_cfg_path = Path(model_cfg_path) if model_cfg_path else None
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.device = device
        self.score_threshold = float(score_threshold)
        self.num_boxes_fake = int(num_boxes_fake)
        self.class_names = class_names if class_names else list(DEFAULT_CLASS_NAMES)

        self._openpcdet_ready = False
        self._try_init_openpcdet()

    def _try_init_openpcdet(self) -> None:
        """尝试导入 OpenPCDet 依赖，不可用时标记 _openpcdet_ready=False。"""
        if self.model_cfg_path is None or self.checkpoint_path is None:
            logger.warning("未提供 OpenPCDet 配置/权重，将使用模拟检测。")
            self._openpcdet_ready = False
            return

        try:
            import torch  # noqa: F401
            import pcdet  # noqa: F401
            self._openpcdet_ready = True
            logger.info("检测到 OpenPCDet 依赖可导入，当前仍使用模拟检测（fallback）。")
        except Exception as exc:
            logger.warning(
                "OpenPCDet 依赖不可用，将使用占位检测实现：%s",
                exc,
            )
            self._openpcdet_ready = False

    def _detect_impl(self, points: np.ndarray) -> List[DetectionBox]:
        """
        执行检测。

        说明：
            BaseDetector 会传入 (N,K) 的点云（K>=3）。占位实现仅使用 xyz；
            后续真实 OpenPCDet 推理可按配置决定使用 (N,4) 等特征。
        """
        points_xyz = np.asarray(points[:, :3], dtype=np.float32)
        if self._openpcdet_ready:
            try:
                return self._detect_openpcdet(points_xyz)
            except NotImplementedError:
                logger.warning("OpenPCDet 真实检测未实现，使用占位检测。")
            except Exception as exc:
                logger.error("OpenPCDet 检测执行失败，fallback 到占位实现: %s", exc)

        return self._detect_fake(points_xyz)

    def _detect_openpcdet(self, points_xyz: np.ndarray) -> List[DetectionBox]:
        """
        真实 OpenPCDet 推理实现替换位置。

        你在后续里程碑只需要在这里替换为真实推理即可，输出仍必须是 List[DetectionBox]。

        建议替换步骤（参考 OpenPCDet tools/demo.py 思路）：
          1) cfg_from_yaml_file / cfg.MODEL 等加载模型配置
          2) build_network + load_params_from_file
          3) 构建 DemoDataset 或 DataDict（prepare_data / collate_batch）
          4) model.forward(data_dict)，读取 pred_dicts[0]['pred_boxes'/'pred_scores'/'pred_labels']
          5) 将 pred_boxes 转为 center/size/yaw 标准格式

        当前阶段为了稳定性，不直接实现复杂推理调用。
        """
        raise NotImplementedError

    def _detect_fake(self, points_xyz: np.ndarray) -> List[DetectionBox]:
        """
        占位检测：根据点云空间范围随机生成若干 3D 框，并输出统一 DetectionBox。

        与里程碑 1 的 `app.core.fake_detector.run_fake_detection` 语义相近，但输出为 `DetectionBox`
        （绕 Z 轴 yaw），供 `BaseDetector` / pipeline 统一接口使用；不直接依赖 fake_detector 模块，
        避免 Open3D 几何类型与 pipeline 数据结构耦合。
        """
        if points_xyz.size == 0:
            return []

        x_min, y_min, z_min = points_xyz[:, 0].min(), points_xyz[:, 1].min(), points_xyz[:, 2].min()
        x_max, y_max = points_xyz[:, 0].max(), points_xyz[:, 1].max()
        if x_max - x_min < 1e-6:
            x_max = x_min + 1.0
        if y_max - y_min < 1e-6:
            y_max = y_min + 1.0

        # 若点云范围很小，margin 也可能导致 a>b，使用 clamp 方式
        margin = 1.0
        cx_low = x_min + margin
        cx_high = x_max - margin
        cy_low = y_min + margin
        cy_high = y_max - margin
        if cx_low > cx_high:
            cx_low, cx_high = x_min, x_max
        if cy_low > cy_high:
            cy_low, cy_high = y_min, y_max

        num_boxes = max(1, min(self.num_boxes_fake, 3))  # 里程碑1主要是2~3，这里放宽到1也能跑

        results: List[DetectionBox] = []
        for _ in range(num_boxes):
            class_name = random.choice(self.class_names)
            if class_name not in DEFAULT_CLASS_SIZES:
                class_name = self.class_names[0]

            l, w, h = DEFAULT_CLASS_SIZES[class_name]
            # 在点云 XY 范围中随机选取中心
            cx = random.uniform(cx_low, cx_high)
            cy = random.uniform(cy_low, cy_high)
            cz = z_min + h / 2.0

            yaw = random.uniform(0.0, np.pi)  # 占位：绕 Z 轴旋转

            score = random.uniform(self.score_threshold, 0.99)
            score = float(np.clip(score, 0.0, 1.0))

            results.append(
                DetectionBox(
                    class_name=class_name,
                    score=score,
                    center=np.array([cx, cy, cz], dtype=np.float32),
                    size=np.array([l, w, h], dtype=np.float32),
                    yaw=float(yaw),
                )
            )

        # 按 score 降序，便于 GUI 展示
        results.sort(key=lambda b: b.score, reverse=True)
        return results

