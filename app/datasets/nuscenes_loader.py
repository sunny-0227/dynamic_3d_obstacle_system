"""
nuScenes mini 数据集适配器（真实读取 + 无标注时模拟降级）

目标：
  1) 若根目录包含完整官方 nuScenes mini 元数据（sample/scene/sample_data 等 json），走“真实模式”
  2) 若元数据不完整，不直接报错退出，而是降级到“模拟模式”：扫描 samples/LIDAR_TOP/*.bin
  3) 模拟模式下仍支持 GUI 的场景选择、帧索引切换、加载当前帧点云
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

    支持模式：
      - real（真实模式）：基于 nuscenes-devkit 解析 sample/scene/sample_data 表
      - simulated（模拟模式）：仅扫描 samples/LIDAR_TOP/*.bin，构造伪 scene / 伪 sample
    """

    MODE_REAL = "real"
    MODE_SIMULATED = "simulated"

    def __init__(self, dataroot: Path, version: str = "v1.0-mini") -> None:
        self.dataroot = Path(dataroot)
        self.version = version

        # --------- 真实模式字段 ---------
        self._nusc = None

        # --------- 模拟模式字段 ---------
        self._mode: str = "unknown"
        self._sim_bins: List[Path] = []
        self._sim_sample_tokens: List[str] = []
        self._sim_token_to_global_index: Dict[str, int] = {}
        self._sim_chunk_size: int = 1
        self._sim_scene_summaries: List[dict] = []
        self._sim_scene_token_to_range: Dict[str, Tuple[int, int]] = {}
        self._sim_scene_token_to_name: Dict[str, str] = {}

        # --------- 统一导航字段 ---------
        self._navigation_mode: NavigationMode = "global"
        self._tokens: List[str] = []
        self._scene_summaries: List[dict] = []

    @property
    def mode(self) -> str:
        """返回当前工作模式：`real` 或 `simulated`。"""
        return self._mode

    @property
    def is_connected(self) -> bool:
        return self._mode in (self.MODE_REAL, self.MODE_SIMULATED)

    @property
    def frame_count(self) -> int:
        return len(self._tokens)

    @property
    def navigation_mode(self) -> NavigationMode:
        return self._navigation_mode

    def _real_meta_dir(self) -> Path:
        return self.dataroot / self.version

    def _real_required_tables_exist(self) -> bool:
        """
        判断元数据是否“足够完整”。

        nuScenes-devkit 在 NuScenes 初始化时会读取 table_root 下的 {table}.json，
        因此这里只检查最关键的三个表：sample / scene / sample_data。
        """
        meta_dir = self._real_meta_dir()
        required = ["sample.json", "scene.json", "sample_data.json"]
        for fname in required:
            if not (meta_dir / fname).is_file():
                return False
        return True

    def _try_connect_real(self) -> bool:
        """
        尝试以真实模式连接；成功则返回 True，失败返回 False（不抛异常）。
        """
        meta_dir = self._real_meta_dir()
        if not meta_dir.is_dir():
            return False
        if not self._real_required_tables_exist():
            return False

        try:
            from nuscenes.nuscenes import NuScenes
        except ImportError:
            logger.warning("nuscenes-devkit 未安装，但检测到元数据存在，将尝试模拟模式降级")
            return False

        try:
            logger.info(
                "检测到完整元数据，进入真实模式 | dataroot=%s | version=%s",
                self.dataroot,
                self.version,
            )
            self._nusc = NuScenes(version=self.version, dataroot=str(self.dataroot), verbose=False)
            self._scene_summaries = list_scene_summaries(self._nusc)
            self._mode = self.MODE_REAL
            return True
        except Exception as exc:
            logger.warning("真实模式连接失败（将降级模拟模式）: %s", exc, exc_info=True)
            return False

    def _scan_simulated_bins(self) -> List[Path]:
        """
        扫描模拟点云文件：dataroot/samples/LIDAR_TOP/*.bin
        """
        bins_dir = self.dataroot / "samples" / "LIDAR_TOP"
        if not bins_dir.is_dir():
            return []
        bins = sorted(bins_dir.glob("*.bin"), key=lambda p: p.name)
        return bins

    def _build_simulated_scenes(self, bins: List[Path]) -> None:
        """
        将一组 bin 文件组织成伪 scene / 伪 frame。

        策略：
          - 按文件名排序
          - 最多划分为 3 个伪 scene（保证只有一个 bin 时仍可形成最小可用结构）
        """
        n = len(bins)
        if n <= 0:
            raise FileNotFoundError("模拟模式需要至少 1 个 .bin 文件，但未扫描到")

        # 将 n 切成最多 3 段，避免 scene combo 过多
        desired_scene_count = min(3, n)
        self._sim_chunk_size = int(math.ceil(n / desired_scene_count))

        self._sim_bins = bins
        self._sim_sample_tokens = [f"FAKE_SAMPLE_{i:06d}" for i in range(n)]
        self._sim_token_to_global_index = {tok: i for i, tok in enumerate(self._sim_sample_tokens)}

        self._sim_scene_summaries = []
        self._sim_scene_token_to_range = {}
        self._sim_scene_token_to_name = {}

        scene_count = int(math.ceil(n / self._sim_chunk_size))
        for scene_id in range(scene_count):
            start = scene_id * self._sim_chunk_size
            end = min((scene_id + 1) * self._sim_chunk_size, n)
            scene_token = f"FAKE_SCENE_{scene_id:03d}"
            scene_name = f"模拟场景 {scene_id + 1}"
            sample_count = end - start

            self._sim_scene_token_to_range[scene_token] = (start, end)
            self._sim_scene_token_to_name[scene_token] = scene_name
            self._sim_scene_summaries.append(
                {"token": scene_token, "name": scene_name, "sample_count": sample_count}
            )

        # 初始化默认导航：global
        self._tokens = []
        self._scene_summaries = list(self._sim_scene_summaries)

    def _try_connect_simulated(self) -> bool:
        """
        尝试以模拟模式连接；成功则返回 True，失败返回 False（不抛异常）。
        """
        bins = self._scan_simulated_bins()
        if not bins:
            return False

        logger.info("未发现/未能满足完整元数据，将进入模拟模式 | bin 文件数: %d", len(bins))
        self._mode = self.MODE_SIMULATED
        self._nusc = None
        self._build_simulated_scenes(bins)
        return True

    def connect(self) -> None:
        """
        连接 nuScenes mini 数据集。

        规则：
          - 若元数据足够完整，优先真实模式
          - 否则降级模拟模式（不会直接报错退出）
          - 若模拟模式所需 bin 也不存在，则抛出异常给 UI 做提示
        """
        # 清理旧状态（避免重复连接）
        self._nusc = None
        self._mode = "unknown"
        self._tokens = []
        self._scene_summaries = []

        real_ok = self._try_connect_real()
        if real_ok:
            logger.info("nuScenes 连接成功，当前走真实模式")
            return

        sim_ok = self._try_connect_simulated()
        if sim_ok:
            logger.info("nuScenes 连接成功，当前走模拟模式")
            return

        # 两种模式都失败：给出明确错误
        bins_dir = self.dataroot / "samples" / "LIDAR_TOP"
        meta_dir = self._real_meta_dir()
        raise FileNotFoundError(
            "无法连接 nuScenes：\n"
            f"- 真实模式元数据目录不存在或不完整：{meta_dir}\n"
            f"- 模拟模式也未扫描到 bin 文件：{bins_dir}\n"
            "请确认你的数据集根目录结构符合 nuscenes mini 组织方式，或补充 samples/LIDAR_TOP/*.bin"
        )

    def get_scene_summaries(self) -> List[dict]:
        """返回场景摘要列表（connect 之后可用）。"""
        return list(self._scene_summaries)

    def set_navigation(self, mode: NavigationMode, scene_token: Optional[str] = None) -> None:
        """
        设置导航模式并重建 sample_token 列表。

        参数：
            mode:        "global" 全表顺序；"scene" 单场景顺序
            scene_token: mode 为 scene 时必填
        """
        self._navigation_mode = mode

        if self._mode == self.MODE_REAL:
            if self._nusc is None:
                raise RuntimeError("真实模式下未初始化，请先调用 connect()")
            if mode == "global":
                self._tokens = build_sample_token_list_global(self._nusc)
            else:
                if not scene_token:
                    raise ValueError("按场景导航时必须提供 scene_token")
                self._tokens = build_sample_token_list_for_scene(self._nusc, scene_token)
        else:
            # simulated
            if mode == "global":
                self._tokens = list(self._sim_sample_tokens)
            else:
                if not scene_token:
                    raise ValueError("按场景导航时必须提供 scene_token")
                if scene_token not in self._sim_scene_token_to_range:
                    raise KeyError(f"未知场景 token: {scene_token}")
                start, end = self._sim_scene_token_to_range[scene_token]
                self._tokens = self._sim_sample_tokens[start:end]

        if not self._tokens:
            logger.warning("当前导航下列表为空（mode=%s）", mode)

    def get_frame_record(self, index: int) -> NuScenesFrameRecord:
        """
        按当前导航列表下标获取 NuScenesFrameRecord。
        """
        if not self._tokens:
            raise IndexError("当前无可用帧（token 列表为空）")
        if index < 0 or index >= len(self._tokens):
            raise IndexError(f"帧索引越界: {index}，有效范围 0..{len(self._tokens)-1}")

        tok = self._tokens[index]

        if self._mode == self.MODE_REAL:
            if self._nusc is None:
                raise RuntimeError("真实模式下未初始化，请先调用 connect()")
            return build_frame_record(
                self._nusc,
                sample_token=tok,
                frame_index=index,
                frame_count=len(self._tokens),
                navigation_mode=self._navigation_mode,
                version=self.version,
            )

        # simulated
        global_idx = self._sim_token_to_global_index[tok]
        lidar_path = self._sim_bins[global_idx]
        scene_id = global_idx // self._sim_chunk_size
        scene_token = f"FAKE_SCENE_{scene_id:03d}"
        scene_name = self._sim_scene_token_to_name.get(scene_token, scene_token)

        return NuScenesFrameRecord(
            sample_token=tok,
            scene_token=scene_token,
            scene_name=scene_name,
            frame_index=index,
            frame_count=len(self._tokens),
            lidar_path=lidar_path,
            timestamp_us=None,
            navigation_mode=self._navigation_mode,
            version=self.version,
        )
