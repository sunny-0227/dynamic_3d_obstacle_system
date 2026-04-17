from __future__ import annotations

"""
Mock 点云相机

功能：
  - 从指定目录循环读取 .bin / .pcd 文件，模拟连续实时点云流
  - 支持固定帧率输出（目标 FPS，通过调用方的帧率控制线程实现节拍）
  - 完整的帧计数、文件索引、点数日志
  - 不依赖任何真实相机硬件
  - 后续可与 RealSenseCamera 无缝替换（同一接口 IPointCloudCamera）

注意：
  - MockCamera 本身不做帧率限速（sleep 由外层 _RealtimeThread 控制）
  - .bin 用纯 numpy 读取，.pcd 用 Open3D 读取（建议在非 GUI 线程调用）
  - Windows 兼容：不使用 os.fork，所有路径均用 pathlib.Path
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from app.io.pointcloud_loader import load_points_xyz_numpy, load_pointcloud
from app.realtime.camera_interface import CameraFrame, CameraInfo, IPointCloudCamera
from app.utils.logger import get_logger

logger = get_logger("realtime.mock_camera")


# 支持的点云文件后缀（小写）
_SUPPORTED_SUFFIXES = {".bin", ".pcd"}


class MockCamera(IPointCloudCamera):
    """
    Mock 点云相机：从目录循环读取点云文件，模拟实时相机输入。

    参数：
      stream_dir  — 包含 .bin / .pcd 文件的目录路径
      loop        — 是否循环（True：循环读取；False：文件读完后抛 StopIteration）
      target_fps  — 目标帧率（仅记录到日志，实际节拍由外层线程控制）
      name        — 显示名称（可自定义，如 "Mock_Dataset_A"）

    用法示例：
      cam = MockCamera(Path("data/mini_samples"), loop=True, target_fps=5.0)
      cam.start()
      while cam.is_running:
          frame = cam.get_next_frame()
          process(frame.points_xyz)
      cam.stop()
    """

    def __init__(
        self,
        stream_dir: Path,
        *,
        loop: bool = True,
        target_fps: float = 5.0,
        name: str = "Mock",
    ) -> None:
        self._dir = Path(stream_dir)
        self._loop = bool(loop)
        self._target_fps = float(max(target_fps, 0.1))
        self._name = str(name)

        # 文件列表（start() 后填充）
        self._files: List[Path] = []

        # 状态
        self._running: bool = False
        self._file_idx: int = 0      # 当前文件索引（0-based，循环）
        self._frame_count: int = 0   # 已输出帧数（自 start() 起累计）

    # ------------------------------------------------------------------
    # IPointCloudCamera 属性
    # ------------------------------------------------------------------

    @property
    def source_name(self) -> str:
        return self._name

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def camera_info(self) -> CameraInfo:
        return CameraInfo(
            source_name=self._name,
            is_running=self._running,
            frame_count=self._frame_count,
            current_index=self._file_idx,
            total_files=len(self._files),
        )

    # ------------------------------------------------------------------
    # 额外只读属性（Mock 专属，供 UI / 日志使用）
    # ------------------------------------------------------------------

    @property
    def stream_dir(self) -> Path:
        """当前流目录路径。"""
        return self._dir

    @property
    def total_files(self) -> int:
        """目录中发现的点云文件总数（start() 后有效）。"""
        return len(self._files)

    @property
    def current_file_index(self) -> int:
        """当前文件索引（0-based），表示下一帧将读取的文件位置。"""
        return self._file_idx

    @property
    def frame_count(self) -> int:
        """已输出的帧数（自最近一次 start() 起累计）。"""
        return self._frame_count

    @property
    def target_fps(self) -> float:
        """目标帧率（记录用途，节拍由外层线程控制）。"""
        return self._target_fps

    @property
    def loop(self) -> bool:
        """是否循环读取。"""
        return self._loop

    # ------------------------------------------------------------------
    # 动态配置（可在 stop() 后调用以更改目录 / 帧率）
    # ------------------------------------------------------------------

    def set_stream_dir(self, stream_dir: Path) -> None:
        """
        更改流目录。
        若相机正在运行，需先调用 stop()，否则抛出 RuntimeError。
        """
        if self._running:
            raise RuntimeError("请先调用 stop() 再更改流目录")
        self._dir = Path(stream_dir)
        logger.info("[MockCamera] 流目录已更新：%s", self._dir)

    def set_target_fps(self, fps: float) -> None:
        """更改目标帧率（记录用途，可在运行期间调用）。"""
        self._target_fps = float(max(fps, 0.1))
        logger.info("[MockCamera] 目标帧率已更新：%.1f FPS", self._target_fps)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        启动 Mock Camera：扫描目录，建立文件列表，重置帧计数。

        异常：
          FileNotFoundError — 目录不存在或不包含任何支持的点云文件
        """
        if self._running:
            logger.warning("[MockCamera] 已处于运行状态，忽略重复 start() 调用")
            return

        self._reload_files()

        if not self._files:
            raise FileNotFoundError(
                f"[MockCamera] 目录中未找到 .bin 或 .pcd 文件：{self._dir}\n"
                f"请将点云文件放入该目录后重试。"
            )

        self._file_idx = 0
        self._frame_count = 0
        self._running = True

        logger.info(
            "[MockCamera] 启动成功 | 目录：%s | 文件数：%d | 目标帧率：%.1f FPS | 循环：%s",
            self._dir,
            len(self._files),
            self._target_fps,
            "是" if self._loop else "否",
        )
        # 列出前 5 个文件，方便调试
        preview = self._files[:5]
        for i, p in enumerate(preview):
            logger.debug("[MockCamera]   [%02d] %s", i, p.name)
        if len(self._files) > 5:
            logger.debug("[MockCamera]   ... 共 %d 个文件", len(self._files))

    def stop(self) -> None:
        """停止 Mock Camera，释放文件列表（幂等）。"""
        if not self._running:
            return
        self._running = False
        logger.info(
            "[MockCamera] 已停止 | 共输出 %d 帧，来自目录：%s",
            self._frame_count,
            self._dir,
        )

    # ------------------------------------------------------------------
    # 帧获取
    # ------------------------------------------------------------------

    def get_next_frame(self) -> CameraFrame:
        """
        获取下一帧点云数据。

        返回：CameraFrame
        异常：
          RuntimeError  — 相机尚未启动
          StopIteration — loop=False 且文件已全部读取完毕
          IOError       — 文件读取失败
        """
        if not self._running:
            raise RuntimeError("[MockCamera] 相机尚未启动，请先调用 start()")

        if not self._files:
            raise FileNotFoundError(
                f"[MockCamera] 文件列表为空，目录：{self._dir}"
            )

        # 非循环模式：所有文件读完后触发 StopIteration
        if not self._loop and self._file_idx >= len(self._files):
            logger.info("[MockCamera] 所有文件已读取完毕（非循环模式），停止输出")
            raise StopIteration("MockCamera 文件已全部读取")

        path = self._files[self._file_idx]
        self._frame_count += 1

        # 推进文件索引：循环模式取模回绕，非循环模式直接递增
        if self._loop:
            self._file_idx = (self._file_idx + 1) % len(self._files)
        else:
            self._file_idx += 1

        # 读取点云
        try:
            xyz = self._read_xyz(path)
        except Exception as exc:
            logger.error(
                "[MockCamera] 读取文件失败，跳过该帧 | 文件：%s | 错误：%s",
                path,
                exc,
            )
            raise IOError(f"读取点云文件失败：{path}") from exc

        n_pts = int(xyz.shape[0])
        logger.debug(
            "[MockCamera] 帧 #%d | 文件：%s | 点数：%d | 进度：%d / %d",
            self._frame_count,
            path.name,
            n_pts,
            self._file_idx,
            len(self._files),
        )

        return CameraFrame(
            points_xyz=xyz,
            frame_id=self._frame_count,
            total_files=len(self._files),
            source_path=path,
        )

    # read_frame() 继承自基类（调用 get_next_frame）

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _reload_files(self) -> None:
        """扫描目录，按文件名排序填充 self._files。"""
        self._files = []
        if not self._dir.exists():
            logger.warning("[MockCamera] 目录不存在：%s", self._dir)
            return
        if not self._dir.is_dir():
            logger.warning("[MockCamera] 路径不是目录：%s", self._dir)
            return

        for p in sorted(self._dir.iterdir()):
            if p.is_file() and p.suffix.lower() in _SUPPORTED_SUFFIXES:
                self._files.append(p)

        logger.debug(
            "[MockCamera] 扫描目录完成：%s | 共找到 %d 个文件",
            self._dir,
            len(self._files),
        )

    @staticmethod
    def _read_xyz(path: Path) -> np.ndarray:
        """
        根据后缀读取点云，统一返回 (N, 3) float32。

        .bin — 纯 numpy 读取（线程安全，不依赖 Open3D）
        .pcd — Open3D 读取（应在非 GUI 线程中调用）
        """
        suffix = path.suffix.lower()
        if suffix == ".bin":
            # load_points_xyz_numpy 返回 float64 (N,3)，转为 float32
            xyz = load_points_xyz_numpy(path)
            return np.asarray(xyz, dtype=np.float32)
        elif suffix == ".pcd":
            pcd = load_pointcloud(path)
            return np.asarray(pcd.points, dtype=np.float32)
        else:
            raise ValueError(f"不支持的点云格式：{suffix}")

    def __repr__(self) -> str:
        return (
            f"MockCamera("
            f"dir={self._dir}, "
            f"files={len(self._files)}, "
            f"fps={self._target_fps:.1f}, "
            f"loop={self._loop}, "
            f"running={self._running}"
            f")"
        )
