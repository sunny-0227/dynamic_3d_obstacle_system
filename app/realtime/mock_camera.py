from __future__ import annotations

"""
Mock 点云相机：从目录循环读取 .bin/.pcd，模拟实时点云流。
"""

from pathlib import Path
from typing import List

import numpy as np

from app.io.pointcloud_loader import load_pointcloud, load_points_xyz_numpy
from app.realtime.camera_interface import CameraFrame, IPointCloudCamera


class MockCamera(IPointCloudCamera):
    def __init__(self, stream_dir: Path):
        self._dir = Path(stream_dir)
        self._files: List[Path] = []
        self._idx = 0
        self._frame_id = 0
        self._running = False

    @property
    def source_name(self) -> str:
        return "Mock"

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stream_dir(self) -> Path:
        return self._dir

    def set_stream_dir(self, stream_dir: Path) -> None:
        self._dir = Path(stream_dir)
        self._reload_files()

    def start(self) -> None:
        self._reload_files()
        if not self._files:
            raise FileNotFoundError(f"目录中未找到 .bin/.pcd 文件: {self._dir}")
        self._running = True

    def stop(self) -> None:
        self._running = False

    def read_frame(self) -> CameraFrame:
        if not self._running:
            raise RuntimeError("MockCamera 尚未启动")
        if not self._files:
            raise FileNotFoundError(f"目录中未找到 .bin/.pcd 文件: {self._dir}")

        path = self._files[self._idx]
        self._idx = (self._idx + 1) % len(self._files)
        self._frame_id += 1

        if path.suffix.lower() == ".bin":
            xyz = load_points_xyz_numpy(path).astype(np.float32, copy=False)
        else:
            pcd = load_pointcloud(path)
            xyz = np.asarray(pcd.points, dtype=np.float32)

        return CameraFrame(points_xyz=xyz, frame_id=self._frame_id, source_path=path)

    def _reload_files(self) -> None:
        self._files = []
        if not self._dir.is_dir():
            return
        files = list(self._dir.iterdir())
        for p in sorted(files):
            if p.is_file() and p.suffix.lower() in {".bin", ".pcd"}:
                self._files.append(p)
        self._idx = 0
