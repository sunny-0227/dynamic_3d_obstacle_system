"""
Intel RealSense 深度相机接入模块

实现 IPointCloudCamera 统一接口，将深度帧通过内参去投影为三维点云，
可与 MockCamera 无缝替换。依赖 pyrealsense2（懒导入，未安装不影响离线模式）。
"""

from __future__ import annotations

import numpy as np

from app.realtime.camera_interface import CameraFrame, CameraInfo, IPointCloudCamera
from app.utils.logger import get_logger

logger = get_logger("realtime.realsense_camera")


# -------------------------------------------------------------------
# pyrealsense2 懒导入：没有安装时仅在 start() 阶段报错，不影响离线模式启动
# -------------------------------------------------------------------
def _import_rs2():
    """延迟导入 pyrealsense2，避免未安装 SDK 时影响离线模式启动。"""
    try:
        import pyrealsense2 as rs  # noqa: F401
        return rs
    except ImportError as exc:
        raise ImportError(
            "未检测到 pyrealsense2 模块。\n"
            "请先安装：pip install pyrealsense2\n"
            "若已安装仍报错，请确认 Python 版本与 SDK 版本匹配。"
        ) from exc


# -------------------------------------------------------------------
# 默认配置常量
# -------------------------------------------------------------------
# 分辨率从 640x480 改为 424x240：点数从 ~21万降到 ~5万，处理速度大幅提升
# D435 支持 424x240 @ 6/15/30/60/90fps，性能演示时推荐 424x240 @ 30fps
_DEFAULT_WIDTH: int = 424          # 深度流分辨率宽
_DEFAULT_HEIGHT: int = 240         # 深度流分辨率高
_DEFAULT_FPS: int = 30             # 流帧率（相机硬件支持值：6/15/30/60/90）
_DEFAULT_TIMEOUT_MS: int = 5000    # wait_for_frames 超时（毫秒）
_MIN_DEPTH_M: float = 0.3          # 最小有效深度（米），过滤过近噪点（0.3m 以内噪声大）
_MAX_DEPTH_M: float = 4.0          # 最大有效深度（米），超出后精度下降且点数增多


class RealSenseCamera(IPointCloudCamera):
    """
    Intel RealSense 深度相机，遵循 IPointCloudCamera 统一接口。

    参数说明：
      width         — 深度流分辨率宽（像素），默认 640
      height        — 深度流分辨率高（像素），默认 480
      fps           — 相机采集帧率，默认 30；可选 6/15/30/60（根据型号）
      serial_number — 设备序列号（字符串），空串表示使用第一台已连接设备
      align_to_color— 是否将深度帧对齐到彩色帧坐标系，默认 False（纯深度模式更快）
      min_depth_m   — 有效深度下限（米），默认 0.1
      max_depth_m   — 有效深度上限（米），默认 10.0
      timeout_ms    — 等待帧超时时间（毫秒），默认 5000
      name          — 显示名称，默认 "RealSense"；连接成功后会追加型号信息

    线程安全说明：
      - start() / stop() 应在同一线程调用（推荐主线程或 _RealtimeThread 调用线程）
      - get_next_frame() 在 _RealtimeThread 工作线程中调用，是安全的
      - pyrealsense2 的 pipeline.wait_for_frames() 内部有锁，跨线程访问安全
    """

    def __init__(
        self,
        *,
        width: int = _DEFAULT_WIDTH,
        height: int = _DEFAULT_HEIGHT,
        fps: int = _DEFAULT_FPS,
        serial_number: str = "",
        align_to_color: bool = False,
        min_depth_m: float = _MIN_DEPTH_M,
        max_depth_m: float = _MAX_DEPTH_M,
        timeout_ms: int = _DEFAULT_TIMEOUT_MS,
        name: str = "RealSense",
    ) -> None:
        self._width = int(width)
        self._height = int(height)
        self._fps = int(fps)
        self._serial = str(serial_number).strip()
        self._align_to_color = bool(align_to_color)
        self._min_depth_m = float(min_depth_m)
        self._max_depth_m = float(max_depth_m)
        self._timeout_ms = int(timeout_ms)
        self._name = str(name)

        # pyrealsense2 对象（start() 后初始化）
        self._rs = None          # pyrealsense2 模块引用
        self._pipeline = None    # rs.pipeline
        self._profile = None     # rs.pipeline_profile
        self._intrinsics = None  # 深度流内参（rs.intrinsics）
        self._align = None       # rs.align（可选）
        self._pc = None          # rs.pointcloud（SDK 内置点云计算器，备用）
        self._depth_scale: float = 0.001  # 深度缩放比例（start() 后从硬件读取并缓存）

        # 运行状态
        self._running: bool = False
        self._frame_count: int = 0

    # ------------------------------------------------------------------
    # IPointCloudCamera 必须实现的抽象属性
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
            current_index=-1,   # 真实相机无文件索引概念
            total_files=0,      # 真实相机无文件数概念
        )

    # ------------------------------------------------------------------
    # 额外只读属性（RealSense 专属）
    # ------------------------------------------------------------------

    @property
    def frame_count(self) -> int:
        """已输出的帧数（自最近一次 start() 起累计）。"""
        return self._frame_count

    @property
    def depth_intrinsics(self):
        """深度流内参（rs.intrinsics），start() 后有效；未启动时返回 None。"""
        return self._intrinsics

    @property
    def device_serial(self) -> str:
        """实际连接的设备序列号（start() 后从硬件读取）。"""
        if self._profile is None:
            return self._serial or "（未连接）"
        try:
            return str(
                self._profile.get_device()
                .get_info(self._rs.camera_info.serial_number)
            )
        except Exception:
            return self._serial or "（未知）"

    # ------------------------------------------------------------------
    # 生命周期：start / stop
    # ------------------------------------------------------------------

    def start(self) -> None:
        """
        初始化并启动 RealSense 流管线。

        流程：
          1. 延迟导入 pyrealsense2
          2. 创建 pipeline 并配置深度流（可选彩色流用于对齐）
          3. 若指定了 serial_number 则优先连接对应设备
          4. 启动流，读取内参，重置帧计数

        异常：
          ImportError   — pyrealsense2 未安装
          RuntimeError  — 未找到相机设备 / 启动失败
        """
        if self._running:
            logger.warning("[RealSenseCamera] 已处于运行状态，忽略重复 start() 调用")
            return

        # 步骤 1：延迟导入 SDK
        self._rs = _import_rs2()
        rs = self._rs

        logger.info(
            "[RealSenseCamera] 初始化相机 | 分辨率：%dx%d | FPS：%d | 序列号：%s",
            self._width, self._height, self._fps,
            self._serial if self._serial else "（自动选择第一台）",
        )

        # 步骤 2：检查是否有可用设备
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            raise RuntimeError(
                "未找到任何 Intel RealSense 设备。\n"
                "请检查：\n"
                "  1. 相机是否已通过 USB 3.0 连接\n"
                "  2. Intel RealSense SDK 驱动是否已安装\n"
                "  3. Windows 设备管理器中是否显示相机"
            )

        # 步骤 3：配置流
        self._pipeline = rs.pipeline()
        config = rs.config()

        if self._serial:
            # 指定序列号时，只连接该设备
            config.enable_device(self._serial)

        # 深度流（始终开启）
        config.enable_stream(
            rs.stream.depth,
            self._width,
            self._height,
            rs.format.z16,
            self._fps,
        )

        # 彩色流（仅在需要对齐时开启，避免不必要的带宽占用）
        if self._align_to_color:
            config.enable_stream(
                rs.stream.color,
                self._width,
                self._height,
                rs.format.rgb8,
                self._fps,
            )

        # 步骤 4：启动流管线
        try:
            self._profile = self._pipeline.start(config)
        except Exception as exc:
            self._pipeline = None
            self._profile = None
            raise RuntimeError(
                f"RealSense 流管线启动失败：{exc}\n"
                f"请确认相机型号支持 {self._width}x{self._height} @ {self._fps}fps 的深度流。"
            ) from exc

        # 步骤 5-8：pipeline 已经启动，后续任何失败都必须先 stop() 再向上抛异常，
        # 否则 SDK pipeline 对象处于"已启动"状态，下次 start() 时会报设备占用。
        try:
            # 步骤 5：读取深度流内参（用于去投影）并缓存深度缩放比例
            depth_stream = self._profile.get_stream(rs.stream.depth)
            self._intrinsics = depth_stream.as_video_stream_profile().get_intrinsics()

            # 缓存 depth_scale，避免每帧都通过 SDK 查询（减少跨语言调用开销）
            depth_sensor = self._profile.get_device().first_depth_sensor()
            self._depth_scale = float(depth_sensor.get_depth_scale())
            logger.debug("[RealSenseCamera] depth_scale = %.6f m/unit", self._depth_scale)

            # 步骤 6：创建对齐器（可选）
            if self._align_to_color:
                self._align = rs.align(rs.stream.color)

            # 步骤 7：读设备信息，更新 name
            try:
                dev = self._profile.get_device()
                model = dev.get_info(rs.camera_info.name)
                serial = dev.get_info(rs.camera_info.serial_number)
                self._name = f"RealSense {model} [{serial}]"
            except Exception:
                pass  # 读取型号失败时保持默认名称，不中断启动流程

            # 步骤 8：预热：丢弃前几帧（相机自动曝光需要几帧收敛，否则初始帧深度偏暗）
            warmup_frames = 5
            logger.info("[RealSenseCamera] 预热中，丢弃前 %d 帧…", warmup_frames)
            for _ in range(warmup_frames):
                try:
                    self._pipeline.wait_for_frames(timeout_ms=self._timeout_ms)
                except Exception:
                    break  # 预热失败不阻断启动，正式读帧时再处理

        except Exception as exc:
            # 步骤 5-8 中出现异常：先停止 pipeline 释放硬件资源，再向上抛出
            logger.error("[RealSenseCamera] 启动后初始化失败，正在释放资源：%s", exc)
            try:
                self._pipeline.stop()
            except Exception:
                pass
            self._pipeline = None
            self._profile = None
            self._intrinsics = None
            self._align = None
            raise RuntimeError(f"RealSense 初始化失败（已释放资源）：{exc}") from exc

        self._frame_count = 0
        self._running = True

        logger.info(
            "[RealSenseCamera] 启动成功 | 型号：%s | "
            "内参：fx=%.1f fy=%.1f cx=%.1f cy=%.1f | depth_scale=%.6f",
            self._name,
            self._intrinsics.fx,
            self._intrinsics.fy,
            self._intrinsics.ppx,
            self._intrinsics.ppy,
            self._depth_scale,
        )

    def stop(self) -> None:
        """
        停止 RealSense 流管线并释放所有资源（幂等）。

        注意：stop() 后可再次调用 start() 重新启动相机。
        """
        if not self._running:
            return

        self._running = False

        if self._pipeline is not None:
            try:
                self._pipeline.stop()
                logger.info(
                    "[RealSenseCamera] 已停止 | 共输出 %d 帧 | 型号：%s",
                    self._frame_count,
                    self._name,
                )
            except Exception as exc:
                logger.warning("[RealSenseCamera] stop() 时出现异常（已忽略）：%s", exc)

        # 释放所有 SDK 对象引用，允许 Python GC 回收
        self._pipeline = None
        self._profile = None
        self._intrinsics = None
        self._align = None
        self._pc = None

    # ------------------------------------------------------------------
    # 帧获取
    # ------------------------------------------------------------------

    def get_next_frame(self) -> CameraFrame:
        """
        从 RealSense 相机获取下一帧，转换为点云并返回 CameraFrame。

        处理流程：
          1. wait_for_frames()     — SDK 阻塞等待下一组帧（深度+可选彩色）
          2. 可选 align()          — 对齐深度帧到彩色坐标系
          3. 深度帧 → xyz 点云     — 利用内参逐像素去投影
          4. 深度过滤               — 移除超出 [min_depth, max_depth] 范围的点
          5. 封装为 CameraFrame 返回

        返回：CameraFrame（points_xyz: (N,3) float32，timestamp 来自硬件时间戳）
        异常：
          RuntimeError — 相机尚未启动
          IOError      — wait_for_frames 超时或读取失败
        """
        if not self._running:
            raise RuntimeError("[RealSenseCamera] 相机尚未启动，请先调用 start()")

        rs = self._rs

        # 等待下一帧集合（阻塞，超时抛出异常）
        try:
            frameset = self._pipeline.wait_for_frames(timeout_ms=self._timeout_ms)
        except Exception as exc:
            raise IOError(
                f"[RealSenseCamera] 等待帧超时或失败（超时设置：{self._timeout_ms}ms）：{exc}"
            ) from exc

        # 可选：对齐深度帧到彩色坐标系
        if self._align is not None:
            frameset = self._align.process(frameset)

        # 提取深度帧
        depth_frame = frameset.get_depth_frame()
        if not depth_frame:
            raise IOError("[RealSenseCamera] 未获取到有效深度帧")

        # 获取硬件时间戳（毫秒 → 秒）
        timestamp_sec: float = depth_frame.get_timestamp() / 1000.0

        self._frame_count += 1

        # 深度帧 → 三维点云
        xyz = self._depth_frame_to_xyz(depth_frame)

        n_pts = int(xyz.shape[0])
        logger.debug(
            "[RealSenseCamera] 帧 #%d | 点数：%d | 时间戳：%.3f s",
            self._frame_count,
            n_pts,
            timestamp_sec,
        )

        return CameraFrame(
            points_xyz=xyz,
            frame_id=self._frame_count,
            total_files=0,          # 真实相机无文件数概念，填 0
            source_path=None,       # 真实相机无文件路径
            timestamp=timestamp_sec,
        )

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _depth_frame_to_xyz(self, depth_frame) -> np.ndarray:
        """
        将 RealSense 深度帧转换为 (N, 3) float32 点云。

        实现方式：纯 numpy 手动去投影（比 rs.pointcloud API 更灵活，可过滤深度范围）

        原理（针孔相机模型）：
          X = (u - cx) * Z / fx
          Y = (v - cy) * Z / fy
          Z = depth_value * depth_scale   （单位：米）

        参数：
          depth_frame — pyrealsense2.depth_frame 对象

        返回：
          shape (N, 3) float32 的 xyz 数组，已过滤无效深度
        """
        intr = self._intrinsics

        # 深度缩放比例已在 start() 时缓存，避免每帧都查询 SDK
        depth_scale: float = self._depth_scale

        # 强制拷贝深度数据：SDK 的 get_data() 返回底层内存视图，
        # 下一帧到来时该内存可能被 SDK 回收，必须 copy 后才能安全使用
        depth_image = np.array(depth_frame.get_data(), dtype=np.uint16)
        H, W = depth_image.shape

        # 转换为实际深度（米），float32
        depth_m = depth_image.astype(np.float32) * depth_scale

        # 构建像素坐标网格
        # u = 列索引（x 方向），v = 行索引（y 方向）
        u_coords = np.arange(W, dtype=np.float32)
        v_coords = np.arange(H, dtype=np.float32)
        uu, vv = np.meshgrid(u_coords, v_coords)  # 形状均为 (H, W)

        # 去投影：利用针孔相机内参将像素坐标 + 深度 → 三维坐标
        X = (uu - intr.ppx) * depth_m / intr.fx
        Y = (vv - intr.ppy) * depth_m / intr.fy
        Z = depth_m

        # 展平为 (H*W, 3)
        xyz_all = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)

        # 过滤无效点：
        #   - depth == 0 表示该像素无深度数据
        #   - 深度超出有效范围的点（噪声、镜面反射等）
        z_col = xyz_all[:, 2]
        valid_mask = (z_col >= self._min_depth_m) & (z_col <= self._max_depth_m)
        xyz_valid = xyz_all[valid_mask]

        return xyz_valid

    def _depth_frame_to_xyz_sdk(self, depth_frame) -> np.ndarray:
        """
        备用方案：使用 SDK 内置 rs.pointcloud 计算点云。

        优点：精度与 SDK 完全一致
        缺点：无法直接在 numpy 层做深度过滤，需额外处理
        保留此方法供调试对比使用，不在主流程中调用。
        """
        rs = self._rs
        if self._pc is None:
            self._pc = rs.pointcloud()

        points = self._pc.calculate(depth_frame)
        vertices = np.asarray(points.get_vertices(), dtype=np.float32)
        # vertices 形状为 (H*W,)，每个元素是 (x, y, z) 结构体
        xyz = vertices.view(np.float32).reshape(-1, 3)

        # 过滤 Z=0（无效点）
        valid = xyz[:, 2] > 0
        return xyz[valid].copy()

    def __repr__(self) -> str:
        return (
            f"RealSenseCamera("
            f"name={self._name!r}, "
            f"res={self._width}x{self._height}, "
            f"fps={self._fps}, "
            f"running={self._running}"
            f")"
        )
