"""
点云加载模块
支持读取 .bin（KITTI格式）和 .pcd 格式的点云文件。
.bin 默认按 float32 读取，兼容 [N,4] 或 [N,5]，至少提取 xyz 坐标。
.pcd 使用 Open3D 读取。
"""

from pathlib import Path

import numpy as np
import open3d as o3d

from app.utils.logger import get_logger

logger = get_logger("io.pointcloud_loader")


def load_pointcloud(file_path: str | Path) -> o3d.geometry.PointCloud:
    """
    根据文件扩展名自动选择加载方式，返回 Open3D PointCloud 对象。

    参数：
        file_path: 点云文件路径（.bin 或 .pcd）

    返回：
        open3d.geometry.PointCloud

    异常：
        ValueError: 文件格式不受支持
        FileNotFoundError: 文件不存在
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"点云文件不存在: {path}")

    suffix = path.suffix.lower()

    if suffix == ".bin":
        return _load_bin(path)
    elif suffix == ".pcd":
        return _load_pcd(path)
    else:
        raise ValueError(f"不支持的点云格式: {suffix}，仅支持 .bin 和 .pcd")


def _load_bin(path: Path) -> o3d.geometry.PointCloud:
    """
    读取 KITTI 风格的 .bin 二进制点云文件。
    数据按 float32 存储，列数为 4（x,y,z,intensity）或 5（x,y,z,intensity,ring）。
    仅提取前三列 xyz 作为点坐标。

    列数检测策略：
        优先尝试 4 列，再尝试 5 列。当 raw.size 同时能被 4 和 5 整除时（如 20 个
        float 元素），固定选择 4 列，符合 KITTI 标准格式约定。
    """
    logger.info("加载 .bin 文件: %s", path)

    # 读取原始二进制数据为 float32 一维数组
    raw = np.fromfile(str(path), dtype=np.float32)

    # 自动判断列数：先尝试 4 列，再尝试 5 列
    for cols in (4, 5):
        if raw.size % cols == 0:
            points_all = raw.reshape(-1, cols)
            logger.debug("解析列数: %d，点数: %d", cols, points_all.shape[0])
            break
    else:
        # 无法整除时，截断为 4 列的整倍数
        cols = 4
        trim = (raw.size // cols) * cols
        points_all = raw[:trim].reshape(-1, cols)
        logger.warning(
            "数据大小 %d 无法整除 4 或 5，已截断为 %d 个点", raw.size, points_all.shape[0]
        )

    # 提取 xyz 三列
    xyz = points_all[:, :3]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))

    logger.info("成功加载 .bin 点云，共 %d 个点", len(pcd.points))
    return pcd


def _load_pcd(path: Path) -> o3d.geometry.PointCloud:
    """
    使用 Open3D 读取 .pcd 格式点云文件。
    """
    logger.info("加载 .pcd 文件: %s", path)

    pcd = o3d.io.read_point_cloud(str(path))

    if len(pcd.points) == 0:
        raise ValueError(f"读取到空点云，请检查文件: {path}")

    logger.info("成功加载 .pcd 点云，共 %d 个点", len(pcd.points))
    return pcd


def get_points_as_numpy(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """
    将 Open3D PointCloud 的点坐标转换为 numpy 数组。

    返回：
        shape (N, 3) 的 float64 数组
    """
    return np.asarray(pcd.points)
