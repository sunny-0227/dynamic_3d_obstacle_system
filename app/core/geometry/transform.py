"""
几何变换工具（里程碑 5）

目标：
  - 明确并统一坐标变换表示：旋转、平移、齐次变换矩阵
  - 提供点云、检测框等对象的坐标变换接口

约定（与当前工程一致）：
  - 点云使用右手系，Z 轴向上（Open3D 默认）
  - 检测框 yaw 为绕 Z 轴旋转（弧度）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


def rot_z(yaw: float) -> np.ndarray:
    """绕 Z 轴旋转矩阵 Rz(yaw)。"""
    c = float(np.cos(yaw))
    s = float(np.sin(yaw))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def make_homogeneous(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """由 R(3,3) 与 t(3,) 生成齐次矩阵 T(4,4)。"""
    RR = np.asarray(R, dtype=np.float64).reshape(3, 3)
    tt = np.asarray(t, dtype=np.float64).reshape(3)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = RR
    T[:3, 3] = tt
    return T


@dataclass(frozen=True)
class Transform:
    """
    3D 刚体变换（SE(3)）：p' = R p + t
    """

    R: np.ndarray  # (3,3)
    t: np.ndarray  # (3,)

    @staticmethod
    def identity() -> "Transform":
        return Transform(R=np.eye(3, dtype=np.float64), t=np.zeros((3,), dtype=np.float64))

    @staticmethod
    def from_rt(R: np.ndarray, t: np.ndarray) -> "Transform":
        RR = np.asarray(R, dtype=np.float64).reshape(3, 3)
        tt = np.asarray(t, dtype=np.float64).reshape(3)
        return Transform(R=RR, t=tt)

    @staticmethod
    def from_yaw_translation(yaw: float, t: np.ndarray) -> "Transform":
        return Transform.from_rt(rot_z(yaw), t)

    @staticmethod
    def from_matrix4(T: np.ndarray) -> "Transform":
        TT = np.asarray(T, dtype=np.float64).reshape(4, 4)
        return Transform(R=TT[:3, :3].copy(), t=TT[:3, 3].copy())

    def matrix4(self) -> np.ndarray:
        return make_homogeneous(self.R, self.t)

    def inverse(self) -> "Transform":
        R_inv = self.R.T
        t_inv = -R_inv @ self.t
        return Transform(R=R_inv, t=t_inv)

    def compose(self, other: "Transform") -> "Transform":
        """
        复合变换：self ∘ other
        先应用 other，再应用 self：
          p1 = other(p)
          p2 = self(p1)
        """
        R_new = self.R @ other.R
        t_new = self.R @ other.t + self.t
        return Transform(R=R_new, t=t_new)

    def apply_points(self, points: np.ndarray) -> np.ndarray:
        """
        对点云应用变换。
        points: (N,3) 或 (N,K)（只变换前 3 维 xyz）
        """
        pts = np.asarray(points)
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"points 形状不合法：{pts.shape}")
        xyz = pts[:, :3].astype(np.float64, copy=False)
        xyz2 = (xyz @ self.R.T) + self.t.reshape(1, 3)
        if pts.shape[1] == 3:
            return xyz2.astype(np.float32)
        out = pts.astype(np.float32, copy=True)
        out[:, :3] = xyz2.astype(np.float32)
        return out

    def yaw_delta_about_z(self) -> float:
        """
        若该变换主要表示“绕 Z 轴旋转”，返回其 yaw（弧度）。
        对于一般旋转矩阵，这里取其在 XY 平面的等效 yaw。
        """
        # R = [[r00 r01 ...],
        #      [r10 r11 ...],
        #      [...         ]]
        r00 = float(self.R[0, 0])
        r10 = float(self.R[1, 0])
        return float(np.arctan2(r10, r00))

    def apply_yaw(self, yaw: float) -> float:
        """将 yaw 绕 Z 的角度变换到新坐标系（近似：加上本变换的等效 yaw）。"""
        return float(yaw + self.yaw_delta_about_z())


def apply_transform_to_detection_box(
    center: np.ndarray,
    size: np.ndarray,
    yaw: float,
    T: Transform,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    对检测框参数应用坐标变换。
    - center 做 SE(3) 变换
    - size 不变（仅平移/旋转）
    - yaw 做绕 Z 的等效角度更新
    """
    c = np.asarray(center, dtype=np.float64).reshape(3)
    c2 = (T.R @ c) + T.t
    s2 = np.asarray(size, dtype=np.float32).reshape(3)
    yaw2 = T.apply_yaw(float(yaw))
    return c2.astype(np.float32), s2.astype(np.float32), float(yaw2)

