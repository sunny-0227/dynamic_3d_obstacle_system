from __future__ import annotations

"""
OpenPCDetJsonDetector — 通过 WSL2 subprocess 调用 infer_to_json.py 的检测器

设计目标：
  - Windows 主项目不直接 import OpenPCDet / torch；
    所有推理由 WSL2 内部的 infer_to_json.py 完成，结果通过 JSON 文件传回。
  - 完全实现 BaseDetector 接口，可替换插入 DetectPipeline。
  - 内置 log_callback 机制：将关键推理步骤的日志实时发送到 UI 日志框。
  - Fallback：WSL 调用失败时自动降级为模拟检测，并在日志中明确说明。

调用链：
  Windows DetectPipeline
      └─ OpenPCDetJsonDetector.detect(points_or_path)
             ├─ log_callback("正在调用 OpenPCDet...")
             ├─ 把点云 numpy 数组写成临时 .bin 文件
             ├─ 把 .bin 路径转换为 WSL /mnt/... 路径
             ├─ subprocess: wsl bash -lc "conda activate ... && python infer_to_json.py ..."
             ├─ log_callback("WSL 命令: ...")
             ├─ 读取 JSON → List[DetectionBox]
             ├─ log_callback("OpenPCDet 检测完成，框数: N")
             └─ 失败时 → log_callback("OpenPCDet 调用失败，已回退到模拟检测") → fake

路径映射（Windows → WSL /mnt）：
  C:\\Users\\me\\data\\frame.bin  →  /mnt/c/Users/me/data/frame.bin
  D:\\foo\\bar.bin                →  /mnt/d/foo/bar.bin
"""

import json
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable, List, Optional

import numpy as np

from app.core.detector.base_detector import BaseDetector, DetectionBox
from app.utils.logger import get_logger

logger = get_logger("core.detector.openpcdet_json_detector")

# UI 日志回调类型：接收一个字符串，无返回值
LogCallback = Callable[[str], None]


# ──────────────────────────────────────────────────────────────
# 工具函数：Windows 路径 → WSL /mnt/... 路径
# ──────────────────────────────────────────────────────────────

def _win_to_wsl(win_path: Path) -> str:
    """
    把 Windows 绝对路径转换为 WSL 可识别的 /mnt/<drive>/... 形式。

    示例：
      C:\\Users\\me\\data\\a.bin  →  /mnt/c/Users/me/data/a.bin
      D:\\foo\\bar.bin            →  /mnt/d/foo/bar.bin

    若已经是 POSIX 风格（不含盘符），原样返回。
    """
    s = str(win_path)
    if len(s) >= 2 and s[1] == ":":
        drive = s[0].lower()
        rest = s[2:].replace("\\", "/").lstrip("/")
        return f"/mnt/{drive}/{rest}"
    return s.replace("\\", "/")


# ──────────────────────────────────────────────────────────────
# Fake 检测（fallback 使用）
# ──────────────────────────────────────────────────────────────

def _fake_detections(
    points_xyz: np.ndarray,
    class_names: List[str],
    num_boxes: int,
    score_threshold: float,
) -> List[DetectionBox]:
    """在点云空间范围内随机生成若干 DetectionBox，作为 fallback 占位。"""
    import random

    DEFAULT_SIZES = {
        "car":        (4.5, 2.0, 1.6),
        "pedestrian": (0.8, 0.8, 1.8),
        "cyclist":    (1.8, 0.8, 1.6),
    }

    if points_xyz.size == 0:
        return []

    pts = np.asarray(points_xyz[:, :3], dtype=np.float32)
    x_min, y_min, z_min = float(pts[:, 0].min()), float(pts[:, 1].min()), float(pts[:, 2].min())
    x_max, y_max       = float(pts[:, 0].max()), float(pts[:, 1].max())
    margin = 1.0
    cx_low  = x_min + margin if x_min + margin < x_max - margin else x_min
    cx_high = x_max - margin if x_min + margin < x_max - margin else x_max
    cy_low  = y_min + margin if y_min + margin < y_max - margin else y_min
    cy_high = y_max - margin if y_min + margin < y_max - margin else y_max

    results: List[DetectionBox] = []
    for _ in range(max(1, min(num_boxes, 5))):
        cn = random.choice(class_names)
        l, w, h = DEFAULT_SIZES.get(cn, (1.0, 1.0, 1.0))
        cx  = random.uniform(cx_low, cx_high)
        cy  = random.uniform(cy_low, cy_high)
        cz  = z_min + h / 2.0
        yaw = random.uniform(0.0, np.pi)
        score = float(np.clip(random.uniform(score_threshold, 0.99), 0.0, 1.0))
        results.append(
            DetectionBox(
                class_name=cn,
                score=score,
                center=np.array([cx, cy, cz], dtype=np.float32),
                size=np.array([l, w, h], dtype=np.float32),
                yaw=float(yaw),
            )
        )
    results.sort(key=lambda b: b.score, reverse=True)
    return results


# ──────────────────────────────────────────────────────────────
# 主检测器
# ──────────────────────────────────────────────────────────────

class OpenPCDetJsonDetector(BaseDetector):
    """
    基于 WSL2 subprocess 的 OpenPCDet 检测器。

    参数：
        cfg_file          WSL 内 OpenPCDet 模型配置文件路径（WSL 绝对路径）
        ckpt_file         WSL 内模型权重路径（WSL 绝对路径）
        infer_script      WSL 内 infer_to_json.py 的路径
        conda_env         WSL 中的 conda 环境名（默认 "openpcdet"）
        ext               点云文件扩展名（默认 ".bin"）
        score_threshold   置信度过滤阈值
        class_names       类别名称列表（1-based，与 infer_to_json 输出的 labels 对应）
        wsl_timeout_s     WSL 子进程超时时间（秒，默认 180）
        num_boxes_fake    fallback 时伪造框数
        tmp_dir           JSON 临时文件输出目录（Windows 路径，None 则用系统临时目录）
        enable_wsl        False 时直接 fallback，方便离线开发调试
        log_callback      可选回调函数 (msg: str) -> None，
                          用于把关键推理日志实时发送到 UI 日志框
    """

    def __init__(
        self,
        cfg_file: str,
        ckpt_file: str,
        infer_script: str,
        conda_env: str = "openpcdet",
        ext: str = ".bin",
        score_threshold: float = 0.1,
        class_names: Optional[List[str]] = None,
        wsl_timeout_s: int = 180,
        num_boxes_fake: int = 3,
        tmp_dir: Optional[Path] = None,
        enable_wsl: bool = True,
        log_callback: Optional[LogCallback] = None,
    ) -> None:
        super().__init__()
        self._cfg_file      = cfg_file
        self._ckpt_file     = ckpt_file
        self._infer_script  = infer_script
        self._conda_env     = conda_env
        self._ext           = ext if ext.startswith(".") else f".{ext}"
        self._score_thr     = float(score_threshold)
        self._class_names   = list(class_names or ["car", "pedestrian", "cyclist"])
        self._timeout_s     = int(wsl_timeout_s)
        self._num_boxes_fake = int(num_boxes_fake)
        self._tmp_dir       = tmp_dir
        self._enable_wsl    = bool(enable_wsl)
        self._log_cb        = log_callback  # UI 日志回调

        if not enable_wsl:
            logger.warning("[OpenPCDetJsonDetector] enable_wsl=False，直接使用模拟检测（fallback 模式）")
            self._ui_log("⚠ OpenPCDet 未启用（enable_wsl=False），将使用模拟检测。")
        else:
            logger.info(
                "[OpenPCDetJsonDetector] 已配置 WSL 推理 | cfg=%s | ckpt=%s | conda_env=%s",
                cfg_file, ckpt_file, conda_env,
            )
            self._ui_log(f"[OpenPCDet] 检测器已就绪 | 环境={conda_env} | 脚本={infer_script}")

    # ------------------------------------------------------------------
    # 内部辅助：发送 UI 日志（同时写 logger，保证文件日志不丢失）
    # ------------------------------------------------------------------

    def _ui_log(self, msg: str) -> None:
        """同时写 logger（文件）和回调（UI 日志框）。"""
        logger.info(msg)
        if self._log_cb is not None:
            try:
                self._log_cb(msg)
            except Exception:
                pass  # UI 回调异常不能影响推理流程

    # ------------------------------------------------------------------
    # BaseDetector 接口
    # ------------------------------------------------------------------

    def _detect_impl(self, points: np.ndarray) -> List[DetectionBox]:
        """
        对 (N, K) 点云执行检测。
        先将点云写入临时 .bin 文件，再通过 WSL subprocess 调用 infer_to_json.py。
        """
        pts_xyz = np.asarray(points[:, :3], dtype=np.float32)

        if not self._enable_wsl:
            return self._fallback(pts_xyz, reason="enable_wsl=False（离线调试模式）")

        # ── 第一步：写临时 .bin 文件 ─────────────────────────────
        try:
            tmp_bin, tmp_json = self._make_tmp_paths()
            self._write_tmp_bin(pts_xyz, tmp_bin)
        except Exception as exc:
            return self._fallback(pts_xyz, reason=f"写临时 .bin 失败：{exc}")

        # ── 第二步：调用 WSL + 解析结果 ──────────────────────────
        try:
            detections = self._run_wsl_and_parse(tmp_bin, tmp_json, pts_xyz)
        finally:
            # 无论成功失败，都清理临时 .bin（result JSON 由用户目录管理，保留以便排查）
            for p in (tmp_bin,):
                try:
                    if p.exists():
                        p.unlink()
                except Exception:
                    pass

        return detections

    # ------------------------------------------------------------------
    # 内部：生成临时文件路径
    # ------------------------------------------------------------------

    def _make_tmp_paths(self):
        """返回 (tmp_bin_path, tmp_json_path)，均为 Windows Path 对象。"""
        if self._tmp_dir is not None:
            base = Path(self._tmp_dir)
            base.mkdir(parents=True, exist_ok=True)
        else:
            base = Path(tempfile.gettempdir())

        ts = int(time.time() * 1000)
        tmp_bin  = base / f"_openpcdet_input_{ts}.bin"
        tmp_json = base / f"_openpcdet_result_{ts}.json"
        return tmp_bin, tmp_json

    # ------------------------------------------------------------------
    # 内部：写点云为 KITTI .bin 格式（float32, 4 列：x y z intensity=0）
    # ------------------------------------------------------------------

    @staticmethod
    def _write_tmp_bin(pts_xyz: np.ndarray, out_path: Path) -> None:
        """把 (N, 3) float32 点云写成 KITTI .bin 格式（4 列，intensity=0）。"""
        N = pts_xyz.shape[0]
        intensity = np.zeros((N, 1), dtype=np.float32)
        data = np.hstack([pts_xyz, intensity])  # (N, 4)
        data.tofile(str(out_path))
        logger.debug("[OpenPCDetJsonDetector] 写临时 .bin | 点数=%d | 路径=%s", N, out_path)

    # ------------------------------------------------------------------
    # 内部：构造 WSL 命令并执行
    # ------------------------------------------------------------------

    def _build_wsl_cmd(self, tmp_bin: Path, tmp_json: Path) -> str:
        """
        构造完整的 WSL bash 命令字符串。
        使用 conda run 代替 conda activate，兼容非交互式 bash。
        """
        wsl_bin  = _win_to_wsl(tmp_bin)
        wsl_json = _win_to_wsl(tmp_json)

        # conda run -n <env> python ... 比 conda activate && python 更适合 subprocess
        parts = [
            f"conda run -n {self._conda_env}",
            f"python {self._infer_script}",
            f"--cfg_file {self._cfg_file}",
            f"--ckpt {self._ckpt_file}",
            f"--data_path {wsl_bin}",
            f"--ext {self._ext}",
            f"--out_json {wsl_json}",
        ]
        return " ".join(parts)

    def _run_wsl_and_parse(
        self,
        tmp_bin: Path,
        tmp_json: Path,
        pts_xyz: np.ndarray,
    ) -> List[DetectionBox]:
        """
        调用 WSL 子进程执行推理，读取 JSON，解析为 DetectionBox 列表。
        遇到任何错误均 fallback 到 fake_detector。
        """
        cmd_inner = self._build_wsl_cmd(tmp_bin, tmp_json)
        full_cmd  = ["wsl", "bash", "-lc", cmd_inner]

        # 记录 WSL 命令到 UI 日志
        self._ui_log(f"[OpenPCDet] 正在调用 OpenPCDet 推理…")
        self._ui_log(f"[OpenPCDet] WSL 命令：{cmd_inner}")
        self._ui_log(f"[OpenPCDet] JSON 输出路径：{tmp_json}")

        logger.info("[OpenPCDetJsonDetector] 启动 WSL | cmd: %s", cmd_inner)

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                full_cmd,
                capture_output=True,
                text=True,
                timeout=self._timeout_s,
            )
        except FileNotFoundError:
            return self._fallback(
                pts_xyz,
                reason="找不到 wsl 可执行文件，请确认 WSL2 已安装并加入 PATH",
            )
        except subprocess.TimeoutExpired:
            return self._fallback(
                pts_xyz,
                reason=f"WSL 子进程超时（>{self._timeout_s}s），可能是模型加载过慢，"
                       f"可在配置页增大 wsl_timeout_s",
            )
        except Exception as exc:
            return self._fallback(pts_xyz, reason=f"subprocess 异常：{exc}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._ui_log(f"[OpenPCDet] WSL 子进程结束 | 耗时={elapsed_ms:.0f}ms | 退出码={proc.returncode}")

        # 把 WSL stdout 也打到 UI 日志（截取前 500 字符，避免刷屏）
        if proc.stdout.strip():
            for line in proc.stdout.strip().splitlines()[:20]:
                self._ui_log(f"  [WSL] {line}")
                logger.debug("[WSL stdout] %s", line)

        if proc.returncode != 0:
            stderr_brief = proc.stderr.strip()[-800:] if proc.stderr else "(无 stderr)"
            self._ui_log(f"[OpenPCDet] ⚠ WSL 退出码={proc.returncode}")
            self._ui_log(f"[OpenPCDet] stderr: {stderr_brief[-300:]}")
            return self._fallback(
                pts_xyz,
                reason=(
                    f"WSL 子进程退出码={proc.returncode}，耗时={elapsed_ms:.0f}ms\n"
                    f"stderr（末 800 字符）：{stderr_brief}"
                ),
            )

        logger.info("[OpenPCDetJsonDetector] WSL 推理完成 | 耗时=%.0f ms", elapsed_ms)

        # 检查 JSON 是否存在
        if not tmp_json.exists():
            return self._fallback(
                pts_xyz,
                reason=f"JSON 输出文件不存在：{tmp_json}（WSL 返回0但未写出文件）",
            )

        self._ui_log(f"[OpenPCDet] JSON 已生成：{tmp_json}")

        # 解析 JSON
        try:
            return self._parse_json(tmp_json, pts_xyz)
        except Exception as exc:
            return self._fallback(pts_xyz, reason=f"解析 JSON 失败：{exc}")

    # ------------------------------------------------------------------
    # 内部：解析 infer_to_json.py 输出的 JSON
    # ------------------------------------------------------------------

    def _parse_json(self, json_path: Path, pts_xyz: np.ndarray) -> List[DetectionBox]:
        """
        读取 infer_to_json.py 生成的 JSON，转换为 List[DetectionBox]。

        JSON 结构（列表，每个元素对应一帧；此处只推理了单帧）：
          [
            {
              "file":   "xxx.bin",
              "boxes":  [[x,y,z,l,w,h,yaw], ...],
              "labels": [1, 2, ...],     # 1-based
              "scores": [0.92, ...]
            }
          ]
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # infer_to_json 输出是列表，单帧推理取第一个元素
        if not isinstance(data, list) or len(data) == 0:
            logger.warning("[OpenPCDetJsonDetector] JSON 为空列表，视为无检测结果（非异常）")
            self._ui_log("[OpenPCDet] OpenPCDet 检测完成，本帧无目标检出（JSON 为空列表）")
            return []

        frame = data[0]
        boxes_raw  = frame.get("boxes",  [])
        labels_raw = frame.get("labels", [])
        scores_raw = frame.get("scores", [])

        if not boxes_raw:
            self._ui_log("[OpenPCDet] OpenPCDet 检测完成，本帧无目标检出（boxes 为空）")
            return []

        detections: List[DetectionBox] = []
        for box, label_id, score in zip(boxes_raw, labels_raw, scores_raw):
            if float(score) < self._score_thr:
                continue  # 过滤低置信度

            box_arr = np.array(box, dtype=np.float32)
            if box_arr.shape[0] < 7:
                logger.warning("[OpenPCDetJsonDetector] 跳过格式不合法的框：%s", box)
                continue

            # label_id 为 1-based
            idx = int(label_id) - 1
            cn  = self._class_names[idx] if 0 <= idx < len(self._class_names) else f"class_{label_id}"

            detections.append(
                DetectionBox(
                    class_name=cn,
                    score=float(score),
                    center=box_arr[:3].copy(),
                    size=box_arr[3:6].copy(),
                    yaw=float(box_arr[6]),
                )
            )

        detections.sort(key=lambda b: b.score, reverse=True)

        self._ui_log(
            f"[OpenPCDet] OpenPCDet 检测完成 ✓ | "
            f"检测框={len(detections)} 个（原始={len(boxes_raw)}，阈值={self._score_thr}）"
        )
        logger.info(
            "[OpenPCDetJsonDetector] OpenPCDet 真实检测 ✓ | 框数=%d（过滤前=%d，阈值=%.2f）",
            len(detections), len(boxes_raw), self._score_thr,
        )
        return detections

    # ------------------------------------------------------------------
    # Fallback：降级为模拟检测
    # ------------------------------------------------------------------

    def _fallback(self, pts_xyz: np.ndarray, reason: str) -> List[DetectionBox]:
        """记录 fallback 原因，改用模拟检测，并在 UI 日志中明确标注。"""
        logger.warning(
            "[OpenPCDetJsonDetector] ⚠ WSL 推理失败，切换至 fallback 模拟检测\n原因：%s",
            reason,
        )
        self._ui_log(f"[OpenPCDet] ⚠ OpenPCDet 调用失败，已回退到模拟检测")
        self._ui_log(f"[OpenPCDet] 失败原因：{reason[:300]}")

        results = _fake_detections(
            pts_xyz,
            class_names=self._class_names,
            num_boxes=self._num_boxes_fake,
            score_threshold=self._score_thr,
        )
        self._ui_log(f"[OpenPCDet] fallback 模拟检测完成 | 生成框={len(results)} 个")
        return results
