from __future__ import annotations

"""
OpenPCDetJsonDetector — 通过 WSL2 subprocess 调用 infer_to_json.py 的检测器

WSL 命令构造策略（解决 conda: command not found）：
  非交互式 bash -lc 不会自动加载 conda 初始化脚本，
  因此显式 source conda.sh 来激活 conda，再 activate 环境。

  优先尝试 ~/miniconda3，若不存在则尝试 ~/anaconda3：
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate <env> && \
    python infer_to_json.py ...

  最终 WSL 调用形式：
    wsl bash -lc "source ~/miniconda3/etc/profile.d/conda.sh && conda activate openpcdet_py310 && python ..."
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

# UI 日志回调类型
LogCallback = Callable[[str], None]


# ──────────────────────────────────────────────────────────────
# 工具：Windows 路径 → WSL /mnt/... 路径
# ──────────────────────────────────────────────────────────────

def _win_to_wsl(win_path: Path) -> str:
    """
    C:\\Users\\me\\a.bin  →  /mnt/c/Users/me/a.bin
    D:\\foo\\bar.bin      →  /mnt/d/foo/bar.bin
    已是 POSIX 路径则原样返回。
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
    x_max, y_max = float(pts[:, 0].max()), float(pts[:, 1].max())
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

    WSL conda 激活策略（解决非交互式 shell 下 conda: command not found）：
      显式 source ~/miniconda3/etc/profile.d/conda.sh（或 anaconda3），
      再 conda activate <env>，最后执行 python infer_to_json.py。

    参数说明：
        cfg_file        WSL 内模型配置路径
        ckpt_file       WSL 内权重路径
        infer_script    WSL 内 infer_to_json.py 路径
        conda_env       conda 环境名
        conda_base      conda 安装根目录的 WSL 路径（默认 ~/miniconda3，失败自动尝试 ~/anaconda3）
        ext             点云文件扩展名
        score_threshold 置信度过滤阈值
        class_names     类别名列表（1-based）
        wsl_timeout_s   WSL 子进程超时秒数
        num_boxes_fake  fallback 伪造框数
        tmp_dir         JSON 临时输出目录（Windows 路径）
        enable_wsl      False 时直接 fallback（离线调试用）
        log_callback    UI 日志回调 (msg: str) -> None
    """

    # conda.sh 搜索顺序（WSL 路径，依次尝试）
    _CONDA_SH_CANDIDATES = [
        "~/miniconda3/etc/profile.d/conda.sh",
        "~/anaconda3/etc/profile.d/conda.sh",
        "~/miniforge3/etc/profile.d/conda.sh",
    ]

    def __init__(
        self,
        cfg_file: str,
        ckpt_file: str,
        infer_script: str,
        conda_env: str = "openpcdet",
        conda_base: str = "",          # 留空则按 _CONDA_SH_CANDIDATES 自动探测
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
        self._conda_base    = conda_base.strip()   # 用户可手动指定 conda 根目录
        self._ext           = ext if ext.startswith(".") else f".{ext}"
        self._score_thr     = float(score_threshold)
        self._class_names   = list(class_names or ["car", "pedestrian", "cyclist"])
        self._timeout_s     = int(wsl_timeout_s)
        self._num_boxes_fake = int(num_boxes_fake)
        self._tmp_dir       = tmp_dir
        self._enable_wsl    = bool(enable_wsl)
        self._log_cb        = log_callback

        # 运行时缓存：首次推理时探测出正确的 conda.sh 路径
        self._resolved_conda_sh: Optional[str] = None

        if not enable_wsl:
            logger.warning("[OpenPCDetJsonDetector] enable_wsl=False，直接使用模拟检测")
            self._ui_log("⚠ OpenPCDet 未启用（enable_wsl=False），将使用模拟检测。")
        else:
            logger.info(
                "[OpenPCDetJsonDetector] 已配置 WSL 推理 | cfg=%s | ckpt=%s | env=%s",
                cfg_file, ckpt_file, conda_env,
            )
            self._ui_log(
                f"[OpenPCDet] 检测器已就绪 | 环境={conda_env} | 脚本={infer_script}"
            )

    # ------------------------------------------------------------------
    # 内部辅助：UI 日志（同步写 logger + 调用 callback）
    # ------------------------------------------------------------------

    def _ui_log(self, msg: str) -> None:
        logger.info(msg)
        if self._log_cb is not None:
            try:
                self._log_cb(msg)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # BaseDetector 接口
    # ------------------------------------------------------------------

    def _detect_impl(self, points: np.ndarray) -> List[DetectionBox]:
        pts_xyz = np.asarray(points[:, :3], dtype=np.float32)

        if not self._enable_wsl:
            return self._fallback(pts_xyz, reason="enable_wsl=False（离线调试模式）")

        # 第一步：写临时 .bin
        try:
            tmp_bin, tmp_json = self._make_tmp_paths()
            self._write_tmp_bin(pts_xyz, tmp_bin)
        except Exception as exc:
            return self._fallback(pts_xyz, reason=f"写临时 .bin 失败：{exc}")

        # 第二步：探测 conda.sh（首次推理时）
        conda_sh = self._get_conda_sh(pts_xyz)
        if conda_sh is None:
            # _get_conda_sh 内部已 fallback
            return self._fallback(
                pts_xyz,
                reason="找不到可用的 conda.sh，请在配置页设置 conda_base 或确认 conda 安装路径",
            )

        # 第三步：调用 WSL
        try:
            detections = self._run_wsl_and_parse(tmp_bin, tmp_json, pts_xyz, conda_sh)
        finally:
            # 清理临时 .bin（JSON 保留，供排查）
            try:
                if tmp_bin.exists():
                    tmp_bin.unlink()
            except Exception:
                pass

        return detections

    # ------------------------------------------------------------------
    # 探测 conda.sh 路径（在 WSL 内用 test -f 逐一尝试）
    # ------------------------------------------------------------------

    def _get_conda_sh(self, pts_xyz: np.ndarray) -> Optional[str]:
        """
        返回 WSL 内可用的 conda.sh 路径。
        首次调用时通过 wsl test -f 探测，结果缓存在 self._resolved_conda_sh。
        若用户已通过 conda_base 指定，则直接使用。
        """
        if self._resolved_conda_sh is not None:
            return self._resolved_conda_sh

        # 用户手动指定了 conda 根目录
        if self._conda_base:
            candidate = f"{self._conda_base.rstrip('/')}/etc/profile.d/conda.sh"
            self._resolved_conda_sh = candidate
            self._ui_log(f"[OpenPCDet] conda.sh（用户指定）: {candidate}")
            return candidate

        # 自动探测
        self._ui_log("[OpenPCDet] 正在探测 WSL 内 conda.sh 路径…")
        candidates = self._CONDA_SH_CANDIDATES
        for sh in candidates:
            probe_cmd = ["wsl", "bash", "-lc", f"test -f {sh} && echo YES || echo NO"]
            try:
                result = subprocess.run(
                    probe_cmd,
                    capture_output=True, text=True, timeout=10,
                )
                out = (result.stdout or "").strip()  # stdout 可能为 None，先做 None 检查
                self._ui_log(f"[OpenPCDet] 探测 {sh} → {out}")
                if out == "YES":
                    self._resolved_conda_sh = sh
                    self._ui_log(f"[OpenPCDet] 使用 conda.sh: {sh}")
                    return sh
            except Exception as e:
                self._ui_log(f"[OpenPCDet] 探测 {sh} 失败: {e}")

        # 全部失败
        self._ui_log(
            "[OpenPCDet] ⚠ 无法找到 conda.sh，已尝试路径：" + ", ".join(candidates)
        )
        self._ui_log(
            "[OpenPCDet] 提示：请在配置页的「conda_base」填写 conda 安装根目录，例如 ~/miniconda3"
        )
        return None

    # ------------------------------------------------------------------
    # 生成临时文件路径
    # ------------------------------------------------------------------

    def _make_tmp_paths(self):
        if self._tmp_dir is not None:
            base = Path(self._tmp_dir)
            base.mkdir(parents=True, exist_ok=True)
        else:
            base = Path(tempfile.gettempdir())
        ts = int(time.time() * 1000)
        return base / f"_openpcdet_input_{ts}.bin", base / f"_openpcdet_result_{ts}.json"

    # ------------------------------------------------------------------
    # 写点云为 KITTI .bin 格式（float32, 4 列：x y z intensity=0）
    # ------------------------------------------------------------------

    @staticmethod
    def _write_tmp_bin(pts_xyz: np.ndarray, out_path: Path) -> None:
        N = pts_xyz.shape[0]
        intensity = np.zeros((N, 1), dtype=np.float32)
        data = np.hstack([pts_xyz, intensity])
        data.tofile(str(out_path))
        logger.debug("[OpenPCDetJsonDetector] 写 .bin | N=%d | path=%s", N, out_path)

    # ------------------------------------------------------------------
    # 构造 WSL bash 命令
    # ------------------------------------------------------------------

    def _build_wsl_cmd(self, tmp_bin: Path, tmp_json: Path, conda_sh: str) -> str:
        """
        构造 WSL bash -lc 内部命令字符串。

        显式 source conda.sh → conda activate → python infer_to_json.py
        避免依赖 ~/.bashrc 的自动初始化（非交互式 shell 不加载）。
        """
        wsl_bin  = _win_to_wsl(tmp_bin)
        wsl_json = _win_to_wsl(tmp_json)

        cmd = (
            f"source {conda_sh} && "
            f"conda activate {self._conda_env} && "
            f"python {self._infer_script} "
            f"--cfg_file {self._cfg_file} "
            f"--ckpt {self._ckpt_file} "
            f"--data_path {wsl_bin} "
            f"--ext {self._ext} "
            f"--out_json {wsl_json}"
        )
        return cmd

    # ------------------------------------------------------------------
    # 调用 WSL 并解析结果
    # ------------------------------------------------------------------

    def _run_wsl_and_parse(
        self,
        tmp_bin: Path,
        tmp_json: Path,
        pts_xyz: np.ndarray,
        conda_sh: str,
    ) -> List[DetectionBox]:

        cmd_inner = self._build_wsl_cmd(tmp_bin, tmp_json, conda_sh)
        full_cmd  = ["wsl", "bash", "-lc", cmd_inner]

        # 打印完整调用信息到 UI 日志
        self._ui_log("[OpenPCDet] 正在调用 OpenPCDet 推理…")
        self._ui_log(f"[OpenPCDet] conda.sh  : {conda_sh}")
        self._ui_log(f"[OpenPCDet] 脚本路径  : {self._infer_script}")
        self._ui_log(f"[OpenPCDet] cfg_file  : {self._cfg_file}")
        self._ui_log(f"[OpenPCDet] ckpt_file : {self._ckpt_file}")
        self._ui_log(f"[OpenPCDet] data_path : {_win_to_wsl(tmp_bin)}")
        self._ui_log(f"[OpenPCDet] out_json  : {tmp_json}")
        self._ui_log(f"[OpenPCDet] WSL 命令  : {cmd_inner}")

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
                reason=f"WSL 子进程超时（>{self._timeout_s}s），模型加载过慢或路径有误",
            )
        except Exception as exc:
            return self._fallback(pts_xyz, reason=f"subprocess 异常：{exc}")

        elapsed_ms = (time.perf_counter() - t0) * 1000
        self._ui_log(
            f"[OpenPCDet] WSL 进程结束 | 耗时={elapsed_ms:.0f}ms | 退出码={proc.returncode}"
        )

        # 完整打印 stdout（最多 30 行）；proc.stdout 可能为 None，先做 None 检查
        raw_stdout = proc.stdout or ""
        stdout_lines = raw_stdout.strip().splitlines() if raw_stdout.strip() else []
        if stdout_lines:
            self._ui_log(f"[OpenPCDet] stdout（共 {len(stdout_lines)} 行）：")
            for line in stdout_lines[:30]:
                self._ui_log(f"  [stdout] {line}")
                logger.debug("[WSL stdout] %s", line)
        else:
            self._ui_log("[OpenPCDet] stdout: (空)")

        # 完整打印 stderr（最多 50 行）；同样做 None 检查
        raw_stderr = proc.stderr or ""
        stderr_lines = raw_stderr.strip().splitlines() if raw_stderr.strip() else []
        if stderr_lines:
            self._ui_log(f"[OpenPCDet] stderr（共 {len(stderr_lines)} 行）：")
            for line in stderr_lines[:50]:
                self._ui_log(f"  [stderr] {line}")
                logger.warning("[WSL stderr] %s", line)
        else:
            self._ui_log("[OpenPCDet] stderr: (空)")

        if proc.returncode != 0:
            reason = (
                f"WSL 退出码={proc.returncode}，耗时={elapsed_ms:.0f}ms。"
                f"stderr 末段：{raw_stderr.strip()[-400:] if raw_stderr.strip() else '(空)'}"
            )
            return self._fallback(pts_xyz, reason=reason)

        # 检查 JSON 是否生成
        if not tmp_json.exists():
            return self._fallback(
                pts_xyz,
                reason=f"JSON 不存在：{tmp_json}（WSL 返回0但未写出文件）",
            )

        self._ui_log(f"[OpenPCDet] JSON 已生成：{tmp_json}")

        try:
            return self._parse_json(tmp_json, pts_xyz)
        except Exception as exc:
            return self._fallback(pts_xyz, reason=f"解析 JSON 失败：{exc}")

    # ------------------------------------------------------------------
    # 解析 JSON → List[DetectionBox]
    # ------------------------------------------------------------------

    def _parse_json(self, json_path: Path, pts_xyz: np.ndarray) -> List[DetectionBox]:
        """
        JSON 结构（infer_to_json.py 输出）：
          [{"file": "xxx.bin", "boxes": [[x,y,z,l,w,h,yaw],...], "labels": [...], "scores": [...]}]
        """
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list) or len(data) == 0:
            self._ui_log("[OpenPCDet] OpenPCDet 检测完成，本帧无目标（JSON 空列表）")
            return []

        frame      = data[0]
        boxes_raw  = frame.get("boxes",  [])
        labels_raw = frame.get("labels", [])
        scores_raw = frame.get("scores", [])

        if not boxes_raw:
            self._ui_log("[OpenPCDet] OpenPCDet 检测完成，本帧无目标（boxes 为空）")
            return []

        detections: List[DetectionBox] = []
        for box, label_id, score in zip(boxes_raw, labels_raw, scores_raw):
            if float(score) < self._score_thr:
                continue
            box_arr = np.array(box, dtype=np.float32)
            if box_arr.shape[0] < 7:
                logger.warning("[OpenPCDetJsonDetector] 跳过非法框：%s", box)
                continue
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
            f"框数={len(detections)}（原始={len(boxes_raw)}，阈值={self._score_thr}）"
        )
        logger.info(
            "[OpenPCDetJsonDetector] ✓ 框数=%d（过滤前=%d，阈值=%.2f）",
            len(detections), len(boxes_raw), self._score_thr,
        )
        return detections

    # ------------------------------------------------------------------
    # Fallback：降级为模拟检测
    # ------------------------------------------------------------------

    def _fallback(self, pts_xyz: np.ndarray, reason: str) -> List[DetectionBox]:
        logger.warning("[OpenPCDetJsonDetector] ⚠ fallback | 原因：%s", reason)
        self._ui_log("[OpenPCDet] ⚠ OpenPCDet 调用失败，已回退到模拟检测")
        self._ui_log(f"[OpenPCDet] 失败原因：{reason[:400]}")
        results = _fake_detections(
            pts_xyz,
            class_names=self._class_names,
            num_boxes=self._num_boxes_fake,
            score_threshold=self._score_thr,
        )
        self._ui_log(f"[OpenPCDet] fallback 模拟检测 | 生成框={len(results)} 个")
        return results
