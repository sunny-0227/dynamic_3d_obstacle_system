"""
scripts/experiments/eval_detection.py

检测精度评估模块。

通过 subprocess 调用 WSL 中 OpenPCDet 官方 tools/test.py，
对比预训练模型与自训练模型，解析官方日志中的 mAP / NDS 等指标。
若解析失败，保留原始日志并在 csv 中标注原因。
"""

from __future__ import annotations

import argparse
import re
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# WSL conda.sh 探测（与主项目 openpcdet_json_detector.py 相同逻辑）
# ---------------------------------------------------------------------------

_CONDA_SH_CANDIDATES = [
    "~/miniconda3/etc/profile.d/conda.sh",
    "~/anaconda3/etc/profile.d/conda.sh",
    "~/miniforge3/etc/profile.d/conda.sh",
    "/opt/conda/etc/profile.d/conda.sh",
    "/usr/local/conda/etc/profile.d/conda.sh",
]

# 缓存探测结果，避免每次调用都重新探测
_cached_conda_sh: Optional[str] = None
_conda_sh_probed: bool = False


def _detect_conda_sh() -> Optional[str]:
    """
    通过 wsl test -f 逐一探测候选 conda.sh 路径是否存在。
    找到第一个存在的路径后缓存并返回，全部失败返回 None。
    """
    global _cached_conda_sh, _conda_sh_probed
    if _conda_sh_probed:
        return _cached_conda_sh

    _conda_sh_probed = True
    for sh in _CONDA_SH_CANDIDATES:
        try:
            # 用 wsl -e bash -c 探测（不加 -l，避免 login shell 副作用）
            probe = subprocess.run(
                ["wsl", "-e", "bash", "-c", f"test -f {sh} && echo YES || echo NO"],
                capture_output=True, text=True,
                encoding="utf-8", errors="replace",
                timeout=10,
            )
            if (probe.stdout or "").strip() == "YES":
                print(f"[eval_detection] 找到 conda.sh: {sh}")
                _cached_conda_sh = sh
                return sh
        except Exception as e:
            print(f"[eval_detection] 探测 {sh} 失败: {e}")

    return None


# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------

def win_to_wsl(win_path: str) -> str:
    """
    将 Windows 路径转换为 WSL 路径。
    例：D:/sunny/foo/bar.pth  →  /mnt/d/sunny/foo/bar.pth
    若已是 /mnt/... 格式则原样返回。
    """
    p = win_path.strip().replace("\\", "/")
    if p.startswith("/"):
        return p
    # 匹配盘符 D:/...
    m = re.match(r"^([A-Za-z]):/(.*)$", p)
    if m:
        drive = m.group(1).lower()
        rest  = m.group(2)
        return f"/mnt/{drive}/{rest}"
    return p  # 无法识别则原样返回


def file_size_mb(win_path: str) -> float:
    """返回文件大小（MB），文件不存在返回 0.0。"""
    p = Path(win_path)
    if p.exists():
        return round(p.stat().st_size / 1024 / 1024, 2)
    return 0.0


# ---------------------------------------------------------------------------
# 日志解析
# ---------------------------------------------------------------------------

def parse_metrics(log_text: str) -> Dict[str, Optional[float]]:
    """
    从 OpenPCDet test.py 的 stdout/stderr 中提取常用指标。
    支持 nuScenes / KITTI 两种格式。
    返回字典：mAP, NDS, mATE, mASE, mAOE（不存在则为 None）。
    """
    result: Dict[str, Optional[float]] = {
        "mAP": None, "NDS": None,
        "mATE": None, "mASE": None, "mAOE": None,
    }

    # nuScenes 格式: "mAP: 0.3456" 或 "NDS: 0.4321"
    for key in ("mAP", "NDS", "mATE", "mASE", "mAOE"):
        m = re.search(rf"{key}\s*[=:]\s*([0-9]+\.[0-9]+)", log_text, re.IGNORECASE)
        if m:
            result[key] = round(float(m.group(1)), 4)

    # KITTI 格式: "Car AP@0.70, 0.70, 0.70: 78.12 ..."（取第一个数值作为 mAP 近似）
    if result["mAP"] is None:
        m = re.search(r"AP@[^:]+:\s*([0-9]+\.[0-9]+)", log_text)
        if m:
            result["mAP"] = round(float(m.group(1)) / 100.0, 4)  # 转成 0~1

    return result


# ---------------------------------------------------------------------------
# 核心：调用 OpenPCDet test.py
# ---------------------------------------------------------------------------

def run_openpcdet_eval(
    openpcdet_root: str,
    cfg_file: str,
    ckpt_file: str,
    conda_env: str,
    output_dir: Path,
    model_name: str,
    timeout: int = 1800,
) -> Dict:
    """
    通过 WSL subprocess 调用 OpenPCDet tools/test.py。

    参数：
        openpcdet_root  WSL 内 OpenPCDet 根目录（如 /home/sunny/OpenPCDet）
        cfg_file        WSL 内配置文件路径
        ckpt_file       Windows 或 WSL 格式的 checkpoint 路径
        conda_env       conda 环境名
        output_dir      日志输出目录（Windows 本地 Path）
        model_name      用于日志命名（如 pretrained / trained_epoch1）
        timeout         最长等待秒数

    返回包含所有指标的字典，写入 csv 时使用。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / f"openpcdet_eval_{model_name}.log"

    ckpt_wsl = win_to_wsl(ckpt_file)
    size_mb  = file_size_mb(ckpt_file)

    # 判断 checkpoint 轮次
    epoch_note = ""
    if "epoch_1" in Path(ckpt_file).name:
        epoch_note = "checkpoint_epoch_1，小轮次训练结果，用于验证训练闭环"

    # 探测 WSL 内 conda.sh 实际路径（与主项目 openpcdet_json_detector.py 相同逻辑）
    conda_sh = _detect_conda_sh()
    if conda_sh is None:
        print("[eval_detection] ✗ 无法找到 WSL 内 conda.sh，请确认 conda 已安装")
        status = "conda_sh_not_found"
        return {
            "model_name":          model_name,
            "checkpoint_path":     ckpt_file,
            "checkpoint_size_mb":  size_mb,
            "mAP": None, "NDS": None, "mATE": None, "mASE": None, "mAOE": None,
            "status":  status,
            "note":    "未找到 WSL conda.sh，请检查 miniconda3/anaconda3 是否已安装",
            "log_file": "",
        }

    print(f"[eval_detection]   conda.sh: {conda_sh}")

    # 构建 bash 命令：显式 source conda.sh → activate → cd → python tools/test.py
    # 使用 wsl -e bash -c（不加 -l），完全依赖显式 source 加载 conda，
    # 避免非交互 shell 中 conda: command not found（exit 127）
    bash_cmd = (
        f"source {conda_sh} && "
        f"conda activate {conda_env} && "
        f"cd {openpcdet_root} && "
        f"python tools/test.py "
        f"--cfg_file {cfg_file} "
        f"--ckpt {ckpt_wsl} "
        f"--batch_size 1"
    )
    cmd = ["wsl", "-e", "bash", "-c", bash_cmd]

    print(f"[eval_detection] 开始评估模型: {model_name}")
    print(f"[eval_detection]   ckpt (WSL): {ckpt_wsl}")
    print(f"[eval_detection]   日志输出: {log_file}")

    status = "success"
    stdout_text = ""
    stderr_text = ""

    try:
        t0 = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
        elapsed = round(time.time() - t0, 1)
        stdout_text = proc.stdout or ""
        stderr_text = proc.stderr or ""

        # 保存完整日志
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"# model: {model_name}\n")
            f.write(f"# ckpt:  {ckpt_wsl}\n")
            f.write(f"# exit_code: {proc.returncode}\n")
            f.write(f"# elapsed_s: {elapsed}\n\n")
            f.write("=== STDOUT ===\n")
            f.write(stdout_text)
            f.write("\n=== STDERR ===\n")
            f.write(stderr_text)

        if proc.returncode != 0:
            status = f"exit_code={proc.returncode}"
            print(f"[eval_detection] ⚠ 退出码={proc.returncode}，见日志: {log_file}")
        else:
            print(f"[eval_detection] ✓ 完成，耗时 {elapsed}s")

    except subprocess.TimeoutExpired:
        status = "timeout"
        print(f"[eval_detection] ✗ 超时（>{timeout}s）")
    except FileNotFoundError:
        status = "wsl_not_found"
        print("[eval_detection] ✗ 未找到 wsl 命令，请确认在 Windows 下运行")
    except Exception as exc:
        status = f"error: {exc}"
        print(f"[eval_detection] ✗ 异常: {exc}")

    # 解析指标
    combined_log = stdout_text + "\n" + stderr_text
    metrics = parse_metrics(combined_log)

    # 拼装说明
    notes = []
    if epoch_note:
        notes.append(epoch_note)
    unresolved = [k for k, v in metrics.items() if v is None]
    if unresolved:
        notes.append(f"未从官方日志中解析到该指标: {', '.join(unresolved)}")
    note_str = "；".join(notes) if notes else ""

    row = {
        "model_name":          model_name,
        "checkpoint_path":     ckpt_file,
        "checkpoint_size_mb":  size_mb,
        "mAP":                 metrics["mAP"],
        "NDS":                 metrics["NDS"],
        "mATE":                metrics["mATE"],
        "mASE":                metrics["mASE"],
        "mAOE":                metrics["mAOE"],
        "status":              status,
        "note":                note_str,
        "log_file":            str(log_file),
    }
    return row


# ---------------------------------------------------------------------------
# 主函数（单独运行时使用）
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    """
    评估预训练模型与自训练模型，输出 detection_metrics.csv。
    返回 DataFrame 供 run_all.py 使用。
    """
    rows = []

    models = [
        ("pretrained",    args.pretrained_ckpt_file),
        ("trained_epoch1", args.trained_ckpt_file),
    ]

    for model_name, ckpt in models:
        if not ckpt:
            print(f"[eval_detection] 跳过 {model_name}：未提供 checkpoint 路径")
            rows.append({
                "model_name": model_name,
                "checkpoint_path": "",
                "checkpoint_size_mb": 0,
                "mAP": None, "NDS": None,
                "mATE": None, "mASE": None, "mAOE": None,
                "status": "skipped",
                "note": "未提供 checkpoint 路径",
                "log_file": "",
            })
            continue

        try:
            row = run_openpcdet_eval(
                openpcdet_root=args.openpcdet_root,
                cfg_file=args.cfg_file,
                ckpt_file=ckpt,
                conda_env=args.conda_env,
                output_dir=output_dir,
                model_name=model_name,
            )
            rows.append(row)
        except Exception as e:
            print(f"[eval_detection] ✗ {model_name} 评估异常: {e}")
            rows.append({
                "model_name": model_name,
                "checkpoint_path": ckpt,
                "checkpoint_size_mb": file_size_mb(ckpt),
                "mAP": None, "NDS": None,
                "mATE": None, "mASE": None, "mAOE": None,
                "status": f"exception: {e}",
                "note": "未从官方日志中解析到该指标",
                "log_file": "",
            })

    df = pd.DataFrame(rows)
    csv_path = output_dir / "detection_metrics.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[eval_detection] 已保存: {csv_path}")
    return df


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="检测精度评估模块")
    parser.add_argument("--output_dir",           default="outputs/experiments")
    parser.add_argument("--openpcdet_root",        default="/home/sunny/OpenPCDet")
    parser.add_argument("--cfg_file",              default="")
    parser.add_argument("--trained_ckpt_file",     default="")
    parser.add_argument("--pretrained_ckpt_file",  default="")
    parser.add_argument("--conda_env",             default="openpcdet_py310")
    args = parser.parse_args()

    out = Path(args.output_dir)
    main(args, out)
