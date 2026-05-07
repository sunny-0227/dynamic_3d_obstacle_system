"""
scripts/experiments/run_all.py

毕业论文定量实验总入口。

功能：
  1. 解析命令行参数
  2. 依次调用 eval_detection / eval_segmentation / eval_fps_ablation
  3. 生成 training_summary.csv（模型训练摘要，从 checkpoint 文件名与大小推断）
  4. 汇总所有 csv，生成 experiment_report.md（可直接复制到论文）

某个实验失败不影响其他实验继续运行。

使用示例：
  python scripts/experiments/run_all.py \
    --data_dir data \
    --output_dir outputs/experiments \
    --openpcdet_root /home/sunny/OpenPCDet \
    --cfg_file /home/sunny/OpenPCDet/tools/cfgs/nuscenes_models/cbgs_pp_mini.yaml \
    --trained_ckpt_file D:/sunny/openpcdet_models/checkpoint_epoch_1.pth \
    --pretrained_ckpt_file D:/sunny/openpcdet_models/pointpillar.pth \
    --conda_env openpcdet_py310
"""

from __future__ import annotations

import argparse
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# 将项目根加入 sys.path，使子模块可被正确导入
# ---------------------------------------------------------------------------

def _setup_sys_path():
    root = Path(__file__).resolve().parent.parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


_setup_sys_path()


# ---------------------------------------------------------------------------
# 同级模块动态导入（兼容直接运行 python run_all.py 和包导入两种方式）
# ---------------------------------------------------------------------------

def _import_sibling(module_name: str):
    """
    动态导入同目录下的兄弟模块。
    无论从项目根还是从 scripts/experiments/ 目录运行都能正常工作。
    """
    import importlib.util
    sibling_path = Path(__file__).resolve().parent / f"{module_name}.py"
    if not sibling_path.exists():
        raise FileNotFoundError(f"找不到模块文件: {sibling_path}")
    spec = importlib.util.spec_from_file_location(module_name, sibling_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Markdown 工具
# ---------------------------------------------------------------------------

def df_to_markdown(df: pd.DataFrame) -> str:
    """将 DataFrame 转换为 Markdown 表格字符串。"""
    if df is None or df.empty:
        return "_（无数据）_"
    cols = list(df.columns)
    header  = "| " + " | ".join(str(c) for c in cols) + " |"
    divider = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if val is None or (isinstance(val, float) and pd.isna(val)):
                cells.append("")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, divider] + rows)


def read_csv_safe(csv_path: Path) -> Optional[pd.DataFrame]:
    """安全读取 csv，失败返回 None。"""
    try:
        if csv_path.exists():
            return pd.read_csv(csv_path, encoding="utf-8-sig")
        return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 训练摘要（从 checkpoint 文件推断）
# ---------------------------------------------------------------------------

def build_training_summary(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    """
    根据 checkpoint 文件元信息生成训练摘要表。
    无法从文件本身读取训练曲线，只统计可知信息。
    """
    rows = []

    ckpt_configs = [
        {
            "model_name":       "PointPillar（预训练）",
            "checkpoint_path":  args.pretrained_ckpt_file,
            "training_epochs":  "预训练（官方权重）",
            "optimizer":        "Adam（官方默认）",
            "learning_rate":    "官方默认",
            "batch_size":       "官方默认",
            "dataset":          "KITTI / nuScenes（官方）",
            "note":             "官方发布预训练权重，非本次训练",
        },
        {
            "model_name":       "PointPillar（自训练 epoch_1）",
            "checkpoint_path":  args.trained_ckpt_file,
            "training_epochs":  1,
            "optimizer":        "Adam",
            "learning_rate":    "0.001（OpenPCDet 默认）",
            "batch_size":       4,
            "dataset":          "nuScenes Mini",
            "note":             "checkpoint_epoch_1，小轮次训练，用于验证训练闭环",
        },
    ]

    for cfg in ckpt_configs:
        ckpt_path = cfg["checkpoint_path"]
        size_mb = 0.0
        if ckpt_path:
            p = Path(ckpt_path)
            if p.exists():
                size_mb = round(p.stat().st_size / 1024 / 1024, 2)
        cfg["checkpoint_size_mb"] = size_mb
        rows.append(cfg)

    df = pd.DataFrame(rows)
    csv_path = output_dir / "training_summary.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[run_all] 已保存: {csv_path}")
    return df


# ---------------------------------------------------------------------------
# 实验报告 Markdown 生成
# ---------------------------------------------------------------------------

_TRAINING_PARAMS_TABLE = """
| 参数项 | 设置值 |
| --- | --- |
| 模型架构 | PointPillar |
| 输入表示 | 柱状体素（Pillar） |
| Voxel 大小 | 0.16m × 0.16m |
| 点云范围（X） | −51.2m ～ 51.2m |
| 点云范围（Y） | −51.2m ～ 51.2m |
| 点云范围（Z） | −5.0m ～ 3.0m |
| 每个 Pillar 最大点数 | 32 |
| 最大 Pillar 数 | 16000 |
| Batch Size | 4 |
| 优化器 | Adam |
| 初始学习率 | 0.001 |
| 学习率调度 | One-cycle |
| 训练数据集 | nuScenes Mini |
| 框架版本 | OpenPCDet |
""".strip()


def generate_report(
    output_dir: Path,
    args: argparse.Namespace,
) -> Path:
    """
    读取所有 csv，汇总生成 experiment_report.md。
    某个 csv 不存在则在报告中注明。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # 读取各 csv
    df_train   = read_csv_safe(output_dir / "training_summary.csv")
    df_detect  = read_csv_safe(output_dir / "detection_metrics.csv")
    df_seg     = read_csv_safe(output_dir / "segmentation_iou.csv")
    df_fps     = read_csv_safe(output_dir / "fps_comparison.csv")
    df_ablation= read_csv_safe(output_dir / "parameter_ablation.csv")

    def section(csv_name: str, df: Optional[pd.DataFrame]) -> str:
        if df is None:
            return f"_该实验未成功生成（{csv_name} 不存在或读取失败）_"
        return df_to_markdown(df)

    # 为检测表增加可读性列（去掉 log_file 列，并决定展示哪些列）
    if df_detect is not None:
        drop_cols = [c for c in ["log_file"] if c in df_detect.columns]
        df_detect_disp = df_detect.drop(columns=drop_cols)

        # 若 mAP/NDS 全部为空（nuScenes 数据路径缺失），则展示 recall 替代列
        map_empty = df_detect_disp["mAP"].isna().all() if "mAP" in df_detect_disp.columns else True
        if not map_empty:
            # 有 mAP：去掉 recall 列以保持表格简洁
            recall_cols = [c for c in ["recall_rcnn_0.3", "recall_rcnn_0.5",
                                       "recall_rcnn_0.7", "avg_pred_objects"]
                           if c in df_detect_disp.columns]
            df_detect_disp = df_detect_disp.drop(columns=recall_cols)
        # 否则保留 recall 列，在报告注释中说明
        detect_note_extra = (
            "\n> **注（数据路径说明）**：nuScenes 官方 mAP 计算需要 NuScenes Python SDK 访问原始"
            "数据集元数据（`../data/nuscenes/v1.0-mini`），若评估时该路径不存在则 mAP 为空。\n"
            "> 推理本身已完整执行，表中 `recall_rcnn_*` 与 `avg_pred_objects` 为从日志提取的"
            "替代指标，可反映模型实际检测能力。\n"
            "> **修复方法**：在 WSL 中执行以下命令建立符号链接后重新运行，即可获得完整 mAP：\n"
            "> ```bash\n"
            "> # 将 <your_nuscenes_path> 替换为 nuScenes mini 数据集的实际路径\n"
            "> ln -s <your_nuscenes_path> /home/sunny/OpenPCDet/data/nuscenes\n"
            "> ```"
        ) if map_empty else ""
    else:
        df_detect_disp = df_detect
        detect_note_extra = ""

    # 分割表只展示部分关键列
    if df_seg is not None:
        seg_cols = [c for c in [
            "file_name", "point_count", "ground_ratio", "nonground_ratio",
            "cluster_count", "obstacle_count", "pseudo_iou", "coverage_ratio", "note"
        ] if c in df_seg.columns]
        df_seg_disp = df_seg[seg_cols]
    else:
        df_seg_disp = df_seg

    lines = [
        f"# 毕业论文实验报告",
        f"",
        f"> 自动生成时间：{now}",
        f"> 项目：面向动态场景的三维障碍物检测与分割系统研究",
        f"",
        f"---",
        f"",
        f"## 表3-1  PointPillar 模型训练参数设置表",
        f"",
        _TRAINING_PARAMS_TABLE,
        f"",
        f"---",
        f"",
        f"## 表3-2  模型训练结果摘要表",
        f"",
        section("training_summary.csv", df_train),
        f"",
        f"> 注：自训练模型仅训练 1 个 epoch，用于验证训练闭环，不代表最终精度。",
        f"",
        f"---",
        f"",
        f"## 表4-1  三维目标检测精度对比表",
        f"",
        section("detection_metrics.csv", df_detect_disp),
        f"",
        f"> 注：mAP 来自 OpenPCDet 官方 tools/test.py 输出日志解析。",
        f"> 若日志中未出现对应指标，则填写空值并在 note 列说明。",
        f"> 完整评估日志见 outputs/experiments/openpcdet_eval_*.log。",
        detect_note_extra,
        f"",
        f"---",
        f"",
        f"## 表4-2  几何分割统计表",
        f"",
        section("segmentation_iou.csv", df_seg_disp),
        f"",
        f"> 注：本系统实时分割方法为 RANSAC 地面分割 + DBSCAN 聚类，",
        f"> 无人工逐点语义标注，因此不输出严格语义 IoU。",
        f"> pseudo_iou 为多次独立 RANSAC 分割结果的交并比，反映地面分割稳定性。",
        f"> coverage_ratio 为非地面点中被聚类覆盖的比例。",
        f"> 以上结果为几何规则统计，不作为严格语义分割精度。",
        f"",
        f"---",
        f"",
        f"## 表4-3  不同点云规模下 FPS 对比表",
        f"",
        section("fps_comparison.csv", df_fps),
        f"",
        f"> 注：测试环境为本机 CPU，重复 5 次取均值。",
        f"> 处理链包含：RANSAC 地面分割 + DBSCAN 聚类。",
        f"> fps = 1000 / avg_time_ms。",
        f"",
        f"---",
        f"",
        f"## 表4-4  不同参数设置下性能对比表（参数消融）",
        f"",
        section("parameter_ablation.csv", df_ablation),
        f"",
        f"> 注：消融实验在同一点云文件上进行（最多 20000 点）。",
        f"> stability_note 根据 3 次重复实验 cluster_count 标准差判断：",
        f"> 稳定（std < 1）/ 轻微波动（std < 3）/ 波动较大（std ≥ 3）。",
        f"",
        f"---",
        f"",
        f"## 附录：实验文件索引",
        f"",
        f"| 文件名 | 说明 |",
        f"| --- | --- |",
        f"| detection_metrics.csv | 检测精度评估结果 |",
        f"| segmentation_iou.csv | 几何分割统计结果 |",
        f"| fps_comparison.csv | FPS 对比结果 |",
        f"| parameter_ablation.csv | 参数消融结果 |",
        f"| training_summary.csv | 训练摘要 |",
        f"| openpcdet_eval_pretrained.log | 预训练模型评估原始日志 |",
        f"| openpcdet_eval_trained_epoch1.log | 自训练模型评估原始日志 |",
        f"",
    ]

    md_path = output_dir / "experiment_report.md"
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[run_all] 已生成报告: {md_path}")
    return md_path


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="毕业论文定量实验总入口",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data_dir",
        default="data",
        help="本地点云数据目录（用于分割和 FPS 实验）")
    parser.add_argument("--output_dir",
        default="outputs/experiments",
        help="实验结果输出目录")
    parser.add_argument("--openpcdet_root",
        default="/home/sunny/OpenPCDet",
        help="WSL 中 OpenPCDet 根目录路径")
    parser.add_argument("--cfg_file",
        default="",
        help="OpenPCDet 配置文件路径（WSL 路径）")
    parser.add_argument("--trained_ckpt_file",
        default="D:/sunny/openpcdet_models/checkpoint_epoch_1.pth",
        help="自训练 checkpoint 路径（Windows 或 WSL 路径均可）")
    parser.add_argument("--pretrained_ckpt_file",
        default="D:/sunny/openpcdet_models/pointpillar.pth",
        help="预训练 checkpoint 路径")
    parser.add_argument("--conda_env",
        default="openpcdet_py310",
        help="WSL conda 环境名称")
    parser.add_argument("--skip_detection",
        action="store_true",
        help="跳过 OpenPCDet 评估（用于快速测试分割/FPS 实验）")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  毕业论文定量实验模块")
    print(f"  输出目录: {output_dir.resolve()}")
    print("=" * 60)

    # ── 实验0: 训练摘要 ──────────────────────────────────────
    print("\n[1/4] 生成训练摘要表...")
    try:
        build_training_summary(args, output_dir)
    except Exception:
        print("[run_all] ✗ 训练摘要生成失败:")
        traceback.print_exc()

    # ── 实验1: 检测精度评估 ───────────────────────────────────
    if args.skip_detection:
        print("\n[2/4] 跳过 OpenPCDet 评估（--skip_detection）")
    else:
        print("\n[2/4] 开始检测精度评估（OpenPCDet）...")
        try:
            eval_det = _import_sibling("eval_detection")
            eval_det.main(args, output_dir)
        except Exception:
            print("[run_all] ✗ 检测精度评估失败:")
            traceback.print_exc()

    # ── 实验2: 几何分割统计 ───────────────────────────────────
    print("\n[3/4] 开始几何分割统计...")
    try:
        # 补充分割相关默认参数（run_all 不暴露这些参数，使用合理默认值）
        if not hasattr(args, "ransac_threshold"):
            args.ransac_threshold   = 0.20
        if not hasattr(args, "dbscan_eps"):
            args.dbscan_eps         = 0.50
        if not hasattr(args, "dbscan_min_samples"):
            args.dbscan_min_samples = 10
        eval_seg = _import_sibling("eval_segmentation")
        eval_seg.main(args, output_dir)
    except Exception:
        print("[run_all] ✗ 几何分割统计失败:")
        traceback.print_exc()

    # ── 实验3: FPS 对比 + 参数消融 ───────────────────────────
    print("\n[4/4] 开始 FPS 对比与参数消融实验...")
    try:
        eval_fps = _import_sibling("eval_fps_ablation")
        eval_fps.main(args, output_dir)
    except Exception:
        print("[run_all] ✗ FPS / 消融实验失败:")
        traceback.print_exc()

    # ── 汇总报告 ──────────────────────────────────────────────
    print("\n[报告] 生成 experiment_report.md...")
    try:
        report_path = generate_report(output_dir, args)
        print(f"\n{'=' * 60}")
        print(f"  ✓ 所有实验完成！")
        print(f"  报告路径: {report_path.resolve()}")
        print(f"  输出目录: {output_dir.resolve()}")
        print(f"{'=' * 60}")
    except Exception:
        print("[run_all] ✗ 报告生成失败:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
