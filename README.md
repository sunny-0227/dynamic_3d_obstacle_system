# 面向动态场景的三维障碍物检测与分割系统研究

三维点云感知与分析系统 | 毕业设计工程

---

## 系统功能

- **离线点云分析**：加载 `.bin` / `.pcd` 单文件或 nuScenes mini 数据集，执行检测、分割与融合可视化
- **实时相机感知**：支持 Intel RealSense 深度相机或 Mock Camera（本地文件循环播放）
- **OpenPCDet 深度学习检测**：通过 WSL2 subprocess 调用 PointPillar 等模型，支持失败自动降级
- **轻量几何算法**：RANSAC 地面分割 + DBSCAN 聚类 + 3D 边界框，无需 GPU
- **统一 3D 可视化**：Open3D 非阻塞窗口，实时展示点云 + 分割着色 + 检测框

---

## 环境要求

- **操作系统**：Windows 10/11（主程序）；WSL2 Ubuntu（OpenPCDet 推理，可选）
- **Python**：3.10 或以上
- **硬件**：Intel RealSense D435 或兼容型号（实时模式，可选）

---

## 安装步骤

### 1. 克隆项目

```bash
git clone <repo_url>
cd dynamic_3d_obstacle_system
```

### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```

> 若需使用 Intel RealSense 实时模式，额外安装：
> ```bash
> pip install pyrealsense2
> ```

### 3. 启动系统

```bash
python main.py
```

---

## 配置说明

主要配置文件：`config/settings.yaml`

### 离线模式（开箱即用，无需修改）

- 单文件点云：GUI 界面选择 `.bin` 或 `.pcd` 文件即可
- nuScenes mini：在 GUI 中选择数据集根目录（需包含 `v1.0-mini/` 子目录）

### 实时模式配置

```yaml
realtime:
  camera_type: "mock"   # "mock"=本地文件循环 | "realsense"=真实相机
  mock_fps: 5.0          # Mock Camera 目标帧率
  width: 424             # RealSense 分辨率（推荐 424x240）
  height: 240
  fps: 30
  min_depth_m: 0.3       # 有效深度范围（米）
  max_depth_m: 4.0
```

### OpenPCDet WSL 推理配置（可选）

若不配置，系统自动使用模拟检测，离线功能不受影响。

```yaml
detector:
  openpcdet_wsl:
    enable_wsl: true
    infer_script: "/mnt/c/Users/<你的用户名>/dynamic_3d_obstacle_system/infer_to_json.py"
    cfg_file: "/home/<wsl用户名>/OpenPCDet/tools/cfgs/kitti_models/pointpillar.yaml"
    ckpt_file: "/mnt/d/<路径>/pointpillar.pth"
    openpcdet_tools_dir: "/home/<wsl用户名>/OpenPCDet/tools"
    conda_env: "openpcdet_py310"
    tmp_dir: "D:/<路径>/openpcdet_result"
```

路径配置完成后，也可在系统 GUI「模型与配置」页面直接修改并保存。

---

## 项目结构

```
dynamic_3d_obstacle_system/
├── main.py                        # 程序入口
├── infer_to_json.py               # OpenPCDet 推理脚本（在 WSL 中运行）
├── requirements.txt
├── config/
│   └── settings.yaml              # 全局配置
├── app/
│   ├── core/                      # 检测/分割 pipeline 与算法
│   │   ├── detector/              # BaseDetector + OpenPCDetJsonDetector
│   │   ├── segmentor/             # BaseSegmentor + MMDet3DSegmentor（占位）
│   │   ├── pipeline/              # DetectPipeline / SegmentPipeline / FullPipeline
│   │   └── fusion/                # 结果坐标融合 → FusedScene
│   ├── realtime/                  # 实时模式
│   │   ├── camera_interface.py    # 相机抽象接口 IPointCloudCamera
│   │   ├── mock_camera.py         # Mock Camera（本地文件循环）
│   │   ├── realsense_camera.py    # Intel RealSense 接入
│   │   ├── realtime_segmentor.py  # RANSAC 地面分割
│   │   ├── realtime_detector.py   # DBSCAN 聚类检测
│   │   └── realtime_pipeline.py   # LightweightRealtimePipeline（主流水线）
│   ├── datasets/                  # nuScenes mini 数据集加载
│   ├── io/                        # 点云文件加载
│   ├── ui/                        # PyQt5 界面
│   │   ├── main_window.py         # 主窗口（左导航 + 右内容）
│   │   ├── controller.py          # 应用控制器 + 状态管理
│   │   └── pages/                 # 四个内容页面
│   │       ├── offline_page.py
│   │       ├── realtime_page.py
│   │       ├── config_page.py
│   │       └── log_page.py
│   └── visualization/             # Open3D 场景渲染
├── data/                          # 示例点云文件（.bin / .pcd）
└── outputs/logs/                  # 运行日志输出目录
```

---

## nuScenes mini 数据集

1. 从 [nuScenes 官网](https://www.nuscenes.org/nuscenes#download) 下载 `v1.0-mini`
2. 解压后目录结构应为：
   ```
   <根目录>/
   ├── v1.0-mini/
   │   ├── scene.json
   │   ├── sample.json
   │   └── ...
   └── samples/
       └── LIDAR_TOP/
           └── *.pcd.bin
   ```
3. 在 GUI「离线点云分析」页面点击「选择 nuScenes 根目录」，指向上述 `<根目录>`，然后点击「加载数据集」

---

## OpenPCDet WSL2 配置（可选）

详细安装步骤请参考：[OpenPCDet 官方文档](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/INSTALL.md)

简要流程：
1. WSL2 中安装 conda 环境并安装 OpenPCDet
2. 将 `infer_to_json.py` 放到项目根目录（已包含）
3. 在 `config/settings.yaml` 或 GUI 配置页中填写路径
4. 系统调用失败时自动降级为模拟检测，不影响其他功能

---

## 常见问题

**Q: 启动时提示 `open3d` 导入失败？**
A: 运行 `pip install open3d`，Windows 需要 Python 3.10/3.11。

**Q: RealSense 提示找不到设备？**
A: 确认相机通过 USB 3.0 连接，安装 [Intel RealSense SDK 2.0](https://github.com/IntelRealSense/librealsense/releases)，再运行 `pip install pyrealsense2`。

**Q: OpenPCDet 检测始终显示"模拟检测"？**
A: 检查 WSL2 是否安装、`conda_env` 名称是否正确、路径是否使用 `/mnt/...` 格式。也可在 GUI「模型与配置」页面保存配置后重新执行分析。

**Q: 实时模式 FPS 很低？**
A: 在 `settings.yaml` 中适当增大 `voxel_size`（如 0.08）或 `process_interval`（如 3），减少每帧处理点数。
