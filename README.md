# 清洁机器人视觉系统

## 目录结构

```text
src/
└── robot_cleaning_vision/
    ├── __init__.py
    ├── main.py                  # 统一命令行入口
    ├── capture/                 # 摄像头采集与图片保存
    ├── preprocessing/           # 通用图像变换与坐标变换信息
    ├── models/
    │   ├── mlp/                 # 手工特征提取与 MLP
    │   ├── cnn_linear/          # 冻结 CNN 特征与线性分类器
    │   └── detector/            # 五类目标检测器
    ├── postprocessing/          # 筛选、坐标还原、计数与 PASS/FAIL
    └── visualization/           # 绘框、文字与画面显示
docs/                            # 需求、架构、方法与数据集说明
```

采集与显示共用，三种模型分别组织。通用图像操作放在 `preprocessing`，方法专用的特征提取放在对应模型目录；启动与调度集中在 `main.py`。

## 依赖包

| 包 | 用途 |
|---|---|
| `numpy` | 图像数组、特征向量和数值计算 |
| `opencv-python` | 摄像头采集、图像处理、绘框与窗口显示 |
| `scikit-image` | HOG 等手工视觉特征 |
| `scikit-learn` | MLP、逻辑回归、特征标准化和评估指标 |
| `torch` | 神经网络训练、推理与 GPU 计算 |
| `torchvision` | 预训练 CNN、标准目标检测器和图像变换 |
| `pillow` | 图像读取与模型输入转换 |
| `pycocotools` | TACO 的 COCO 标注读取、掩膜处理与评估 |
| `matplotlib` | 训练曲线、混淆矩阵和实验结果绘图 |

PyTorch 与 torchvision 使用匹配的 2.11／0.26 版本系列。Windows/Linux 从[官方 CUDA 12.8 源](https://download.pytorch.org/whl/cu128)安装，macOS 使用 PyPI 版本。

## 环境与入口

使用 Python 3.12，uv 管理环境：

```bash
uv sync --locked
uv run --locked robot-cleaning-vision --help
```

`pyproject.toml` 声明依赖范围，`uv.lock` 固定具体版本，两者一并提交。`uv sync --locked` 根据锁文件安装到项目的 `.venv`。

检查 PyTorch 是否可使用 GPU：

```bash
uv run --locked python -c "import torch; print(torch.cuda.is_available())"
```

