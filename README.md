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


## 摄像头采集

在项目根目录运行（需要摄像头和可显示 OpenCV 窗口的本地桌面）：

```bash
uv run robot-cleaning-vision capture --camera 0 --output data/raw/local --scene-id scene_001
uv run robot-cleaning-vision capture --help
```

| 参数 | 含义 |
|---|---|
| `--camera SOURCE` | 整数为摄像头设备索引，默认 `0`；也可传入 OpenCV 可读取的视频流地址或视频文件路径。 |
| `--output DIR` | 输出根目录，默认 `data/raw/local`；相对路径基于进程启动时的工作目录。 |
| `--scene-id SCENE` | 必填，物理场景的单级目录名；同一物理布置的重复拍摄共用，改变物体布置后更换。 |
| `--width PIXELS` | 请求采集宽度，正整数像素；省略时使用设备默认值。 |
| `--height PIXELS` | 请求采集高度，正整数像素；省略时使用设备默认值。 |

例如，使用 `--width 1280 --height 720` 请求宽 1280、高 720；设备可能忽略设置，实际预览和保存尺寸以读到的图像为准。流地址可用 `--camera "rtsp://host/path"` 指定，支持的协议与编解码器取决于本地 OpenCV。视频文件读完视为读帧失败并报错退出。

| 按键 | 操作 |
|---|---|
| `S` / `s` / 空格 | 保存当前显示的这一帧原图；终端输出完整文件位置，失败时输出原因。 |
| `Q` / `q` / `Esc` | 退出；关闭窗口或终端按 `Ctrl+C` 也会释放摄像头并关闭窗口。 |

窗口显示当前场景、本次运行的成功保存数量和操作提示；文字只绘制在显示副本上。窗口标题保留完整场景名，画面内非 ASCII 场景字符使用转义显示（OpenCV 内置字体限制）。保存失败不会增加数量，可以继续采集；打开或读取设备失败则明确报错并释放资源。

### 图片与元数据

默认图片保存至 `data/raw/local/<scene_id>/`，元数据位于 `data/raw/local/images.csv`。使用其他 `--output` 时，两者相对该根目录组织。PNG 无损保存原始分辨率和像素，不包含提示文字。文件名由采集时间和 UUID 组成，并以独占创建方式防止覆盖；重复运行追加 CSV，不清空旧图片或旧记录。

| CSV 字段 | 含义 |
|---|---|
| `image_id` | 唯一 PNG 文件名，包含扩展名。 |
| `scene_id` | 物理场景名。 |
| `file_name` | 相对输出根目录的图片路径，形式为 `<scene_id>/<image_id>`，使用 `/` 分隔。 |
| `captured_at` | 该帧读取完成时的本地 ISO 8601 时间，包含时区偏移。 |
| `width`、`height` | 实际保存宽、高，单位像素。 |
| `camera_source` | 设备索引或视频流地址/视频文件路径的字符串表示。 |

先写成功图片，再追加对应元数据；CSV 表头不匹配会报错而非覆盖。可处理的写入失败会清理本次未登记图片并回退 CSV 追加；不保证断电、进程强杀或多个采集进程同时写同一输出目录时的事务性，请使用单个写入进程。采集只生成图片和元数据；类别、边界框及 `PASS/FAIL` 由后续标注流程生成。

### 供模型循环复用的接口

`capture.camera.Camera` 提供 `open()`、`read()`、`release()` 和 `with` 上下文管理，不依赖窗口或按键。`read()` 返回原始 BGR `numpy.ndarray`，形状为 `(height, width, 3)`，`dtype=uint8`，不会缩放或叠字：

```python
from robot_cleaning_vision.capture.camera import Camera

with Camera(0, width=640, height=480) as camera:
    frame = camera.read()
```

`capture.storage.ImageStore(output, scene_id, camera_source).save(frame)` 可独立保存原始 BGR 帧并返回元数据 `ImageRecord`；可用 `captured_at=` 传入带时区的 `datetime`，未传入时使用保存时刻。`capture.preview.run_capture()` 单独负责窗口和按键。

存储回归验证（标准库 `unittest`，无需摄像头）：

```bash
uv run --locked python -m unittest discover -s tests -p test_capture_storage.py -v
```
