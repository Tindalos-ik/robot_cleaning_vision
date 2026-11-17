# 清洁机器人视觉系统

机器人视觉系统，识别物品进行五项分类

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
    └── visualization/           # 原图标注与右侧中文信息栏（仅绘制 API）
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
| `pillow` | 图像读取、模型输入转换和中文标注绘制 |
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


## 手工特征 + MLP

已接入最优保留模型的整图 PASS/FAIL 分类，模型固定为 `artifacts/mlp/model.pt`。

```bash
# 默认摄像头0实时识别
uv run robot-cleaning-vision run
# 单张图片真实推理
uv run robot-cleaning-vision run --input data/raw/taco/batch_5/000103.JPG
```

重新训练同一结构（配置在`models/mlp/train.py`顶部）：

```bash
uv run python -m robot_cleaning_vision.models.mlp.train
```

新权重与训练报告保存到`artifacts/mlp/training/<时间戳>/`，不覆盖当前实测模型。

摄像头窗口按 `Q` / `Esc` 退出，按 `S` / 空格保存当前原图、预测JSON及中文标注图。
模型已保存判定阈值，不需要额外输入模型参数或预测文件。
特征原理、保留模型的效果和场景实测说明见 [手工特征 + MLP](docs/手工特征+MLP.md)。
当前保留一个实测MLP模型及同一结构的完整训练流程；CNN与检测器目录仍未实现推理。

## 离线处理入口

`main.py` 提供 `process` 子命令，将原图、已有模型预测与三个公共模块接通。
该命令**读取真实预测文件，不执行推理或生成模拟预测**，
也不打开摄像头或窗口。原有 `capture` 命令不变。

```bash
uv run robot-cleaning-vision process --input data/raw/example.png --predictions predictions.json --target-size 640 640 --threshold 0.5 --font "C:/Windows/Fonts/msyh.ttc" --elapsed-ms 23.6 --output annotated.png
uv run robot-cleaning-vision process --help
```

输入为原始图片与 UTF-8 JSON。分类预测文件示例：

```json
{
  "task": "classification",
  "model_name": "CNN + 线性分类器",
  "fail_probability": 0.82
}
```

检测预测文件示例：

```json
{
  "task": "detection",
  "model_name": "五类目标检测器",
  "boxes": [[100, 190, 300, 340], [20, 150, 80, 200]],
  "class_ids": [2, 4],
  "scores": [0.9, 0.3]
}
```

检测框必须是**同一原图经本项目 `preprocess` 等比例居中填充后**的 xyxy 像素坐标，
并已解码、完成 NMS；类别 ID 为项目的 1–5。`--target-size` 按**高、宽**指定
产生预测时的画布尺寸，必须匹配，不能传已经还原到原图的框。
空检测写作 `"boxes": [], "class_ids": [], "scores": []`。
分类也须传入预处理尺寸，以保持统一的离线调用链。

入口读取原图为 BGR，调用预处理取得变换信息，再按 `task` 调用对应后处理，
最后在原图副本上绘制并保存 PNG。外部预测已经存在，故本次预处理图像不再送入模型。
对于上述检测示例，若原图为高 360、宽 640，则保留瓶目标，
还原框为 `(100, 50, 300, 200)`；杯目标因低于阈值被移除。

`--threshold` 必填，范围 [0, 1]；`--elapsed-ms` 可选，仅显示传入的最近完成帧
耗时，省略显示“—”，不是本次离线命令耗时。`--font` 必须指向可用的中文字体。
输出含右侧中文信息栏，父目录须已存在；只创建新的 `.png`，不覆盖已有文件，
包括输入原图。成功时打印输出路径及 PASS/FAIL；输入、字体、预测或保存失败时以状态码 1 退出。

## 摄像头采集

模块原理与代码阅读说明见[第一模块：图像采集知识整理](docs/01_图像采集模块知识整理.md)。

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

先写成功图片，再追加对应元数据；CSV 表头不匹配会报错而非覆盖。可处理的写入失败会清理本次未登记图片并回退 CSV 追加。采集只生成图片和元数据；类别、边界框及 `PASS/FAIL` 由后续标注流程生成。

## 图像预处理

`robot_cleaning_vision.preprocessing.preprocess` 接受摄像头的 BGR `uint8`
`numpy.ndarray`（HWC）或数据集的 PIL RGB 图像，不修改原始输入。
按调用方指定的 `(高, 宽)` 双线性等比例缩放并居中填充；
奇数填充余量放在右侧或下侧。`fill` 为 0–255 的单值或按输出通道顺序的三通道值。
输出颜色可选 RGB（默认）或 BGR，默认保留 `uint8` 像素值；
指定浮点 `dtype` 后可使用 `(像素 * scale - mean) / std`，
`scale` 默认 1，省略的 `mean`、`std` 分别为 0、1，三通道统计量按输出颜色顺序配置。

返回 `(image, transform)`：`image` 为独立的 HWC 数组；`transform` 包含
`original_size`、`resized_size`、`target_size`（均为高、宽），
`scale`（实际宽度比例、实际高度比例）和 `padding`（左、上、右、下）。
`forward_boxes`、`inverse_boxes` 接受 `(..., 4)` 的 xyxy 坐标，按实际缩放尺寸
正向映射或还原，返回浮点数组。

```python
import numpy as np
from PIL import Image
from robot_cleaning_vision.preprocessing import preprocess

with Image.open("data/raw/example.jpg") as source:
    image, transform = preprocess(
        source.convert("RGB"), target_size=(384, 512),
        fill=(114, 114, 114), color="RGB", dtype=np.float32,
        scale=1 / 255, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    )
boxes = np.array([[10, 20, 100, 120]], dtype=np.float32)
mapped = transform.forward_boxes(boxes)
restored = transform.inverse_boxes(mapped)
```

## 预测后处理

`robot_cleaning_vision.postprocessing` 仅提供 Python API，不绘图、不导出文件。
分类与检测各自显式传入 `threshold`（[0, 1]），模块不自动选择阈值。

- `postprocess_classification(fail_probability, *, threshold)`：输入有限的 FAIL
  概率（[0, 1]）；概率 **大于等于** 阈值时为 `FAIL`，否则为 `PASS`。
- `postprocess_detection(boxes, class_ids, scores, transform, *, threshold)`：
  接收模型已解码并完成 NMS 的结果，不重复 NMS。`boxes` 为预处理图像像素坐标下的
  `(N, 4)` xyxy 数组，坐标有限且宽高为正；`class_ids` 为 `(N,)` 整数项目类别 ID，
  `scores` 为 `(N,)` 有限得分（[0, 1]）。五类共用一个阈值，保留得分大于等于阈值的目标，
  保持输入顺序，通过本帧预处理返回的 `transform.inverse_boxes` 还原原图坐标。
  不修改输入、不裁剪越界坐标。空输入使用 `(0, 4)` 框数组和空类别、得分数组。
  检测器的零起始 ID 或其他类别编码须由调用方先映射为项目 ID。

两种接口均返回 `ImageResult`：

| 字段 | 分类结果 | 检测结果 |
|---|---|---|
| `decision` | `PASS` 或 `FAIL` | 至少保留一个目标为 `FAIL`，否则 `PASS` |
| `image_score` | 输入的 FAIL 概率 | `None` |
| `objects` | `None` | `Detection` 列表，无目标为 `[]`；每项含 `bbox`（原图浮点 xyxy 元组）、`class_id`、`score` |
| `counts` | `None` | `{类别 ID: 数量}`，始终包含五类及其零计数 |

项目类别 ID：`1=罐`、`2=瓶`、`3=包装物`、`4=杯`、`5=纸类物品`；
`CATEGORY_NAMES[class_id - 1]` 可获取类别名称。

```python
import numpy as np
from robot_cleaning_vision.preprocessing import preprocess
from robot_cleaning_vision.postprocessing import (
    postprocess_classification,
    postprocess_detection,
)

classification = postprocess_classification(0.7, threshold=0.7)
# ImageResult(decision='FAIL', image_score=0.7, objects=None, counts=None)

frame = np.zeros((360, 640, 3), dtype=np.uint8)  # 示例原图，也可使用摄像头帧
image, transform = preprocess(frame, target_size=(640, 640))
# 下列数组代表模型对 image 输出的已解码、已 NMS 结果。
detection = postprocess_detection(
    boxes=np.array([[100, 190, 300, 340], [20, 150, 80, 200]]),
    class_ids=np.array([2, 4]),
    scores=np.array([0.9, 0.3]),
    transform=transform,
    threshold=0.5,
)
# detection.decision == 'FAIL'，detection.image_score is None
# detection.objects[0].bbox == (100.0, 50.0, 300.0, 200.0)
# detection.objects[0].class_id == 2，detection.objects[0].score == 0.9
# detection.counts == {1: 0, 2: 1, 3: 0, 4: 0, 5: 0}
```

## 结果可视化

`robot_cleaning_vision.visualization.ResultRenderer` 在初始化时加载明确指定的
中文字体，逐帧调用 `draw` 返回独立、连续的 BGR `uint8` 数组，不修改输入。
左侧保留原图像素尺寸与位置，右侧追加中文信息栏，不覆盖或缩放原图；
小图高度不足时向下补足画布，长模型名和错误信息自动换行。

| 输入 | 含义 |
|---|---|
| `ResultRenderer(font_path, font_size=20)` | 中文字体文件路径（如微软雅黑 `.ttc` 或思源黑体 `.otf`），字号为像素；字体无法加载时直接报错 |
| `draw(frame, result=None, *, model_name, status="completed", elapsed_ms=None, error_message=None)` | `frame` 为非空 HWC BGR `uint8` 原图；`result` 为后处理返回的 `ImageResult` |
| `model_name` | 显示的模型名称 |
| `status` | `completed` 显示结果，且必须提供 `result`；`pending` 显示“待处理”；`error` 显示“错误”和可选错误信息。后两者忽略旧 `result`，不显示框、计数、概率或判定 |
| `elapsed_ms` | 调用方传入的最近完成帧耗时，非负有限毫秒数，显示一位小数；`None` 显示“—” |

分类侧栏显示模型名称、处理状态、最近完成帧耗时、PASS/FAIL 和 FAIL 概率，
不显示框或计数。检测在原图绘制原图坐标框、中文类别名和目标得分，
侧栏显示判定与五类计数（含零），不显示图像级置信度。
PASS 为绿色、FAIL 为红色；类别固定颜色为：罐橙色、瓶蓝色、包装物紫色、
杯青绿色、纸类物品粉色。得分与概率显示为一位小数的百分比。

```python
import numpy as np
from robot_cleaning_vision.postprocessing import postprocess_classification
from robot_cleaning_vision.visualization import ResultRenderer

# 整个会话复用一个实例；其他系统替换为本机中文字体的明确路径。
renderer = ResultRenderer("C:/Windows/Fonts/msyh.ttc")
frame = np.full((360, 640, 3), 80, dtype=np.uint8)  # 原始 BGR 帧
result = postprocess_classification(0.82, threshold=0.5)
annotated = renderer.draw(
    frame, result, model_name="CNN + 线性分类器", elapsed_ms=23.6,
)
# 检测时直接将后处理返回的 detection 作为 result，无需更换绘制接口。
pending = renderer.draw(
    frame, model_name="五类检测器", status="pending", elapsed_ms=23.6,
)
error = renderer.draw(
    frame, model_name="五类检测器", status="error",
    elapsed_ms=23.6, error_message="推理失败",
)
# annotated / pending / error 均可交给调用方的 OpenCV 显示或其他下游处理。
```
