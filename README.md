# 清洁机器人视觉系统

清洁机器人的图像采集和整图 PASS/FAIL 识别。画面包含罐、瓶、包装物、杯或纸类物品时，应判为 FAIL。

## 安装

在项目根目录用 uv 安装依赖：

```bash
uv sync --locked
uv run --locked robot-cleaning-vision --help
```

## 图片和摄像头识别

| `--model` | 方法 | 默认权重 |
|---|---|---|
| `baseline1` | [手工特征 + MLP](docs/手工特征+MLP.md) | `artifacts/mlp/model.pt` |
| `baseline2` | [冻结 ResNet-50 + 逻辑回归](docs/冻结CNN+线性分类器.md) | `artifacts/cnn_linear/model.pt` |

```bash
uv run --locked robot-cleaning-vision run --model baseline2 --input image.jpg
uv run --locked robot-cleaning-vision run --model baseline2 --camera 0
```

把 `image.jpg` 换成图片路径。用 `--model-path` 指定其他权重；默认中文字体不可用时，用 `--font` 指定本机字体文件。

单图识别自动保存原图、`prediction.json` 和标注图到 `artifacts/inference/` 下的新目录。摄像头窗口按 `S` 或空格保存，按 `Q` 或 `Esc` 退出。

## 摄像头采集

```bash
uv run --locked robot-cleaning-vision capture --camera 0 --scene-id scene_001
```

按 `S` 或空格保存原图到 `data/raw/local/<scene-id>/`，采集记录写入 `data/raw/local/images.csv`。改变物体布置后更换 `--scene-id`；按 `Q` 或 `Esc` 退出。其他参数见 `capture --help` 和[图像采集说明](docs/图像采集模块.md)。

## 用已有预测生成标注图

用 `process` 读取原图和已有的预测 JSON，按指定阈值重新生成标注图。

例如，取 `baseline2` 保存的 `original.png` 和 `prediction.json`，把 FAIL 阈值改为 `0.5`：

```bash
uv run --locked robot-cleaning-vision process --input original.png --predictions prediction.json --target-size 384 384 --threshold 0.5 --font "C:/Windows/Fonts/msyh.ttc" --output annotated.png
```

把原图、JSON 和字体换成实际路径。`--target-size` 按高、宽填写预测所用的预处理尺寸；输出必须是尚不存在的 PNG 文件，父目录须已存在。

JSON 还可来自外部模型，须使用 UTF-8 并包含 `model_name` 和 `task`：

- `classification`：提供 `fail_probability`，达到阈值即判 FAIL。
- `detection`：提供 `boxes`、`class_ids` 和 `scores`。框须使用等比例缩放并居中填充后的画布像素坐标 `xyxy`，已完成 NMS；类别 ID 为 1–5。`process` 会筛选检测框、还原坐标并计数。

## 训练

先按[数据准备指南](dataprocessing/README.md)准备数据。方法一的正式训练命令见[手工特征 + MLP](docs/手工特征+MLP.md#4-重新训练)；方法二运行：

```bash
uv run --locked python -m robot_cleaning_vision.models.cnn_linear.train
```

方法二仅用 TACO 验证集校准阈值并报告验证指标，新权重不会覆盖默认权重。试用时用 `run --model-path` 指定新生成的 `model.pt`。

## 开发文档

[架构](docs/架构设计.md) · [图像预处理](docs/图像预处理模块.md) · [预测后处理](docs/后处理模块.md) · [结果可视化](docs/可视化模块.md)
