# YOLO 目标检测

## 1. 运行

在项目根目录使用锁定环境：

```bash
uv sync --locked
uv run --locked robot-cleaning-vision run --model baseline3 --input data/raw/taco/batch_5/000103.JPG
uv run --locked robot-cleaning-vision run --model baseline3 --camera 0
```

`baseline3` 默认从 `artifacts/detector/model.pt` 加载 Ultralytics 原生 checkpoint。图片按 EXIF 转正，摄像头直接使用 BGR 帧；两者均等比例缩放，居中填充像素值 114 至 640×640，再由 Ultralytics 转为归一化 RGB。离线推理只需本地默认权重，无须下载 COCO 权重。

图片模式直接保存结果。摄像头按 `S` 或空格保存当前帧，按 `Q`、`Esc` 或关闭窗口退出。每次保存会在 `artifacts/inference/` 新建目录，写入 `original.png`、`prediction.json`、`result.png`。结果包含原图像素坐标、类别、置信度、五类计数和 `PASS/FAIL`；画布坐标用于回放。试用新模型时追加 `--model-path <新权重路径>`。

回放时，将保存目录中的 `original.png` 和 `prediction.json` 路径传给 `process`：

```bash
uv run --locked robot-cleaning-vision process --input original.png --predictions prediction.json --target-size 640 640 --threshold 0.25 --font "C:/Windows/Fonts/msyh.ttc" --output replay.png
```

JSON 的 `boxes` 是已执行 NMS 的画布 `xyxy` 坐标，`original_boxes` 是显示方向下的原图坐标；两组框与 `class_ids`、`scores` 顺序一致。把 JSON 的 `elapsed_ms` 作为 `--elapsed-ms` 可以同时复现结果图中的耗时。

## 2. 模型结构

模型是 [Ultralytics YOLO11n Detect](https://docs.ultralytics.com/models/yolo11/)，库版本为 `ultralytics==8.4.87`，结构由官方 [`yolo11.yaml`](https://github.com/ultralytics/ultralytics/blob/v8.4.87/ultralytics/cfg/models/11/yolo11.yaml) 的 `n` 档（深度系数 0.50，宽度系数 0.25，通道上限 1024）定义。从 Ultralytics 官方 [COCO 预训练 `yolo11n.pt`](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt) 初始化，检查 SHA-256 后加载；训练将 80 类检测头重建为五类，只复用形状相容的权重。五类模型共有 **2,590,815 个参数**，其中 2,590,799 个参与优化；固定的 16 个参数用于 DFL 解码的加权投影。Ultralytics 提供 [AGPL-3.0 和 Enterprise 许可](https://www.ultralytics.com/license)，再分发前查看适用条款。

### Backbone：下采样并提取特征

3×3、步长 2 的 `Conv` 逐级将高宽减半，卷积后接 BatchNorm 和 SiLU。`C3k2` 走部分通道的短路径，其余通道经过瓶颈块后再拼接融合；较深的 `C3k2` 使用 `C3k` 子块。第 9 层 `SPPF` 在 P5 上连续做步长 1 的 5×5 最大池化，拼接不同感受野的结果。第 10 层 `C2PSA` 对一部分通道施加空间自注意力与前馈处理，再与旁路融合；输出高宽保持 20×20。

### Neck：双向融合三层尺度

先把深层 P5 特征上采样，与 backbone 的第 6 层 P4 拼接，再上采样与第 4 层 P3 拼接。接着通过两次步长 2 卷积沿反方向下采样，分别接回第 13 层 P4 和第 10 层 P5；每次拼接后由 `C3k2` 混合通道。

| 检测尺度 | Backbone 层号及输出 | Neck 输出层号及尺寸（通道 × 高 × 宽） | Stride | 位置数 |
|---|---|---|---:|---:|
| P3 | 4：128×80×80 | 16：64×80×80 | 8 | 6,400 |
| P4 | 6：128×40×40 | 19：128×40×40 | 16 | 1,600 |
| P5 | 10：256×20×20 | 22：256×20×20 | 32 | 400 |

### Head：从网格位置到检测框

第 23 层 `Detect` 接收上述三张特征图。每个网格中心采用独立回归和分类分支：回归分支用两个 3×3 卷积块及 1×1 输出层，预测上、下、左、右四个边界距离，各有 16 个 `logits`；分类分支用两组 3×3 depthwise + 1×1 pointwise 卷积及 1×1 输出层，输出五个类别 `logits`。每个位置有 64 个回归值和 5 个分类值，无单独的 objectness 通道，也不依赖预设长宽的 anchor box。

回归距离经 softmax 和 DFL 期望解码，再乘各层 stride；类别经 sigmoid。正方形输入下，解码后、NMS 前的输出为 $B\times9\times8400$（四个 `xywh` 和五个类别分数）。部署使用 `conf=0.25` 及类别内 NMS `iou=0.7`，最多返回 300 个检测。NMS 按置信度保留同类中得分更高的重叠框；不同类不互相抑制。项目再裁去填充区域、去掉零面积框并反算原图 `xyxy` 坐标。每个保留框按类别计一次；至少一个目标即 `FAIL`，零目标即 `PASS`。

### 损失

`TaskAlignedAssigner` 同时考虑类别评分与框重合度，从候选网格中心为每个真实框找正样本。训练损失由 CIoU 框回归、BCE 分类、DFL 距离分布三项组成，默认权重分别为 7.5、0.5、1.5。CIoU 衡量重合、中心距离和宽高比；DFL 对每条边的连续距离，用相邻两个离散位置按距离加权监督。空标签的负图仍提供分类背景信号。

## 3. 数据与标注

| 分区 | 来源 | 图数 | 用途 |
|---|---|---:|---|
| Train | TACO train 621、CoxPWD v3 110、Wikimedia 21、Indoor Waste train 52、本地审核训练照片 195 | 999 | 微调，含 233 张无目标负图与 1,262 个真值框 |
| TACO val | 项目固定验证分区 | 134 | 每轮验证，仅按检测 mAP50:95 选最佳 checkpoint |
| TACO test | 项目固定测试分区 | 134 | 权重冻结后独立评估 |
| Indoor holdout | 冻结清单，和 Indoor train 按拍摄场景分开 | 52 | 权重冻结后独立评估 |

| 项目 ID | 项目名 | YOLO ID | checkpoint 名 |
|---:|---|---:|---|
| 1 | 罐 | 0 | `can` |
| 2 | 瓶 | 1 | `bottle` |
| 3 | 包装物 | 2 | `wrapper` |
| 4 | 杯 | 3 | `cup` |
| 5 | 纸类物品 | 4 | `paper` |

项目标注提供 EXIF 显示方向下的像素 `xywh`；只转正像素，框保持原方向。转换验证显示尺寸、框边界、类别、计数与 `PASS/FAIL`，随后写出归一化 `class xc yc w h`。按正式图像 ID 命名临时文件，核对清单摘要、来源路径、场景组以及文件和显示像素的跨分区重复。正式 `PASS` 图保留空标签。TACO 已将影响计数的 `crowd/ignore`、待定类别和无效框所在图留出；转换时再拒绝正式分区出现此类目标，避免把它们当背景。195 张本地图全部属于训练，其中 39 个 `difficult` 仍是审核确认的可计数目标，保留原类和原框。旧 90 张本地清单已包含在 195 张清单中，训练只读完整清单一次。

## 4. 重新训练和评估

按 [TACO 准备指南](../dataprocessing/taco/README.md) 和[补充数据操作](../dataprocessing/README.md)恢复原图、标注及冻结清单，再运行：

```bash
uv run --locked python -m robot_cleaning_vision.models.detector.train
```

默认以 seed 42 微调全训练集，上限 50 轮、patience 10、batch 8，使用临时磁盘缓存。Windows 默认由主进程加载数据，需要时可用 `--workers` 调整。CUDA 可用时使用 GPU 并运行 Ultralytics AMP 检查；可用 `--device cpu --no-amp` 在 CPU 训练，或用 `--batch` 适配显存。首次训练自动下载官方 COCO 权重，也可用 `--pretrained <yolo11n.pt>` 指定同一文件。混合精度预检可能单独下载 Ultralytics 的检查权重，它不参与模型初始化或微调，与临时数据一起清理。

每次复训生成 `artifacts/detector/training/<运行目录>/fit/weights/best.pt`、训练曲线数据及 `training.json`，不覆盖默认权重。训练只向 Ultralytics 提供 train 与 TACO val；审计留出分区仅用于核对来源和隔离。评估固定权重时运行：

```bash
uv run --locked python -m robot_cleaning_vision.models.detector.evaluate --weights artifacts/detector/model.pt --output artifacts/detector/evaluation.json
```

评估读取 TACO val、TACO test 和冻结的 Indoor holdout。AP 用 Ultralytics 检测验证器、`conf=0.001` 及 NMS IoU 0.7，按预测得分排序；mAP50 用真值匹配 IoU 0.50，mAP50:95 使用 0.50–0.95、间隔 0.05 的十档 IoU。部署判定和计数使用原图、项目推理和固定 `conf=0.25`；同一张图上比较总数和五类计数向量。评估新权重可追加 `--training-report <运行目录>/training.json`，核对权重与输入清单摘要。

## 5. 实测与局限

默认权重来自 RTX 4050 Laptop GPU 上的 50 轮微调：实际 batch 16、1 个数据加载进程、AMP、AdamW、seed 42；可在复训命令后追加 `--batch 16 --workers 1` 使用相同设置。按 TACO val 的训练期 mAP50:95 选出第 49 轮（mAP50 48.06%，mAP50:95 38.97%），之后以同一冻结权重独立验证下列三个分区。独立验证使用 batch 8 的矩形批量推理；下表的 TACO val 是重测值，选型依据仍是训练期指标。

| 分区 | 图数 | mAP50 | mAP50:95 | 判定准确率 | PASS 召回率 | FAIL 召回率 | 总数 MAE（个/图） | 五类计数全对率 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| TACO val | 134 | 48.38% | 39.43% | 83.58% | 51.85% | 91.59% | 0.701 | 36.57% |
| TACO test | 134 | 42.90% | 30.39% | 85.07% | 73.08% | 87.96% | 0.612 | 39.55% |
| Indoor holdout | 52 | 72.24% | 56.03% | 76.92% | 85.00% | 71.88% | 0.385 | 59.62% |

总数 MAE 是每图检测总数与真值总数的绝对差的平均值；五类计数全对要求五维计数逐项相同。逐类 AP、混淆矩阵、逐类计数误差及清单摘要见 [`metrics.json`](../artifacts/detector/metrics.json)；每轮验证数据见 [`training.csv`](../artifacts/detector/training.csv)，当时的 Ultralytics 参数见 [`training-args.yaml`](../artifacts/detector/training-args.yaml)。复训命令会重建其中的临时数据路径。

### 按图片看漏检和误报

部署时保留置信度达到 `0.25` 的检测框。只要有一个框就判为 FAIL，因此整图检出与框的位置、类别、数量是否全部正确要分别看。

| 分区 | 有垃圾 / 干净（张） | 检出垃圾图 | 漏检 | 正确放行干净图 | 误报 | 平衡准确率 |
|---|---:|---:|---:|---:|---:|---:|
| TACO 验证集 | 107 / 27 | 98 | 9 | 14 | 13 | 71.72% |
| **TACO 测试集** | **108 / 26** | **95** | **13** | **19** | **7** | **80.52%** |
| Indoor Waste 室内留出集 | 32 / 20 | 23 | 9 | 17 | 3 | 78.44% |

TACO 测试集上，整图判定正确 **114/134 张**，五类数量逐项都正确的有 **53/134 张**。即使一张图正确判为有垃圾，其中的第二个物体仍可能漏掉，物体也可能分错类。

### 各类物体的检测精度

下表按真实物体个数统计。`AP50:95` 同时考虑类别和框的位置；无该类真值用「—」表示，不能记成零分。

| 类别 | TACO 测试目标数 | TACO AP50 | TACO AP50:95 | 室内留出目标数 | 室内 AP50:95 |
|---|---:|---:|---:|---:|---:|
| 罐 | 27 | 35.76% | 26.49% | 0 | — |
| 瓶 | 37 | 42.23% | 28.45% | 0 | — |
| 包装物 | 71 | 49.31% | 34.61% | 9 | 58.64% |
| 杯 | 17 | 47.22% | 28.19% | 0 | — |
| 纸类 | 11 | 39.97% | 34.20% | 26 | 53.43% |

Indoor holdout 的真值只覆盖包装物和纸类；罐、瓶、杯的 AP 无法估计，这三类的误检仍计入整图判定和计数误差。TACO 的 PASS 由项目映射后的官方标注决定，未进行逐图负样本复核。独立吸管属于背景，但默认权重在阈值 0.25 下会把部分细长背景物误判为包装物。195 张本地图片都用于训练，不能用来估计新本地场景的性能。遮挡、小目标和相邻同类物体会影响检出和计数，逐帧预测也不会跨帧跟踪；图像判定或总数正确时，物体位置及类别仍可能有误。
