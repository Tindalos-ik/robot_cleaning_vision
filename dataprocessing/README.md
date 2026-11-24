# 数据准备与训练

原始 TACO 各分区和项目标注继续沿用，见 [TACO 数据处理](taco/README.md)。补充数据只追加到训练集；验证、测试及本地实测仍各自独立。

## 准备补充图片

- CoxPWD 使用 [官方 v3 归档](https://data.mendeley.com/datasets/bdzg4tjy63/3) 的 `train/`，原封保留 `valid/`、`test/` 在 ZIP 内。将 ZIP 保存为 `data/downloads/coxpwd-v3.zip`。DOI 为 `10.17632/bdzg4tjy63.3`，许可为 CC BY 4.0，作者见[转换代码](supplemental/convert.py)。
- 室内照片来自 [Wikimedia Commons: Vacant rooms](https://commons.wikimedia.org/wiki/Category:Vacant_rooms) 及子分类；按各自文件页署名和使用。清单的 `original_url` 指向原图，若有 `download_url`，使用 Commons 提供的全幅等比版本。
- [CoxPWD 审核清单](../data/processed/supplemental/coxpwd_reviews.json) 与 [Commons 审核清单](../data/processed/supplemental/wikimedia_reviews.json) 保存每张入选图的原始 ID、相对路径、图像 SHA-256、场景组、结论和可见依据。Commons 清单另列作者、文件页和图像许可。未能确认的图不写入清单。

```bash
curl.exe --fail --location --create-dirs --output data/downloads/coxpwd-v3.zip https://data.mendeley.com/public-files/datasets/bdzg4tjy63/files/4967ad1f-8844-4a50-a19f-7da25fa1528a/file_downloaded
uv run --locked python -m dataprocessing.supplemental --restore-images
```

`--restore-images` 只恢复缺失图片到 `data/raw/supplemental/{coxpwd,wikimedia}/`。已备齐图片时省略该开关可离线重新转换。命令校验归档、图片及显示尺寸，输出 `data/processed/supplemental/annotations_project.json`。原 TACO 标注只读；补充图与 TACO、CoxPWD 官方留出、本地实测图片及彼此比对文件/像素哈希、pHash 和 dHash，另排除有可识别时序的同类留出近邻。`group` 用于隔离已识别的相关拍摄场景；缺少定位信息的沙地场景组不等于地理地点数。

仓库保留两份审核清单和生成标注；`data/raw/` 与 `data/downloads/` 是本地数据目录，新工作区按上面的命令恢复图片及归档。

## 维护审核清单

1. 以整图显示方向逐张检查角落、墙边、远景和表面，放大难辨物体；根据 [CoxPWD 映射](supplemental/mapping.py) 复核原框及可能的漏标。PASS 无目标；同图有瓶和独立瓶盖时为 FAIL，只计瓶。
2. `objects` 中每条决定指定项目 `category_id` 及依据 `reason`。`source_index` 是 CoxPWD 官方 `train/_annotations.coco.json` 的零基标注数组索引；保留原框时无需写坐标。改框或补框填原显示方向的像素 `bbox: [x, y, w, h]`，无原框时 `source_index` 为 `null`。改判明确类别时记录 `override: true`，排除原来的目标或待定框时在 `discarded` 写索引和原因。
3. 在 `background` 仅记录图中确认可见的瓶盖、吸管、勺、绳；按图统计，同图可计入多个类别。相同场景使用相同 `group`；同一物和同一场景的近重复照片只留一张。Commons 入选图只收全图无五类目标的 PASS。
4. 运行转换命令。五类计数和标签由保留的目标框验证；MLP 和 CNN 读取同一份生成标注。照片与许可一起保存，外发时保留署名、来源及许可链接。

## 启动训练

本地 JPG 和逐图 JSON 保存在 `data/raw/supplemental/local_training/训练集/`，需从用户原件准备。方法一训练命令见[手工特征 + MLP](../docs/手工特征+MLP.md#4-重新训练)。
方法二还需准备下节室内数据，并按 [本地审核清单](../data/processed/supplemental/local_training_annotations_project.json) 的 `file_name` 提供全部 195 张本地原图。准备完成后运行：

```bash
uv run --locked python -m robot_cleaning_vision.models.cnn_linear.train
```

MLP 的 `--check-data` 只提取特征并打印数据统计。指定补充来源时检查所选补充图；未指定时检查 TACO train。正式训练在 `artifacts/mlp/training/` 新建目录，不覆盖默认权重。

方法二复训使用 TACO train、CoxPWD、Wikimedia、Indoor Waste train 和本地审核实拍图，共 999 张。
首次特征提取需要 torchvision 官方 ResNet-50 权重。每次只保存新权重 `artifacts/cnn_linear/training/<新目录>/model.pt`，终端报告 TACO val 的校准阈值和评估指标，默认权重保持不变。
详细流程见[冻结 CNN + 线性分类器](../docs/冻结CNN+线性分类器.md)。

## Indoor Waste v1：CNN 训练数据

[官方数据](https://data.mendeley.com/datasets/dy7smxr93r/1)由 Yangke L 发布（DOI `10.17632/dy7smxr93r.1`，[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)）。[来源记录](../data/processed/supplemental/indoor_source.json)保存下载链接和 SHA-256。两个 ZIP 都是平铺归档，以文件名配对 JPEG 和 LabelMe JSON。原框是 1280×720 显示图上的两个对角点；官方没有拆分训练、测试。

```bash
curl.exe --fail --location --create-dirs --output data/downloads/indoor-waste-v1-images.zip https://data.mendeley.com/public-files/datasets/dy7smxr93r/files/aed008e5-86af-4d8d-acc0-d74c6cadfac3/file_downloaded
curl.exe --fail --location --create-dirs --output data/downloads/indoor-waste-v1-annotations.zip https://data.mendeley.com/public-files/datasets/dy7smxr93r/files/48e9801f-ca7a-4a3a-8e2d-fa88e5901734/file_downloaded
uv run --locked python -m dataprocessing.supplemental.indoor --restore-images
```

先准备原 TACO 和 CoxPWD/Wikimedia 标注。室内转换读取[审核清单](../data/processed/supplemental/indoor_reviews.json)，校验两个 ZIP、逐图 SHA-256、尺寸及场景分区，使用 TACO 和现有补充来源的图像签名查重。它只恢复缺失原图到 `data/raw/supplemental/indoor_waste/`，生成独立的 `data/processed/supplemental/indoor_annotations_project.json`，并核验[留出清单](../data/processed/supplemental/indoor_holdout.json)及其 [SHA-256](../data/processed/supplemental/indoor_holdout.sha256)。恢复数据使用已冻结的清单。`--freeze-holdout` 只用于在首次预测前创建尚不存在的清单和摘要，不覆盖已有清单。

维护审核清单时使用 [Indoor 类别规则](supplemental/mapping.py)。逐图检查全幅与小物体，合并已识别的同场景、连拍和近重复；每段视频最多一帧。`source_index` 是该图 LabelMe `shapes` 数组零基索引。每个原框须有保留或排除决定。新增或修正框用显示像素 `[x, y, w, h]`；新增时 `source_index` 为 `null`。五类目标数量为零才记 PASS。独立瓶盖、果壳和种壳记背景；散纸计纸类、纸杯计杯、柔性包装纸计包装物，`plastic` 逐件确认。将官方背景框改判为目标时，写明理由及 `override: true`。

CNN 从 Indoor Waste 和本地审核清单各自只取 train 分区。冻结的室内 test 与原 TACO val/test 不加入训练；阈值仅由原 TACO val 确定。

试用新训练权重时，将下例路径替换为训练命令打印的实际路径：

```bash
uv run --locked robot-cleaning-vision run --model baseline2 --model-path artifacts/cnn_linear/training/<新目录>/model.pt --input data/raw/taco/batch_5/000103.JPG
```

拍摄实时画面时将 `--input` 换成 `--camera 0`。两种入口均使用该权重内的预处理配置与阈值。
