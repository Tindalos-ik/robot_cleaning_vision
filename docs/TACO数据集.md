# TACO 数据集

三种模型共用 `data/raw/taco/annotations_project.json` 和其中固定的 train/val/test。原图仍按官方 `file_name` 位于 `data/raw/taco/`，不复制图片。生成命令、统一读取示例见 [dataprocessing/README.md](../dataprocessing/README.md)。

```bash
uv run python -m dataprocessing --input data/raw/taco --output data/raw/taco/annotations_project.json --seed 42
```

## 官方真值与标签口径

原始数据包含 1,500 张图片、4,784 条物体标注、60 个细分类别和 28 个上级类别。项目标注仅由官方 `annotations.json` 按固定规则转换。

21 类映射为目标、25 类为背景、14 类待定。完整映射及依据见 `project.mapping`。纸杯归杯；纸袋和包装纸归包装物。

正式图片计数非零为 FAIL，五类全零为 PASS。**PASS 仅表示官方标注经项目映射后无目标，未额外完成人工或 AI 全图核验。** 背景本身不排除图片；待定类别、无效框或影响准确计数的 crowd/ignore 等按图片留出。同组其他完整图片继续保留。

读取原图时应用 `ImageOps.exif_transpose`，框和分割不重复旋转。框保留 COCO 像素 xywh，接口同时提供 xyxy。

## 实际处理结果（seed=42，转换版本 official-project-2.0）

正式使用 **889 张、1,064 个目标实例**，留出 **611 张**。正式 FAIL **712** 张、PASS **177** 张。

| 类别 | 全部正式实例 | train | val | test |
|---|---:|---:|---:|---:|
| 罐 | 165 | 114 | 24 | 27 |
| 瓶 | 246 | 172 | 37 | 37 |
| 包装物 | 469 | 328 | 70 | 71 |
| 杯 | 109 | 76 | 16 | 17 |
| 纸类物品 | 75 | 53 | 11 | 11 |

| 分区 | 图片 | 比例 | 场景组 | FAIL | PASS |
|---|---:|---:|---:|---:|---:|
| train | 621 | 69.85% | 449 | 497 | 124 |
| val | 134 | 15.07% | 113 | 107 | 27 |
| test | 134 | 15.07% | 121 | 108 | 26 |

留出原因：类别待定 608 张、无效框 7 张，其中 4 张重叠。逐图原因见 `project.held_out`。

## 场景分组

按源 URL、文件及方向归一像素哈希、相机拍摄日期、GPS 和确定性图像相似性建立场景组，疑似关联保守合并。batch 仅表示存储位置，缺少 EXIF/GPS 不排除图片。

按组分配 train/val/test，目标比例 70%/15%/15%，seed=42；val 固定，不做 K 折。标注不完整的图片单独留出，同组完整图片可保留，但不跨分区。增强在划分后进行。分组规则和依据见 `project.grouping`。

## 数据局限

官方标注可能漏标或错标；PASS 不等于额外核验后的无垃圾场景。纸板片缺少可直接无歧义映射的原始类别。自动分组可能过度合并或遗漏关联，不能保证绝对场景独立。TACO 与本地机器人视角存在域差异。

## 来源与许可

- [官方仓库](https://github.com/pedropro/TACO)
- [官方标注](https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json)
- [官方归档](https://zenodo.org/records/3587843)
- [许可说明](http://tacodataset.org/)

标注采用 CC BY 4.0；图片按各自来源许可使用。项目 JSON 保留官方来源及许可字段，原始 `licenses` 数组为空。
