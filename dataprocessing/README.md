# 三模型共用 TACO 数据

```bash
uv run python -m dataprocessing --input data/raw/taco --output data/raw/taco/annotations_project.json --seed 42
uv run python -m dataprocessing --help
```

输出 `annotations_project.json`，复用原图。物体真值来自官方 `annotations.json`。

```python
from pathlib import Path
from dataprocessing import ProjectDataset

root = Path('data/raw/taco')
train = ProjectDataset(root, root / 'annotations_project.json', 'train')
sample = train[0]
image = sample['image']  # PIL RGB，已应用 EXIF 方向
label = sample['label']  # PASS / FAIL
counts = sample['counts']  # 字符串键 1..5
objects = sample['objects']  # 五类 category_id、原始像素 bbox xywh 和 xmin/ymin/xmax/ymax
```

三种模型使用同一接口、同一批正式图片和固定 train/val/test；图像模型取 image/label，检测模型额外取 objects。增强只在划分之后进行。

## 标签与来源

类别 1..5 为罐、瓶、包装物、杯、纸类物品；纸杯归杯，纸袋和包装纸归包装物。完整原始类别映射及依据保存在 `project.mapping`。

PASS 表示官方标注经项目映射后无目标，未额外全图核验；有效目标计数非零为 FAIL。类别待定、无效框、影响计数的 crowd/ignore 或尺寸不符等图片留出，见 `project.held_out`。背景物体本身不排除图片。

标准 COCO 主体只含正式图片和有效目标，bbox/segmentation 沿用官方显示方向，不二次旋转、不裁剪。项目 annotation `id`/`annotation_id` 等于官方数组下标加一，`source_index` 为零基下标，`object_id` 保留官方编号（即使重复）。来源路径、图像ID、原类别均可回查。

## 划分与数据说明

三种模型共用固定 train/val/test，目标比例 70%/15%/15%，seed=42。相关图片按场景组分配，增强在划分后进行。

实际规模、分组规则、来源许可和数据局限见 [数据集说明](../docs/TACO数据集.md)。
