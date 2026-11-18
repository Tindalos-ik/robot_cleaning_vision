# 重建补充数据集

在项目根目录执行以下命令。重建使用[已保存的审核清单](../../data/processed/supplemental/)，无需重新挑图或标框。

## 准备原始资料

| 资料 | 本地位置 | 用途 |
|---|---|---|
| [CoxPWD v3](https://data.mendeley.com/datasets/bdzg4tjy63/3) 官方 ZIP | `data/downloads/coxpwd-v3.zip` | 恢复 110 张训练图和原始框，核对官方留出分区 |
| [Indoor Waste v1](https://data.mendeley.com/datasets/dy7smxr93r/1) 官方图片和标注 ZIP | `data/downloads/indoor-waste-v1-images.zip`、`data/downloads/indoor-waste-v1-annotations.zip` | 恢复 52 张训练图和 52 张室内留出图 |
| Wikimedia Commons 的 21 张原图 | `data/raw/supplemental/wikimedia/` | 请保留审核时的文件和文件名。缺图时，恢复程序按清单下载并核对 SHA-256 |
| TACO 项目标注 | `data/raw/taco/annotations_project.json` | 提供图像 ID、场景组及查重签名；准备方法见 [TACO 指南](../taco/README.md) |

ZIP、原图和权重保存在 Git 仓库外。Commons 的 [审核清单](../../data/processed/supplemental/wikimedia_reviews.json) 记录作者、许可、网址和图片摘要；网络不可用或下载内容已变时，用保存的原图恢复。需要重现相同的查重记录时，保持 `data/raw/local/` 参与查重的图片集合一致。

## 恢复图片和标注

```bash
uv sync --locked
uv run --locked python -m dataprocessing.supplemental --restore-images
uv run --locked python -m dataprocessing.supplemental.indoor --restore-images
```

先运行第一条转换命令，生成 `data/processed/supplemental/annotations_project.json`；室内转换读取这份文件进行查重，再生成同目录的 `indoor_annotations_project.json`。图片已齐时，两条命令都可以省略 `--restore-images`。室内转换会核验已有的 `indoor_holdout.json` 和 `indoor_holdout.sha256`，复现时保留这两个文件。

| 分区 | PASS | FAIL | 总数 |
|---|---:|---:|---:|
| CoxPWD 训练 | 70 | 40 | 110 |
| Wikimedia 训练 | 21 | 0 | 21 |
| Indoor Waste 训练 | 15 | 37 | 52 |
| Indoor Waste 留出 | 20 | 32 | 52 |

## 维护清单

[`mapping.py`](mapping.py) 定义外部类别到项目五类的映射。三个 `*_reviews.json` 保存逐图决定、场景组、标签及目标框的保留或修正依据。更新数据时先改相应审核清单，再运行转换。模型训练使用 183 张补充训练图；52 张室内留出图单独用于评估。用法和审核字段见[数据准备与训练](../README.md)。
