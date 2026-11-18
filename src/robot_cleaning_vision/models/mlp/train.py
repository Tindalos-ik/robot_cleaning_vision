"""训练区域与整图 MLP，支持补充训练数据及真实特征检查。"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from dataprocessing.taco import ProjectDataset
from dataprocessing.supplemental.datasets import (
    SUPPLEMENTAL_ANNOTATIONS, add_supplemental_argument, supplemental_dataset,
    training_dataset,
)

from .training import prepare_dataset, train_model

# ===== 数据与保存位置 =====
PROJECT_ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = PROJECT_ROOT / 'data/raw/taco'
ANNOTATIONS = DATA_ROOT / 'annotations_project.json'
OUTPUT_ROOT = PROJECT_ROOT / 'artifacts/mlp/training'

# ===== 固定的特征与网络方案 =====
# 长边640的等比图上生成多尺度窗口；区域HSV160，整图为区域统计30＋背景58。
# 网络结构固定为区域160→32→1、整图88→8→1，与现有推理代码一致。
PROPOSAL = {
    'long_edge': 640,
    'scales': [0.12, 0.24, 0.48, 0.9],
    'aspects': [0.5, 1.0, 2.0],
    'stride': 0.8,
}

# 区域网络沿用选定方案的17轮。折外训练与最终区域训练使用同样轮数，
# 不根据当前留出折选择轮次，保证它的标签不影响对应区域模型的训练。
REGION_CONFIG = {
    'epochs': 17,
    'learning_rate': 0.001,
    'weight_decay': 0.001,
    'batch_size': 128,
}

# 整图网络用验证损失早停；训练结束后恢复验证损失最低的权重。
IMAGE_CONFIG = {
    'epochs': 200,
    'patience': 25,
    'learning_rate': 0.001,
    'weight_decay': 0.01,
    'batch_size': 32,
}
FOLDS = 3  # 按原图的group_id分折，同一场景不能跨折。
SEED = 42
MINIMUM_FAIL_RECALL = 0.9  # 仅在验证集上选择满足此召回率的整图阈值。


def data_provenance(supplemental):
    """按 supplemental 模式记录训练标注路径和 SHA256；none 不读取补充标注。"""
    report = {
        'supplemental': supplemental,
        'annotations': str(ANNOTATIONS),
        'annotations_sha256': hashlib.sha256(ANNOTATIONS.read_bytes()).hexdigest(),
    }
    if supplemental != 'none':
        report.update(
            supplemental_annotations=str(SUPPLEMENTAL_ANNOTATIONS),
            supplemental_annotations_sha256=hashlib.sha256(
                SUPPLEMENTAL_ANNOTATIONS.read_bytes()).hexdigest(),
        )
    return report


def check_data(dataset, supplemental, provenance):
    """从 dataset 提取真实训练区域，按 supplemental 模式检查标签并打印 provenance。"""
    if not len(dataset):
        raise ValueError('检查数据不能为空')
    if supplemental != 'none':
        cox_labels = {image['label'] for image in dataset.images if image['source'] == 'coxpwd'}
        if cox_labels != {'PASS', 'FAIL'}:
            raise ValueError('Cox 数据必须同时包含 PASS 和 FAIL')
    bags, patches = prepare_dataset(dataset, PROPOSAL, training=True, seed=SEED)
    labels = np.asarray([bag['label'] for bag in bags])
    if not np.isfinite(patches['features']).all():
        raise ValueError('MLP 区域特征包含非有限值')
    report = {
        'data_provenance': provenance,
        'images': len(bags),
        'labels': {'PASS': int((labels == 0).sum()), 'FAIL': int((labels == 1).sum())},
        'groups': len({bag['group_id'] for bag in bags}),
        'regions': len(patches['labels']),
        'region_labels': {
            'background': int((patches['labels'] == 0).sum()),
            'target': int((patches['labels'] == 1).sum()),
        },
        'region_feature_shape': list(patches['features'].shape),
        'window_count': sum(len(bag['features']) for bag in bags),
    }
    if supplemental != 'none' and set(patches['labels'].tolist()) != {0, 1}:
        raise ValueError('补充检查数据必须生成背景和目标区域')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def main(argv=None):
    """解析 argv，检查真实数据或按原有验证分区执行训练。"""
    parser = argparse.ArgumentParser(description=__doc__)
    add_supplemental_argument(parser)
    parser.add_argument('--check-data', action='store_true', help='提取真实区域特征并打印数据统计，不训练或保存模型')
    args = parser.parse_args(argv)
    provenance = data_provenance(args.supplemental)
    if args.check_data:
        dataset = (supplemental_dataset(args.supplemental) if args.supplemental != 'none'
                   else ProjectDataset(DATA_ROOT, ANNOTATIONS, 'train'))
        return check_data(dataset, args.supplemental, provenance)
    # 读取已经划好的正式分区，不重新划分图片，不使用test。
    train_dataset = training_dataset(DATA_ROOT, ANNOTATIONS, args.supplemental)
    val_dataset = ProjectDataset(DATA_ROOT, ANNOTATIONS, 'val')
    return train_model(
        train_dataset,
        val_dataset,
        OUTPUT_ROOT,
        annotations=ANNOTATIONS,
        proposal=PROPOSAL,
        region_config=REGION_CONFIG,
        image_config=IMAGE_CONFIG,
        folds=FOLDS,
        seed=SEED,
        minimum_recall=MINIMUM_FAIL_RECALL,
        provenance=provenance,
    )


if __name__ == '__main__':
    main()
