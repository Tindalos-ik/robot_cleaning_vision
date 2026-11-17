"""修改顶部配置后直接运行，训练结果另存目录，保留当前实测模型。"""

from pathlib import Path

from dataprocessing import ProjectDataset

from .training import train_model

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


def main():
    # 读取已经划好的正式分区，不重新划分图片，不使用test。
    train_dataset = ProjectDataset(DATA_ROOT, ANNOTATIONS, 'train')
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
    )


if __name__ == '__main__':
    main()
