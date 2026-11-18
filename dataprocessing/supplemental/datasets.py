"""为 MLP 提供补充来源选项，并读取共用训练分区及室内分区。"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from dataprocessing.taco import ProjectDataset

_REPO = Path(__file__).resolve().parents[2]
SUPPLEMENTAL_ROOT = _REPO / 'data/raw/supplemental'
SUPPLEMENTAL_ANNOTATIONS = _REPO / 'data/processed/supplemental/annotations_project.json'
INDOOR_ANNOTATIONS = SUPPLEMENTAL_ANNOTATIONS.with_name('indoor_annotations_project.json')
INDOOR_HOLDOUT = SUPPLEMENTAL_ANNOTATIONS.with_name('indoor_holdout.json')
INDOOR_HOLDOUT_SHA256 = INDOOR_HOLDOUT.with_suffix('.sha256')
_BASE_MODES = ('none', 'coxpwd', 'all')


class TrainingDataset:
    """按顺序拼接训练分区，保留各自的图片根目录和样本格式。"""

    split = 'train'

    def __init__(self, first: ProjectDataset, *extra: ProjectDataset) -> None:
        """拼接 first 和各 extra 训练分区；不同分区的图像、场景组及标注 ID 不得冲突。"""
        self._parts = (first, *extra)
        self.images = []
        image_ids, groups, annotation_ids = set(), set(), set()
        for dataset in self._parts:
            if dataset.split != self.split:
                raise ValueError(f'不能拼接非训练分区：{dataset.split}')
            current_ids = {record['id'] for record in dataset.images}
            current_groups = {record['group_id'] for record in dataset.images}
            current_annotations = {o['id'] for objects in dataset.objects.values() for o in objects}
            if len(current_ids) != len(dataset) or image_ids & current_ids:
                raise ValueError('训练分区的图像 ID 冲突')
            if groups & current_groups or annotation_ids & current_annotations:
                raise ValueError('训练分区的场景组或标注 ID 冲突')
            image_ids.update(current_ids)
            groups.update(current_groups)
            annotation_ids.update(current_annotations)
            self.images.extend(dataset.images)

    def __len__(self) -> int:
        """返回训练分区的样本总数。"""
        return len(self.images)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """按 index 读取单样本，负索引以组合分区末尾为基准。"""
        if index < 0:
            index += len(self)
        if not 0 <= index < len(self):
            raise IndexError(index)
        for dataset in self._parts:
            if index < len(dataset):
                return dataset[index]
            index -= len(dataset)
        raise IndexError(index)


def add_supplemental_argument(parser: ArgumentParser) -> None:
    """为 parser 添加 MLP 的补充来源选项。"""
    help_text = '训练补充来源：none 不启用，coxpwd 只用 CoxPWD，all 使用 CoxPWD 和 Wikimedia'
    parser.add_argument('--supplemental', choices=_BASE_MODES, default='none', help=help_text)


def supplemental_dataset(supplemental: str) -> ProjectDataset:
    """按 supplemental（coxpwd 或 all）读取已审核补充训练分区。"""
    if supplemental not in {'coxpwd', 'all'}:
        raise ValueError(f'无效的补充来源：{supplemental}')
    dataset = ProjectDataset(SUPPLEMENTAL_ROOT, SUPPLEMENTAL_ANNOTATIONS, 'train')
    if supplemental == 'coxpwd':
        dataset.images = [record for record in dataset.images if record['source'] == supplemental]
        ids = {record['id'] for record in dataset.images}
        for image_id in dataset.objects.keys() - ids:
            del dataset.objects[image_id]
    if not dataset.images:
        raise ValueError(f'补充来源没有审核图像：{supplemental}')
    return dataset


def indoor_dataset(split: str) -> ProjectDataset:
    """返回单独的室内 split（train 或 test）；拒绝空分区。"""
    if split not in {'train', 'test'}:
        raise ValueError(f'室内数据没有分区：{split}')
    dataset = ProjectDataset(SUPPLEMENTAL_ROOT, INDOOR_ANNOTATIONS, split)
    if not len(dataset):
        raise ValueError(f'室内分区没有审核图像：{split}')
    return dataset


def training_dataset(taco_root: Path, taco_annotations: Path,
                     supplemental: str) -> ProjectDataset | TrainingDataset:
    """从 taco_root 和 taco_annotations 读取原 train，按 supplemental 追加审核图。"""
    taco = ProjectDataset(taco_root, taco_annotations, 'train')
    return taco if supplemental == 'none' else TrainingDataset(taco, supplemental_dataset(supplemental))
