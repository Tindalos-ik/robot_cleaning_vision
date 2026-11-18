"""从官方归档、原图和已保存的审核决定重新生成补充标注。"""

import argparse
import json
from pathlib import Path

from .convert import convert
from .datasets import SUPPLEMENTAL_ANNOTATIONS, SUPPLEMENTAL_ROOT


def main() -> None:
    """解析命令行路径和恢复开关，执行转换并打印审核图像统计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, default=Path('data/downloads/coxpwd-v3.zip'),
                        help='官方 CoxPWD v3 ZIP，原样保留 train、valid、test')
    parser.add_argument('--images', type=Path, default=SUPPLEMENTAL_ROOT,
                        help='补充图像根目录')
    parser.add_argument('--taco-annotations', type=Path,
                        default=Path('data/raw/taco/annotations_project.json'), help='TACO 只读查重参考')
    parser.add_argument('--local-images', type=Path, default=Path('data/raw/local'),
                        help='本地实测图像根目录，只用于查重')
    parser.add_argument('--output', type=Path, default=SUPPLEMENTAL_ANNOTATIONS,
                        help='补充 COCO 文件，须与两个审核 JSON 同目录')
    parser.add_argument('--restore-images', action='store_true',
                        help='恢复缺失的已审核原图（Wikimedia 需要网络）')
    args = parser.parse_args()
    data = convert(args.images, args.archive, args.taco_annotations, args.local_images,
                   args.output, args.restore_images)
    print(json.dumps(data['project']['statistics'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
