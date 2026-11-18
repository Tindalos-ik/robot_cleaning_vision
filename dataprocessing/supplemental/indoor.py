"""从 Indoor Waste v1 原始 ZIP 和逐图审核决定生成独立补充标注与冻结留出清单。"""

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from contextlib import ExitStack
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from dataprocessing.taco.images import file_hash, inspect_image, source_path
from dataprocessing.taco.mapping import PROJECT_CATEGORIES

from .convert import _box, _check_duplicates, _restore, load_json, write_json
from .mapping import INDOOR_BACKGROUNDS, INDOOR_RULES

_BASE = Path('data/processed/supplemental')
_ID = re.compile(r'video_\d{8}_\d{6}_\d{5}')
_SHA = re.compile(r'[0-9a-f]{64}')
_SIGNATURE_KEYS = ('source_sha256', 'pixel_sha256', 'phash', 'dhash')


def _members(archive: ZipFile, suffix: str) -> dict:
    """检查 archive 的平铺成员与 suffix，按官方字符串 ID 建立唯一索引。"""
    members = {}
    for info in archive.infolist():
        name = PurePosixPath(info.filename)
        if (info.is_dir() or len(name.parts) != 1 or '\\' in info.filename
                or name.name != info.filename or name.suffix != suffix
                or not _ID.fullmatch(name.stem) or info.flag_bits & 1):
            raise ValueError(f'不安全或未知的 Indoor ZIP 成员：{info.filename}')
        if name.stem in members:
            raise ValueError(f'Indoor ZIP 成员重复：{info.filename}')
        members[name.stem] = info.filename
    return members


def _references(path: Path, name: str) -> tuple[dict, list]:
    """从 path 读取 name 来源的完整生成签名，拒绝缺少图像覆盖的参考产物。"""
    data = load_json(path)
    if data['categories'] != PROJECT_CATEGORIES:
        raise ValueError(f'{name} 项目类别与共享类别不一致：{path}')
    ids = [str(row['id']) for row in data['images']]
    checks = data['project']['image_checks']
    if not ids or len(set(ids)) != len(ids) or not set(ids) <= checks.keys():
        raise ValueError(f'{name} 查重签名未完整覆盖图像：{path}')
    for image_id, signature in checks.items():
        for key in _SIGNATURE_KEYS:
            size = 64 if key.endswith('sha256') else 16
            value = signature.get(key)
            if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{' + str(size) + '}', value):
                raise ValueError(f'{name}:{image_id} 缺少有效查重签名 {key}')
    return data, [(f'{name}:{image_id}', signature) for image_id, signature in checks.items()]


def _targets(review: dict, original: dict) -> list:
    """用 review 显式处理 original 的每个原框，并检查新增目标及显示像素框。"""
    image_id = review['original_id']
    if (original.get('version') != '5.1.1'
            or (original.get('imageWidth'), original.get('imageHeight')) != (1280, 720)
            or PurePosixPath(original['imagePath'].replace('\\', '/')).name != image_id + '.jpg'):
        raise ValueError(f'{image_id}：官方标注版本、尺寸或图片名不符')
    shapes = original['shapes']
    categories = sorted({shape['label'] for shape in shapes})
    if categories != review['source_categories'] or not set(categories) <= INDOOR_RULES.keys():
        raise ValueError(f'{image_id}：审核来源类别与官方形状不符：{categories}')
    originals = {}
    for index, shape in enumerate(shapes):
        points = shape['points']
        if (shape.get('shape_type') != 'rectangle' or len(points) != 2
                or any(len(point) != 2 for point in points)
                or any(type(v) not in (int, float) or not math.isfinite(v) for p in points for v in p)):
            raise ValueError(f'{image_id}：原框 {index} 须为有限数值的两角矩形')
        (x1, y1), (x2, y2) = points
        originals[index] = [min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)]
    resolved, targets = set(), []
    for item in review['objects']:
        index, category = item['source_index'], item['category_id']
        if type(category) is not int or category not in range(1, 6) or not str(item['reason']).strip():
            raise ValueError(f'{image_id}：目标类别或依据无效：{item}')
        if index is None:
            box = item['bbox']
            source_category = None
        else:
            if type(index) is not int or index not in originals or index in resolved:
                raise ValueError(f'{image_id}：目标原框索引无效或重复：{index}')
            resolved.add(index)
            source_category = shapes[index]['label']
            if INDOOR_RULES[source_category][0] == 'background' and item.get('override') is not True:
                raise ValueError(f'{image_id}：改判原框 {index} ({source_category}) 需明确记录 override')
            box = item.get('bbox', originals[index])
        if (not isinstance(box, list) or len(box) != 4
                or any(type(v) not in (int, float) for v in box)):
            raise ValueError(f'{image_id}：目标原框 {index} 的 bbox 格式无效：{box}')
        try:
            _box(box, 1280, 720)
        except ValueError as error:
            raise ValueError(f'{image_id}：目标原框 {index} 无效：{error}') from error
        targets.append({
            'category_id': category, 'bbox': box, 'area': box[2] * box[3], 'iscrowd': 0,
            'source_index': index, 'original_category': source_category,
            'box_source': 'manual' if 'bbox' in item else 'original',
            'mapping_reason': item['reason'],
        })
    for item in review['discarded']:
        index = item['source_index']
        if (type(index) is not int or index not in originals or index in resolved
                or not str(item['reason']).strip()):
            raise ValueError(f'{image_id}：排除原框索引或依据无效：{item}')
        resolved.add(index)
    if resolved != set(originals):
        raise ValueError(f'{image_id}：原框未逐个处理：{sorted(set(originals) - resolved)}')
    return targets


def _holdout_rows(images: list) -> list:
    """从已验证 images 的 test 记录派生可冻结且顺序稳定的留出清单。"""
    return [{
        'image_id': row['id'], 'original_id': row['source_image_id'],
        'file_name': row['file_name'], 'sha256': row['sha256'], 'group_id': row['group_id'],
        'label': row['label'], 'source_categories': row['source_categories'],
        'background': row['background'], 'counts': row['counts'], 'review_note': row['review_note'],
    } for row in images if row['split'] == 'test']


def _frozen(path: Path, rows: list, freeze: bool) -> None:
    """将 rows 与 path 及 SHA256 旁文件核对；freeze 只允许首次独占创建完整文件对。"""
    sidecar = path.with_suffix('.sha256')
    if path.exists() != sidecar.exists():
        raise ValueError(f'留出清单与摘要必须成对存在：{path} / {sidecar}')
    if path.exists():
        expected = file_hash(path) + '\n'
        if sidecar.read_bytes() != expected.encode('ascii'):
            raise ValueError(f'冻结留出清单 SHA256 不符：{path}')
        if load_json(path) != rows:
            raise ValueError(f'实际验证 test 记录与冻结留出清单不同：{path}')
        if freeze:
            raise ValueError(f'留出清单已冻结，拒绝覆盖：{path}')
        return
    if not freeze:
        raise ValueError(f'缺少冻结留出清单，请在首次预测前使用 --freeze-holdout：{path}')
    payload = (json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(payload)
    with sidecar.open('xb') as stream:
        stream.write((hashlib.sha256(payload).hexdigest() + '\n').encode('ascii'))


def convert(root: Path, downloads: Path, source: Path, reviews_path: Path,
            taco_annotations: Path, supplemental_annotations: Path, output: Path,
            holdout: Path, restore: bool = False, freeze: bool = False) -> dict:
    """按 source、reviews_path 审核 root 原图，以两个只读参考查重后写 output 与 holdout。

    downloads 提供官方 ZIP；restore 仅恢复缺失原件；freeze 只首次创建留出清单。
    """
    protected = {path.resolve() for path in (source, reviews_path, taco_annotations,
                                            supplemental_annotations, holdout, holdout.with_suffix('.sha256'))}
    if output.resolve() in protected:
        raise ValueError(f'输出会覆盖输入或冻结清单：{output}')
    provenance, reviews = load_json(source), load_json(reviews_path)
    holdout_paths = {holdout.resolve(), holdout.with_suffix('.sha256').resolve()}
    input_paths = {path.resolve() for path in (source, reviews_path, taco_annotations,
                                              supplemental_annotations)}
    if holdout_paths & input_paths:
        raise ValueError(f'留出路径会覆盖输入：{holdout}')
    if len(holdout_paths) != 2:
        raise ValueError(f'留出清单与摘要路径必须不同：{holdout}')
    if not isinstance(reviews, list) or not reviews:
        raise ValueError(f'审核清单须为非空 JSON 数组：{reviews_path}')
    if str(provenance['version']) != '1':
        raise ValueError('仅支持 Indoor Waste Image Dataset v1')
    taco, references = _references(taco_annotations, 'TACO')
    supplemental, extra = _references(supplemental_annotations, 'supplemental')
    references.extend(extra)
    used_ids = {r['id'] for data in (taco, supplemental) for r in data['images']}
    used_paths = {r['file_name'].casefold() for data in (taco, supplemental) for r in data['images']}
    used_groups = {r['group_id'] for data in (taco, supplemental) for r in data['images']}
    used_targets = {r['id'] for data in (taco, supplemental) for r in data['annotations']}
    images, annotations, checks, selected = [], [], {}, []
    groups, clips, reviewed_ids = {}, {}, set()
    with ExitStack() as stack:
        archives, archive_paths = {}, {}
        for name in ('images', 'annotations'):
            metadata = provenance['archives'][name]
            filename = metadata['filename']
            if (not isinstance(filename, str) or Path(filename).name != filename
                    or '/' in filename or '\\' in filename or not filename.endswith('.zip')
                    or not _SHA.fullmatch(metadata['sha256']) or not metadata['url'].startswith('https://')):
                raise ValueError(f'来源 ZIP 路径、摘要或 URL 无效：{name}')
            path = source_path(downloads, filename)
            if file_hash(path) != metadata['sha256']:
                raise ValueError(f'Indoor {name} 官方归档 SHA256 校验失败：{path}')
            archive_paths[name] = path.resolve()
            archives[name] = stack.enter_context(ZipFile(path))
        if ({output.resolve()} | holdout_paths) & set(archive_paths.values()):
            raise ValueError('产物路径与官方 ZIP 冲突')
        image_members = _members(archives['images'], '.jpg')
        annotation_members = _members(archives['annotations'], '.json')
        if len(image_members) != 4000 or image_members.keys() != annotation_members.keys():
            raise ValueError('Indoor v1 须有 4000 个按同名 stem 配对的 JPG/JSON')
        ordinals = {original_id: index for index, original_id in enumerate(sorted(image_members))}
        for review in reviews:
            required = {'original_id', 'sha256', 'source_categories', 'group', 'split',
                        'label', 'note', 'background', 'objects', 'discarded'}
            if not isinstance(review, dict) or not required <= review.keys():
                raise ValueError(f'审核记录缺少必要字段：{review}')
            for key in ('source_categories', 'background', 'objects', 'discarded'):
                if not isinstance(review[key], list):
                    raise ValueError(f'{review["original_id"]}：{key} 须为数组')
            for key in ('objects', 'discarded'):
                fields = {'source_index', 'reason'} | ({'category_id'} if key == 'objects' else set())
                for item in review[key]:
                    if (not isinstance(item, dict) or not fields <= item.keys()
                            or not isinstance(item['reason'], str) or not item['reason'].strip()):
                        raise ValueError(f'{review["original_id"]}：{key} 决定缺少索引、类别或依据：{item}')
            original_id = review['original_id']
            if not isinstance(original_id, str) or not _ID.fullmatch(original_id) or original_id not in ordinals:
                raise ValueError(f'审核 ID 未对应官方图像和标注成员：{original_id}')
            if original_id in reviewed_ids:
                raise ValueError(f'审核 ID 重复：{original_id}')
            reviewed_ids.add(original_id)
        for review in sorted(reviews, key=lambda row: row['original_id']):
            original_id = review['original_id']
            split, group = review['split'], review['group']
            if (split not in {'train', 'test'} or not isinstance(group, str) or not group.strip()
                    or group != group.strip() or not isinstance(review['note'], str) or not review['note'].strip()
                    or not isinstance(review['sha256'], str) or not _SHA.fullmatch(review['sha256'])):
                raise ValueError(f'{original_id}：划分、组名、审核说明或摘要无效')
            clip = original_id.rsplit('_', 1)[0]
            for key, assignments, kind in ((group, groups, '场景组'), (clip, clips, '拍摄 clip')):
                if key in assignments and assignments[key] != split:
                    raise ValueError(f'{original_id}：{kind} {key} 跨 train/test')
                assignments[key] = split
            background = review['background']
            if (not isinstance(background, list) or any(not isinstance(v, str) for v in background)
                    or len(background) != len(set(background)) or not set(background) <= INDOOR_BACKGROUNDS):
                raise ValueError(f'{original_id}：背景标记无效：{background}')
            original = json.loads(archives['annotations'].read(annotation_members[original_id]))
            targets = _targets(review, original)
            counts = Counter(target['category_id'] for target in targets)
            count_record = {str(i): counts[i] for i in range(1, 6)}
            label = 'FAIL' if targets else 'PASS'
            if review['label'] != label or ('counts' in review and review['counts'] != count_record):
                raise ValueError(f'{original_id}：审核标签或计数与目标不一致：{label} {count_record}')
            relative = f'indoor_waste/{original_id}.jpg'
            if 'file_name' in review and review['file_name'] != relative:
                raise ValueError(f'{original_id}：审核图像路径与官方 ID 不符')
            image_id = 2_000_000_000 + ordinals[original_id]
            group_id = f'supplemental:indoor_waste:{group}'
            if image_id in used_ids or relative.casefold() in used_paths or group_id in used_groups:
                raise ValueError(f'{original_id}：与既有产物的 ID、路径或组名冲突')
            official_bytes = archives['images'].read(image_members[original_id])
            if hashlib.sha256(official_bytes).hexdigest() != review['sha256']:
                raise ValueError(f'{original_id}：审核摘要与官方图像成员不同')
            path = source_path(root, relative)
            if restore and not path.exists():
                _restore(path, official_bytes)
            if file_hash(path) != review['sha256']:
                raise ValueError(f'{original_id}：本地图像与审核原件不同：{path}')
            record = {
                'id': image_id, 'file_name': relative, 'width': 1280, 'height': 720,
                'split': split, 'source': 'indoor_waste', 'source_image_id': original_id,
                'source_path': image_members[original_id], 'source_annotation_path': annotation_members[original_id],
                'group_id': group_id, 'clip_id': clip, 'sha256': review['sha256'],
                'label': label, 'counts': count_record, 'background': background,
                'source_categories': review['source_categories'], 'review_note': review['note'],
                'source_annotations': original['shapes'], 'discarded': review['discarded'],
                **{key: provenance[key] for key in ('doi', 'source_url', 'author', 'license', 'license_url')},
            }
            signature, issues = inspect_image(record, root)
            if issues or signature['exif_orientation'] != 1:
                raise ValueError(f'{original_id}：原图尺寸或方向无效：{issues}，EXIF={signature["exif_orientation"]}')
            _check_duplicates(record, signature, references)
            _check_duplicates(record, signature, selected)
            selected.append((f'{original_id} ({split})', signature))
            checks[str(image_id)] = {key: signature[key] for key in
                                     (*_SIGNATURE_KEYS, 'exif_orientation', 'display_size')}
            images.append(record)
            if len(targets) > 100:
                raise ValueError(f'{original_id}：目标数超过稳定标注 ID 的 100 个上限')
            for index, target in enumerate(targets):
                target_id = image_id * 100 + index
                if target_id in used_targets:
                    raise ValueError(f'{original_id}：标注 ID 冲突：{target_id}')
                annotations.append({**target, 'id': target_id, 'image_id': image_id})
    if set(groups.values()) != {'train', 'test'}:
        raise ValueError('Indoor 审核清单必须同时包含 train 和 test')
    statistics = {}
    for split in ('train', 'test'):
        rows = [row for row in images if row['split'] == split]
        statistics[split] = {
            'images': len(rows), 'groups': len({row['group_id'] for row in rows}),
            'clips': len({row['clip_id'] for row in rows}),
            'labels': {label: sum(row['label'] == label for row in rows) for label in ('PASS', 'FAIL')},
            'targets': {str(i): sum(row['counts'][str(i)] for row in rows) for i in range(1, 6)},
            'background_images': {key: sum(key in row['background'] for row in rows)
                                  for key in sorted(INDOOR_BACKGROUNDS)},
        }
    holdout_rows = _holdout_rows(images)
    _frozen(holdout, holdout_rows, freeze)
    result = {
        'images': images, 'annotations': annotations, 'categories': PROJECT_CATEGORIES,
        'project': {
            'source': provenance, 'mapping': INDOOR_RULES, 'statistics': statistics,
            'image_checks': checks, 'reviews_sha256': file_hash(reviews_path),
            'holdout_sha256': file_hash(holdout),
            'stable_ids': {'base': 2_000_000_000, 'ordinal': 'zero-based sorted official image stem',
                           'annotation': 'image_id * 100 + zero-based reviewed target ordinal'},
            'duplicate_check': {
                'taco_images': len(taco['project']['image_checks']),
                'supplemental_images': len(supplemental['project']['image_checks']),
                'supplemental_sources': dict(Counter(row['source'] for row in supplemental['images'])),
                'taco_annotations_sha256': file_hash(taco_annotations),
                'supplemental_annotations_sha256': file_hash(supplemental_annotations),
                'selected_images': len(images), 'selected_pairs': len(images) * (len(images) - 1) // 2,
                'phash_max_distance': 10, 'dhash_max_distance': 16,
                'exact_signatures': ['source_sha256', 'pixel_sha256'],
                'selected_pair_policy': 'reject all flagged pairs, including same-split pairs',
            },
        },
    }
    write_json(output, result)
    return result


def main() -> None:
    """解析官方归档、审核与产物路径及两个开关，执行转换并打印实际统计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images', type=Path, default=Path('data/raw/supplemental'))
    parser.add_argument('--downloads', type=Path, default=Path('data/downloads'))
    parser.add_argument('--source', type=Path, default=_BASE / 'indoor_source.json')
    parser.add_argument('--reviews', type=Path, default=_BASE / 'indoor_reviews.json')
    parser.add_argument('--taco-annotations', type=Path, default=Path('data/raw/taco/annotations_project.json'))
    parser.add_argument('--supplemental-annotations', type=Path, default=_BASE / 'annotations_project.json')
    parser.add_argument('--output', type=Path, default=_BASE / 'indoor_annotations_project.json')
    parser.add_argument('--holdout', type=Path, default=_BASE / 'indoor_holdout.json')
    parser.add_argument('--restore-images', action='store_true', help='仅从已校验官方 ZIP 恢复缺失审核原图')
    parser.add_argument('--freeze-holdout', action='store_true', help='首次预测前创建尚不存在的留出清单及摘要')
    args = parser.parse_args()
    result = convert(args.images, args.downloads, args.source, args.reviews,
                     args.taco_annotations, args.supplemental_annotations,
                     args.output, args.holdout, args.restore_images, args.freeze_holdout)
    print(json.dumps(result['project']['statistics'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
