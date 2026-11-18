"""将经过全图审核的决定转换为训练用 COCO；保持原图和原划分。"""

import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.request import Request, urlopen
from zipfile import ZipFile

from PIL import Image, ImageOps

from dataprocessing.taco.images import file_hash, inspect_image, source_path
from dataprocessing.taco.mapping import PROJECT_CATEGORIES

from .mapping import COXPWD_RULES, HARD_BACKGROUNDS

COXPWD_SHA256 = '777c04c6792fcb17ef5da62e9ffced4b0154269a66813818e3e1857bb384ba93'
COXPWD_SOURCE = {
    'version': 3,
    'doi': '10.17632/bdzg4tjy63.3',
    'source_url': 'https://data.mendeley.com/datasets/bdzg4tjy63/3',
    'author': ['Tamim Hasan Saykat Tamim', 'Saif Khan', 'Md. Tarek Aziz Leyon',
               'Md Abrar Hossain Zahin', 'Sanvir Rahman', 'Md. Ehsanul Haque',
               'Rakib Hossen', 'MD Mir Hasif'],
    'license': 'CC BY 4.0',
    'license_url': 'https://creativecommons.org/licenses/by/4.0/',
    'original_split': 'train',
}


def load_json(path: Path):
    """读取 path 指向的 UTF-8 JSON。"""
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, data) -> None:
    """把 data 写入 path 指向的产物，拒绝无效浮点值。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def _box(box: list, width: int, height: int) -> None:
    """检查 box 为在 width × height 显示图内的非空 xywh 像素框。"""
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        raise ValueError(f'无效的框：{box}')
    x, y, w, h = box
    if not (x >= 0 and y >= 0 and w > 0 and h > 0 and x + w <= width and y + h <= height):
        raise ValueError(f'框越出显示图 {width}×{height}：{box}')


def _cox_targets(review: dict, originals: dict, width: int, height: int) -> list[dict]:
    """按 review 中逐物体决定转换 originals（原数组索引到标注）；尺寸为显示像素。"""
    resolved = set()
    targets = []
    for item in review.get('objects', []):
        idx, category = item['source_index'], item['category_id']
        if type(category) is not int or category not in range(1, 6) or not item['reason'].strip():
            raise ValueError(f'目标类别或判定依据无效：{item}')
        if idx is None:
            original = None
            bbox = item['bbox']
        else:
            original = originals[idx]
            if idx in resolved:
                raise ValueError(f'原框重复处理：{idx}')
            resolved.add(idx)
            status, mapped, _ = COXPWD_RULES[original['original_category']]
            if (status, mapped) != ('target', category) and status != 'review' and not item.get('override'):
                raise ValueError(f'改判原类别须明确记录 override：{item}')
            bbox = item.get('bbox', original['bbox'])
        _box(bbox, width, height)
        targets.append({
            'category_id': category, 'bbox': bbox, 'area': bbox[2] * bbox[3], 'iscrowd': 0,
            'source_index': idx, 'original_annotation_id': original['id'] if original else None,
            'original_category_id': original['category_id'] if original else None,
            'original_category': original['original_category'] if original else None,
            'box_source': 'manual' if 'bbox' in item else 'original', 'mapping_reason': item['reason'],
        })
    for item in review.get('discarded', []):
        idx = item['source_index']
        if idx not in originals or idx in resolved or not item['reason'].strip():
            raise ValueError(f'无效的排除决定：{item}')
        resolved.add(idx)
    unresolved = {i for i, a in originals.items() if COXPWD_RULES[a['original_category']][0] != 'background'} - resolved
    if unresolved:
        raise ValueError(f"图像 {review['original_id']} 有未处理的原框：{sorted(unresolved)}")
    return targets


def _restore(path: Path, data: bytes) -> None:
    """使用原始字节 data 新建 path，不覆盖本地图像。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as stream:
        stream.write(data)


def _record(source: str, review: dict, root: Path, metadata: dict, width: int, height: int,
            targets: list[dict]) -> dict:
    """将 source 的 review、图源 metadata、尺寸和 targets 组合成 ProjectDataset 图像记录。"""
    original_id, group = review['original_id'], review['group']
    if type(original_id) is not int or not 0 <= original_id < 900_000_000:
        raise ValueError(f'无效原图 ID：{original_id}')
    if not group.strip() or not review['note'].strip():
        raise ValueError(f'缺少场景组或目视审核说明：{source}:{original_id}')
    relative = Path(review['file_name'])
    if relative.parts != (source, relative.name) or relative.stem != str(original_id):
        raise ValueError(f'图像路径须以来源和原图 ID 命名：{relative}')
    path = source_path(root, relative.as_posix())
    if 'sha256' in review and file_hash(path) != review['sha256']:
        raise ValueError(f'图像字节与审核时不同：{path}')
    counts = Counter(a['category_id'] for a in targets)
    if review['label'] != ('FAIL' if targets else 'PASS'):
        raise ValueError(f'图像结论与目标框不一致：{source}:{original_id}')
    return {
        'id': (1_000_000 if source == 'coxpwd' else 1_000_000_000) + original_id,
        'file_name': relative.as_posix(), 'width': width, 'height': height, 'split': 'train',
        'counts': {str(i): counts[i] for i in range(1, 6)}, 'label': review['label'],
        'source': source, 'source_image_id': original_id, 'source_path': review['original_path'],
        'group_id': f'supplemental:{source}:{group}', 'review_note': review['note'],
        **metadata,
    }


def _capture_key(name: str) -> tuple[str, int, int] | None:
    """从原始文件名 name 读取相机序列、序号和近邻跨度；无拍摄序号则返回 None。"""
    pixel = re.match(r'PXL_(\d{8})_(\d{2})(\d{2})(\d{2})', name)
    if pixel:
        day, hour, minute, second = pixel.groups()
        return f'PXL:{day}', int(hour) * 3600 + int(minute) * 60 + int(second), 12
    numbered = re.match(r'IMG_(\d+)\.([A-Za-z]+)', name)
    if numbered:
        number, suffix = numbered.groups()
        return f'IMG:{suffix}', int(number), 2
    return None


def _cox_holdouts(archive: ZipFile) -> tuple[dict, list[tuple[str, dict]]]:
    """从 archive 读取 valid/test 的拍摄序列与图像签名，不更改或交付留出图副本。"""
    captures = defaultdict(list)
    signatures = []
    with TemporaryDirectory(prefix='supplemental-holdouts-') as directory:
        scratch = Path(directory)
        path = scratch / 'image.jpg'
        for split in ('valid', 'test'):
            official = json.loads(archive.read(f'{split}/_annotations.coco.json'))
            by_image = defaultdict(set)
            names = {a['id']: a['name'] for a in official['categories']}
            for annotation in official['annotations']:
                by_image[annotation['image_id']].add(names[annotation['category_id']])
            for image in official['images']:
                member = f"{split}/{image['file_name']}"
                if key := _capture_key(image['extra']['name']):
                    series, serial, _ = key
                    captures[series].append((serial, by_image[image['id']], member))
                path.write_bytes(archive.read(member))
                meta, issues = inspect_image({**image, 'file_name': path.name}, scratch)
                if issues:
                    raise ValueError(f'CoxPWD 留出图尺寸异常：{member}')
                signatures.append((f'CoxPWD:{member}', meta))
    return captures, signatures


def _check_capture(record: dict, captures: dict) -> None:
    """拒绝 record 与 captures 中官方留出图的同类拍摄近邻，排除转角、变焦的连拍风险。"""
    key = _capture_key(record['original_name'])
    if key is None:
        return
    series, serial, gap = key
    categories = {ann['original_category'] for ann in record['source_annotations']}
    for other, tags, name in captures.get(series, ()):
        if abs(other - serial) <= gap and categories & tags:
            raise ValueError(f"CoxPWD 官方留出有同类拍摄近邻：{record['source_path']} ↔ {name}")


def _collect_cox(archive: ZipFile, reviews: list[dict], root: Path, restore: bool):
    """读官方 archive 的 train，将 reviews 中的图从 root 读取或按 restore 恢复。"""
    official = json.loads(archive.read('train/_annotations.coco.json'))
    names = {a['id']: a['name'] for a in official['categories']}
    if set(names.values()) != COXPWD_RULES.keys():
        raise ValueError('CoxPWD 官方类别与 v3 映射不一致')
    images = {r['id']: r for r in official['images']}
    by_image = defaultdict(dict)
    for index, ann in enumerate(official['annotations']):
        by_image[ann['image_id']][index] = {**ann, 'original_category': names[ann['category_id']]}
    for review in sorted(reviews, key=lambda item: item['original_id']):
        original = images[review['original_id']]
        original_path = 'train/' + original['file_name']
        if review['original_path'] != original_path:
            raise ValueError(f'审核记录与官方 train 路径不一致：{review}')
        path = source_path(root, review['file_name'])
        official_bytes = archive.read(original_path)
        if restore and not path.exists():
            _restore(path, official_bytes)
        if file_hash(path) != hashlib.sha256(official_bytes).hexdigest():
            raise ValueError(f'CoxPWD 本地图像与官方归档不一致：{path}')
        width, height = original['width'], original['height']
        originals = by_image[original['id']]
        targets = _cox_targets(review, originals, width, height)
        backgrounds = review['background']
        if len(set(backgrounds)) != len(backgrounds) or not set(backgrounds) <= HARD_BACKGROUNDS:
            raise ValueError(f'无效的易误报背景标记：{review}')
        yield _record('coxpwd', review, root, {
            **COXPWD_SOURCE, 'original_name': original['extra']['name'], 'background': backgrounds,
            'source_annotations': [{'source_index': idx, **ann} for idx, ann in originals.items()],
            'discarded': review.get('discarded', []),
        }, width, height, targets), targets


def _collect_wikimedia(reviews: list[dict], root: Path, restore: bool):
    """读取 Wikimedia reviews 和 root 中的原图；restore 启用时下载缺失的原图。"""
    for review in sorted(reviews, key=lambda item: item['original_id']):
        attribution = {key: review[key] for key in ('author', 'source_url', 'original_url', 'license', 'license_url')}
        if not all(v.strip() for v in attribution.values()) or not review['sha256']:
            raise ValueError(f'Wikimedia 署名、许可或图片摘要不完整：{review}')
        path = source_path(root, review['file_name'])
        if restore and not path.exists():
            req = Request(review.get('download_url', attribution['original_url']), headers={
                'User-Agent': 'RobotCleaningVision/1.0 (Wikimedia dataset; 2837798419@qq.com)'})
            with urlopen(req, timeout=60) as response:
                data = response.read()
            if hashlib.sha256(data).hexdigest() != review['sha256']:
                raise ValueError(f'下载图像与审核时的原件不同：{review["original_url"]}')
            _restore(path, data)
        if 'download_url' in review:
            attribution['download_url'] = review['download_url']
        with Image.open(path) as image:
            width, height = ImageOps.exif_transpose(image).size
        yield _record('wikimedia', review, root, attribution, width, height, []), []


def _check_duplicates(record: dict, signature: dict, references: list[tuple[str, dict]]) -> None:
    """将 record 的 signature 与 references 逐项比较，遇到精确或感知重复候选即中止。"""
    phash, dhash = (int(signature[k], 16) for k in ('phash', 'dhash'))
    for name, other in references:
        exact = any(signature[k] == other.get(k) for k in ('source_sha256', 'pixel_sha256'))
        near = (other.get('phash') and other.get('dhash')
                and (phash ^ int(other['phash'], 16)).bit_count() <= 10
                and (dhash ^ int(other['dhash'], 16)).bit_count() <= 16)
        if exact or near:
            raise ValueError(f"{'精确重复' if exact else '近重复候选'}：{record['file_name']} ↔ {name}")


def convert(root: Path, archive_path: Path, taco_annotations: Path, local_root: Path,
            output: Path, restore: bool) -> dict:
    """用 root 的图、archive_path 的官方标注和 output 同目录的审核 JSON 生成补充产物。

    taco_annotations 和 local_root 提供查重参考；restore 启用时仅恢复缺失图片。
    """
    if output.resolve() == taco_annotations.resolve():
        raise ValueError('补充标注不能写入原 TACO 标注路径')
    if file_hash(archive_path) != COXPWD_SHA256:
        raise ValueError(f'CoxPWD v3 归档校验失败：{archive_path}')
    taco = load_json(taco_annotations)
    taco_checks = taco['project']['image_checks']
    if any(not all(check.get(k) for k in ('source_sha256', 'pixel_sha256', 'phash', 'dhash'))
           for check in taco_checks.values()):
        raise ValueError('TACO 的重复排查签名不完整')
    references = [(f'TACO:{k}', v) for k, v in taco_checks.items()]
    for path in sorted(local_root.rglob('*')):
        if path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}:
            rel = path.relative_to(local_root).as_posix()
            meta, _ = inspect_image({'file_name': rel, 'width': 0, 'height': 0}, local_root)
            references.append((f'local:{rel}', meta))
    used_ids = {r['id'] for r in taco['images']}
    used_annotation_ids = {r['id'] for r in taco['annotations']}
    used_groups = {r['group_id'] for r in taco['images']}
    used_paths = {r['file_name'].casefold() for r in taco['images']}
    images, annotations, signatures = [], [], {}
    cox_reviews = load_json(output.parent / 'coxpwd_reviews.json')
    wiki_reviews = load_json(output.parent / 'wikimedia_reviews.json')
    reference_count = len(references)
    with ZipFile(archive_path) as archive:
        captures, heldout_signatures = _cox_holdouts(archive)
        references.extend(heldout_signatures)
        for collection in (_collect_cox(archive, cox_reviews, root, restore),
                           _collect_wikimedia(wiki_reviews, root, restore)):
            for record, targets in collection:
                image_id = record['id']
                relative = record['file_name'].casefold()
                if image_id in used_ids or relative in used_paths or record['group_id'] in used_groups:
                    raise ValueError(f'ID、路径或 TACO 组名冲突：{relative}')
                used_ids.add(image_id)
                used_paths.add(relative)
                if record['source'] == 'coxpwd':
                    _check_capture(record, captures)
                meta, issues = inspect_image(record, root)
                if issues or (record['source'] == 'coxpwd' and meta['exif_orientation'] != 1):
                    raise ValueError(f'图像方向或尺寸与显示坐标不匹配：{relative}，{issues}')
                _check_duplicates(record, meta, references)
                references.append((relative, meta))
                signatures[str(image_id)] = {key: meta[key] for key in (
                    'source_sha256', 'pixel_sha256', 'phash', 'dhash', 'exif_orientation', 'display_size')}
                images.append(record)
                for n, target in enumerate(targets):
                    if n >= 100:
                        raise ValueError(f'一图超过 100 个目标：{relative}')
                    ann_id = image_id * 100 + n
                    if ann_id in used_annotation_ids:
                        raise ValueError(f'标注 ID 冲突：{ann_id}')
                    annotations.append({**target, 'id': ann_id, 'image_id': image_id})
    statistics = {}
    for source in ('coxpwd', 'wikimedia'):
        rows = [r for r in images if r['source'] == source]
        statistics[source] = {
            'images': len(rows), 'groups': len({r['group_id'] for r in rows}),
            'labels': {k: sum(r['label'] == k for r in rows) for k in ('PASS', 'FAIL')},
            'background_images': {k: sum(k in r.get('background', []) for r in rows)
                                  for k in sorted(HARD_BACKGROUNDS)},
        }
    result = {
        'images': images, 'annotations': annotations, 'categories': PROJECT_CATEGORIES,
        'project': {
            'mapping': COXPWD_RULES, 'statistics': statistics, 'image_checks': signatures,
            'duplicate_check': {
                'taco_images': len(taco_checks), 'local_images': reference_count - len(taco_checks),
                'coxpwd_holdout_images': len(heldout_signatures), 'phash_max_distance': 10,
                'dhash_max_distance': 16, 'same_category_camera_gap': {'Pixel_seconds': 12, 'IMG_index': 2},
            },
        },
    }
    write_json(output, result)
    return result
