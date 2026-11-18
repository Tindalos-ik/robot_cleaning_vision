"""唯一真值为官方 COCO；只生成一个可追溯项目 JSON。"""
from __future__ import annotations
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from .annotations import convert_bbox
from .images import file_hash, inspect_image, source_path
from .mapping import PROJECT_CATEGORIES, build_mapping
from .splitting import build_groups, split_dataset

VERSION = 'official-project-2.0'


def validate(data: dict[str, Any]) -> None:
    """参数为项目 COCO；验证引用、像素 xywh、身份、计数标签及分区组隔离；成功返回 None，错误抛 ValueError。"""
    images = {im['id']: im for im in data['images']}
    if len(images) != len(data['images']):
        raise ValueError('重复图像主键')
    counts = {i: {str(c):0 for c in range(1,6)} for i in images}
    identities: set[int] = set()
    groups: dict[str,str] = {}
    for ann in data['annotations']:
        if ann['id'] in identities or ann['id'] != ann['source_index']+1:
            raise ValueError('项目实例身份不唯一')
        identities.add(ann['id'])
        im = images[ann['image_id']]
        convert_bbox(ann['bbox'],im['width'],im['height'])
        if ann['iscrowd'] or ann.get('ignore',0):
            raise ValueError('不可计数目标')
        counts[im['id']][str(ann['category_id'])] += 1
    for im in images.values():
        if im['counts'] != counts[im['id']] or im['label'] != ('FAIL' if sum(im['counts'].values()) else 'PASS'):
            raise ValueError('标签计数矛盾')
        if im['split'] not in ('train','val','test') or groups.setdefault(im['group_id'],im['split']) != im['split']:
            raise ValueError('分区组交叉')


def run_pipeline(input_dir: Path, output: Path, seed: int) -> dict[str, Any]:
    """参数为原图根目录、输出 JSON 和种子；返回完整 COCO。框保留显示方向像素 xywh；无修改/裁剪。I/O 抛 OSError，结构/引用错误抛 ValueError。"""
    input_dir = Path(input_dir); output = Path(output)
    original = input_dir / 'annotations.json'
    if output.resolve() == original.resolve() or output.suffix.lower() != '.json':
        raise ValueError('输出必须是非官方原文件的 JSON 路径')
    source_hash = file_hash(original)
    raw = json.loads(original.read_text(encoding='utf-8'))
    for key in ('images','categories'):
        ids = [r['id'] for r in raw[key]]
        if any(type(i) is not int for i in ids) or len(ids) != len(set(ids)):
            raise ValueError(f'{key} 主键必须唯一整数')
    mapping = build_mapping(raw['categories'])
    by_category = {m['original_category_id']:m for m in mapping}
    by_image = {im['id']:im for im in raw['images']}
    metadata: dict[int,dict[str,Any]] = {}
    reasons: dict[int,list[dict[str,Any]]] = defaultdict(list)
    for im in raw['images']:
        try:
            metadata[im['id']], issues = inspect_image(im,input_dir)
            reasons[im['id']].extend({'reason':r} for r in issues)
        except (OSError,ValueError) as exc:
            metadata[im['id']] = {}
            reasons[im['id']].append({'reason':'image_read_error','detail':str(exc)})
    targets: dict[int,list[dict[str,Any]]] = defaultdict(list)
    orphan: list[dict[str,Any]] = []
    for index, ann in enumerate(raw['annotations']):
        image_id = ann.get('image_id'); im = by_image.get(image_id)
        evidence = {'source_index':index,'object_id':ann.get('id'),'source_path':'annotations.json', 'reason':''}
        if im is None:
            orphan.append({**evidence,'reason':'unknown_image_reference','image_id':image_id})
            continue
        failures: list[str] = []
        category = by_category.get(ann.get('category_id'))
        if category is None:
            failures.append('unknown_category_reference')
        elif category['status'] == 'review':
            failures.append('ambiguous_category')
        if type(ann.get('id')) is not int:
            failures.append('invalid_source_object_id')
        try:
            convert_bbox(ann.get('bbox'),im['width'],im['height'])
        except ValueError as exc:
            failures.append(f'invalid_bbox: {exc}')
        if category and category['status'] != 'background' and (ann.get('iscrowd',0) or ann.get('ignore',0)):
            failures.append('target_count_unknown')
        reasons[image_id].extend({**evidence,'reason':r,'original_category_id':ann.get('category_id')} for r in failures)
        if not failures and category and category['status'] == 'target':
            targets[image_id].append({**ann,'id':index+1,'annotation_id':index+1,'object_id':ann['id'],
                'source_index':index,'source_path':'annotations.json','source_image_id':image_id,
                'source_file_name':im['file_name'],'original_category_id':ann['category_id'],
                'category_id':category['project_category_id'],'iscrowd':0})
    # 孤立标注无法归属任何图像，不能声称完整数据可用。
    if orphan:
        raise ValueError(f'官方标注有 {len(orphan)} 条未知图像引用: {orphan}')
    grouping = build_groups(raw['images'],metadata)
    images: list[dict[str,Any]] = []; held: list[dict[str,Any]] = []
    for im in raw['images']:
        group_id = grouping['assignments'][im['id']]
        if reasons[im['id']]:
            held.append({'id':im['id'],'file_name':im['file_name'],'group_id':group_id,'reasons':reasons[im['id']]})
            continue
        counts = {str(c):sum(a['category_id']==c for a in targets[im['id']]) for c in range(1,6)}
        images.append({**im,'source_image_id':im['id'],'source_path':'annotations.json','counts':counts,
                       'label':'FAIL' if sum(counts.values()) else 'PASS','group_id':group_id})
    partitions = split_dataset(images,seed)
    split_by_id = {i:s for s,ids in partitions.items() for i in ids}
    for im in images:
        im['split'] = split_by_id[im['id']]
    annotations = [a for im in images for a in targets[im['id']]]
    annotations.sort(key=lambda a:a['id'])
    duplicates = {str(k):v for k,v in Counter(a.get('id') for a in raw['annotations']).items() if v>1}
    stats = {'source_images':len(raw['images']),'source_annotations':len(raw['annotations']),
             'formal_images':len(images),'held_out_images':len(held),'target_instances':len(annotations),
             'class_instances':dict(sorted(Counter(str(a['category_id']) for a in annotations).items())),
             'labels':dict(Counter(im['label'] for im in images)),
             'splits':{s:{'images':len(ids),'labels':dict(Counter(im['label'] for im in images if im['split']==s)),
                           'class_instances':dict(Counter(str(a['category_id']) for a in annotations if split_by_id[a['image_id']]==s)),
                           'groups':len({im['group_id'] for im in images if im['split']==s})} for s,ids in partitions.items()},
             'held_out_reasons':dict(Counter(r for im in held for r in {e['reason'].split(':')[0] for e in im['reasons']})),
             'duplicate_object_ids':duplicates,'exif_transformed':sum(m.get('exif_orientation') in range(2,9) for m in metadata.values()),
             'nonstandard_exif':sum(m.get('exif_orientation',1) not in range(1,9) for m in metadata.values())}
    data = {'info':raw.get('info',{}),'licenses':raw.get('licenses',[]),'categories':PROJECT_CATEGORIES,
            'images':images,'annotations':annotations,
            'project':{'conversion_version':VERSION,'seed':seed,'source_sha256':source_hash,
                'annotation_source':'https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json',
                'official_repository':'https://github.com/pedropro/TACO',
                'license_note':'TACO 标注 CC BY 4.0；图像许可及署名以各官方图像记录和来源为准，官方 licenses 数组可能为空，不补造许可。',
                'label_policy':'PASS 仅表示官方标注经项目映射后无目标，未进行人工或 AI 全图核验；FAIL 表示至少一个可计数目标。待定或无效标注所在图留出。',
                'coordinate_policy':'EXIF 仅转图像，bbox 为原始显示方向 xywh，segmentation 不旋转、不修正。',
                'identity_policy':'annotation_id=id=原标注数组0基位置+1；object_id保留官方id，重复id不合并实例。',
                'mapping':mapping,'held_out':held,'grouping':{k:v for k,v in grouping.items() if k!='assignments'},
                'image_checks':{str(k):v for k,v in metadata.items()},'statistics':stats,
                'ai_review_used':False,'augmentation':'仅在固定划分后由模型训练端进行'}}
    validate(data)
    if file_hash(original) != source_hash:
        raise ValueError('官方标注发生变化')
    for im in raw['images']:
        expected = metadata[im['id']].get('source_sha256')
        if expected and file_hash(source_path(input_dir,im['file_name'])) != expected:
            raise ValueError('原图发生变化')
    data['project']['verification'] = {'references_counts_labels_groups':True,'raw_hashes_unchanged':True}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    return data
