"""依据真实元数据保守归并场景，并按完整场景组生成固定训练、验证、测试集。"""

from __future__ import annotations

import json
import math
import random
from datetime import datetime
from typing import Any


class _UnionFind:
    """维护图像编号的传递等价关系，不允许拆分已建立的自动关联。"""

    def __init__(self, image_ids: list[int]) -> None:
        """初始化；image_ids 为无重复整数图像编号列表；返回 None；不主动抛出异常。"""
        self.parents = {image_id: image_id for image_id in image_ids}

    def find(self, image_id: int) -> int:
        """查找整数图像编号的代表元并压缩路径；返回整数编号；未知编号抛出 KeyError。"""
        root = image_id
        while self.parents[root] != root:
            root = self.parents[root]
        while self.parents[image_id] != image_id:
            parent = self.parents[image_id]
            self.parents[image_id] = root
            image_id = parent
        return root

    def union(self, first: int, second: int) -> None:
        """合并两个整数图像编号，保留最小编号为根；返回 None；未知编号抛出 KeyError。"""
        first_root = self.find(first)
        second_root = self.find(second)
        self.parents[max(first_root, second_root)] = min(first_root, second_root)


def build_groups(
    images: list[dict[str, Any]],
    metadata: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    """仅对传入正式图片建立保守场景分组，返回分配、成员证据与策略。

    images 使用整数 id；metadata 按 id 提供哈希、URL 列表、相机、EXIF 日期、
    经纬度、64 位十六进制感知哈希及 HSV 直方图。缺失元数据不排除图片。
    所有符合条件的关系均作传递合并；输入顺序不影响组名、成员或证据顺序。
    不修改输入；重复编号、非法日期、坐标、哈希或直方图抛出 ValueError。
    """
    image_ids: list[int] = []
    seen_ids: set[int] = set()
    for image in images:
        image_id = image["id"]
        if type(image_id) is not int or image_id in seen_ids:
            raise ValueError("图像 id 必须为无重复整数")
        seen_ids.add(image_id)
        image_ids.append(image_id)
    image_ids.sort()
    union = _UnionFind(image_ids)
    evidence: list[dict[str, Any]] = []
    first_by_key: dict[tuple[str, str], int] = {}
    positions: list[tuple[int, tuple[float, float, float]]] = []
    visual: list[tuple[int, int, int | None, list[float] | None]] = []
    missing: dict[int, list[str]] = {}

    def associate(kind: str, value: str, image_id: int) -> None:
        """合并共享非空元数据值的图片并记录一条确定性关系证据。"""
        key = (kind, value)
        previous = first_by_key.get(key)
        if previous is None:
            first_by_key[key] = image_id
        elif previous != image_id:
            union.union(previous, image_id)
            evidence.append({"kind": kind, "images": [previous, image_id], "value": value})

    def parse_hash(value: Any, image_id: int, field: str) -> int | None:
        """将可缺失的 64 位十六进制哈希解析成整数，非法值抛出 ValueError。"""
        if value is None or value == "":
            return None
        if (not isinstance(value, str) or len(value) != 16
                or any(character not in "0123456789abcdefABCDEF" for character in value)):
            raise ValueError(f"图像 {image_id} 的 {field} 必须为 64 位十六进制字符串")
        return int(value, 16)

    for image_id in image_ids:
        meta = metadata.get(image_id, {})
        absent: list[str] = []
        for kind in ("source_sha256", "pixel_sha256"):
            value = meta.get(kind)
            if value:
                associate(kind, str(value).lower(), image_id)
            else:
                absent.append(kind)
        urls = meta.get("source_urls") or []
        if not isinstance(urls, list) or any(not isinstance(url, str) for url in urls):
            raise ValueError(f"图像 {image_id} 的 source_urls 必须为字符串列表")
        for url in sorted(set(url for url in urls if url)):
            associate("source_url", url, image_id)
        if not any(urls):
            absent.append("source_urls")
        camera = str(meta.get("camera") or "").strip()
        captured_at = str(meta.get("captured_at") or "").strip()
        if captured_at:
            try:
                captured = datetime.strptime(captured_at, "%Y:%m:%d %H:%M:%S")
            except ValueError as exc:
                raise ValueError(f"图像 {image_id} 的 EXIF 日期无效：{captured_at}") from exc
            if camera:
                day = captured.strftime("%Y-%m-%d")
                associate("camera_day", json.dumps([camera, day], ensure_ascii=False), image_id)
        if not camera or not captured_at:
            absent.append("camera_day")
        latitude, longitude = meta.get("latitude"), meta.get("longitude")
        if latitude is not None and longitude is not None:
            lat, lon = float(latitude), float(longitude)
            if not (math.isfinite(lat) and math.isfinite(lon)
                    and -90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError(f"图像 {image_id} 的 GPS 坐标无效")
            lat_r, lon_r = math.radians(lat), math.radians(lon)
            cos_lat = math.cos(lat_r)
            positions.append((image_id, (cos_lat * math.cos(lon_r),
                                         cos_lat * math.sin(lon_r), math.sin(lat_r))))
        else:
            absent.append("gps")
        phash = parse_hash(meta.get("phash"), image_id, "phash")
        dhash = parse_hash(meta.get("dhash"), image_id, "dhash")
        histogram = meta.get("hsv_hist")
        normalized_histogram: list[float] | None = None
        if histogram is not None and histogram != []:
            if not isinstance(histogram, list):
                raise ValueError(f"图像 {image_id} 的 hsv_hist 必须为浮点列表")
            values = [float(value) for value in histogram]
            if any(not math.isfinite(value) or value < 0 for value in values):
                raise ValueError(f"图像 {image_id} 的 hsv_hist 必须为有限非负数")
            norm = math.hypot(*values)
            if norm:
                normalized_histogram = [value / norm for value in values]
        for field, value in (("phash", phash), ("dhash", dhash),
                             ("hsv_hist", normalized_histogram)):
            if value is None:
                absent.append(field)
        missing[image_id] = absent
        if phash is not None:
            visual.append((image_id, phash, dhash, normalized_histogram))

    earth_radius_m = 6371008.8
    chord_limit_squared = (2 * math.sin(100.0 / (2 * earth_radius_m))) ** 2
    for index, (first_id, first_position) in enumerate(positions):
        for second_index in range(index + 1, len(positions)):
            second_id, second_position = positions[second_index]
            chord_squared = sum((a - b) ** 2 for a, b in zip(first_position, second_position))
            if chord_squared <= chord_limit_squared:
                union.union(first_id, second_id)
                distance_m = 2 * earth_radius_m * math.asin(min(1.0, math.sqrt(chord_squared) / 2))
                evidence.append({"kind": "gps_distance", "images": [first_id, second_id],
                                 "distance_m": distance_m})

    for index, (first_id, first_phash, first_dhash, first_histogram) in enumerate(visual):
        for second_index in range(index + 1, len(visual)):
            second_id, second_phash, second_dhash, second_histogram = visual[second_index]
            phash_distance = (first_phash ^ second_phash).bit_count()
            if phash_distance > 18:
                continue
            if phash_distance <= 10 and first_dhash is not None and second_dhash is not None:
                dhash_distance = (first_dhash ^ second_dhash).bit_count()
                if dhash_distance <= 16:
                    union.union(first_id, second_id)
                    evidence.append({"kind": "near_duplicate_candidate",
                                     "images": [first_id, second_id],
                                     "phash_distance": phash_distance,
                                     "dhash_distance": dhash_distance})
            if (first_histogram is not None and second_histogram is not None
                    and len(first_histogram) == len(second_histogram)):
                similarity = min(1.0, sum(a * b for a, b in zip(first_histogram, second_histogram)))
                if similarity >= 0.97:
                    union.union(first_id, second_id)
                    evidence.append({"kind": "related_scene_candidate",
                                     "images": [first_id, second_id],
                                     "phash_distance": phash_distance,
                                     "hsv_cosine_similarity": similarity})

    members: dict[int, list[int]] = {}
    for image_id in image_ids:
        members.setdefault(union.find(image_id), []).append(image_id)
    group_evidence: dict[int, list[dict[str, Any]]] = {root: [] for root in members}
    for item in evidence:
        group_evidence[union.find(item["images"][0])].append(item)
    limitations = [
        "候选关系只表示潜在重复或相关场景，不证明图片相同；为防止泄漏一律保守合并。",
        "传递合并可能过度归组，组内任意两张图片不一定满足阈值，GPS 总直径可超过 100 米。",
        "相机同日可能覆盖不同场景，跨日期 GPS 邻近也可能不是同一拍摄事件。",
        "缺少元数据或视觉相似度未达阈值不证明场景独立；可能仍有未检出的跨组关联。",
        "直方图缺失、全零或维度不同不产生场景候选；感知哈希不能保证识别视角或内容大幅变化。",
    ]
    assignments: dict[int, str] = {}
    groups: list[dict[str, Any]] = []
    for root, member_ids in members.items():
        group_id = f"group_{root}"
        for image_id in member_ids:
            assignments[image_id] = group_id
        groups.append({"group_id": group_id, "members": member_ids,
                       "evidence": group_evidence[root],
                       "missing_metadata": {str(image_id): missing[image_id]
                                            for image_id in member_ids if missing[image_id]},
                       "limitations": list(limitations)})
    return {
        "assignments": assignments,
        "groups": groups,
        "policy": {
            "exact_match": ["source_urls", "source_sha256", "pixel_sha256"],
            "camera_same_day_conservative_merge": True,
            "gps_threshold_m": 100,
            "gps_cross_date": True,
            "near_duplicate": {"phash_distance_max": 10, "dhash_distance_max": 16},
            "related_scene": {"phash_distance_max": 18, "hsv_cosine_min": 0.97},
            "candidate_search": "所有具备相应元数据的正式图片对，按图像 id 升序穷举",
            "all_candidates_conservatively_merged": True,
            "transitive_union": True,
            "missing_metadata_excludes_image": False,
            "group_id": "group_加组内最小图像编号",
            "limitations": limitations,
        },
    }


def split_dataset(images: list[dict[str, Any]], seed: int) -> dict[str, list[int]]:
    """按场景整组生成约 70/15/15 的固定训练、验证、测试分区，不执行 K 折。

    参数：images 为正式 COCO 图像列表，每行 id 为整数、group_id 为非空字符串、
    label 为 PASS/FAIL，counts 含字符串键 1..5 的非负整数实例数。
    仅对传入图片分组，所有图片均进入正式分区，不依据额外资格字段筛选。
    seed 为确定性随机种子。返回 train、val、test 对应的升序图像编号列表。
    目标同时平衡图像量、五类 presence 图像数和实例数、PASS/FAIL 图像数；
    使用带种子次序的贪心平方误差、整组移动和交换优化。组数或类别分布受限时
    不保证比例或五类覆盖，绝不拆组凑数；同输入与种子可复现，亦不依赖输入顺序。
    不修改输入。重复/非整数编号、空组名、无效标签或计数抛出 ValueError；
    缺失必要字段可抛出 KeyError，不符合字典结构的输入可抛出 TypeError。
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    seen_ids: set[int] = set()
    for image in images:
        image_id = image["id"]
        group_id = image["group_id"]
        if type(image_id) is not int or image_id in seen_ids:
            raise ValueError("分区图像编号必须为无重复整数")
        if not isinstance(group_id, str) or not group_id.strip():
            raise ValueError(f"图像 {image_id} 缺少有效 group_id")
        if image["label"] not in ("PASS", "FAIL"):
            raise ValueError(f"图像 {image_id} 的 label 必须为 PASS 或 FAIL")
        counts = image["counts"]
        for category in range(1, 6):
            count = counts[str(category)]
            if type(count) is not int or count < 0:
                raise ValueError(f"图像 {image_id} 的 counts[{category!s}] 必须为非负整数")
        seen_ids.add(image_id)
        grouped.setdefault(group_id, []).append(image)
    vectors: list[list[float]] = []
    member_ids: list[list[int]] = []
    for group_id in sorted(grouped):
        rows = grouped[group_id]
        vector = [float(len(rows))] + [0.0] * 12
        for row in rows:
            for category in range(1, 6):
                count = row["counts"][str(category)]
                vector[category] += int(count > 0)
                vector[category + 5] += count
            vector[11] += int(row["label"] == "PASS")
            vector[12] += int(row["label"] == "FAIL")
        vectors.append(vector)
        member_ids.append(sorted(row["id"] for row in rows))
    names = ("train", "val", "test")
    result: dict[str, list[int]] = {name: [] for name in names}
    if not vectors:
        return result
    totals = [sum(vector[column] for vector in vectors) for column in range(13)]
    # 先按全量归一化，避免实例较多的类别淹没罕见类；图像量的权重为四。
    weights = [4.0] + [1.0] * 12
    normalized = [[value / total if total else 0.0 for value, total in zip(vector, totals)]
                  for vector in vectors]
    ratios = (0.70, 0.15, 0.15)
    # 至少出现于三个独立组的类别/标签具备覆盖三个分区的基础；
    # 缺类惩罚优先保护覆盖，再用平方误差贴近样本比例。
    coverage_columns = [column for column in (*range(1, 6), 11, 12)
                        if sum(vector[column] > 0 for vector in vectors) >= 3]
    loads = [[0.0] * 13 for _ in names]
    assignments = [-1] * len(vectors)
    rng = random.Random(seed)
    order = list(range(len(vectors)))
    rng.shuffle(order)
    order.sort(key=lambda index: sum(weight * value * value
                                    for weight, value in zip(weights, normalized[index])), reverse=True)

    def change_cost(partition: int, vector: list[float], direction: int) -> float:
        """计算某分区增减一组的目标差；参数为分区索引、无单位特征和 ±1；返回浮点差，无主动异常。"""
        balance_cost = sum(weight * (2 * (load - ratios[partition]) * direction * value + value * value)
                           for weight, load, value in zip(weights, loads[partition], vector))
        coverage_cost = sum(
            int(loads[partition][column] + direction * vector[column] <= 1e-12)
            - int(loads[partition][column] <= 1e-12) for column in coverage_columns
        )
        return balance_cost + 10.0 * coverage_cost

    def apply(partition: int, vector: list[float], direction: int) -> None:
        """就地增减分区负载；参数为整数分区索引、无单位特征和 ±1；返回 None，无主动异常。"""
        for column, value in enumerate(vector):
            loads[partition][column] += direction * value

    for index in order:
        vector = normalized[index]
        choices = list(range(3))
        rng.shuffle(choices)
        destination = min(choices, key=lambda partition: change_cost(partition, vector, 1))
        assignments[index] = destination
        apply(destination, vector, 1)

    # 限定交换候选量以控制大数据集开销；严格下降保证优化不会循环。
    tolerance = 1e-12
    for _ in range(8):
        improved = False
        rng.shuffle(order)
        for index in order:
            source = assignments[index]
            vector = normalized[index]
            removal_cost = change_cost(source, vector, -1)
            destination = min((partition for partition in range(3) if partition != source),
                              key=lambda partition: change_cost(partition, vector, 1))
            if removal_cost + change_cost(destination, vector, 1) < -tolerance:
                apply(source, vector, -1)
                apply(destination, vector, 1)
                assignments[index] = destination
                improved = True
        for offset, first in enumerate(order):
            source = assignments[first]
            for step in range(1, min(25, len(order))):
                second = order[(offset + step) % len(order)]
                destination = assignments[second]
                if source == destination:
                    continue
                difference = [b - a for a, b in zip(normalized[first], normalized[second])]
                delta = change_cost(source, difference, 1) + change_cost(destination, difference, -1)
                if delta < -tolerance:
                    apply(source, difference, 1)
                    apply(destination, difference, -1)
                    assignments[first], assignments[second] = destination, source
                    improved = True
                    break
        if not improved:
            break
    for index, partition in enumerate(assignments):
        result[names[partition]].extend(member_ids[index])
    for image_ids in result.values():
        image_ids.sort()
    return result
