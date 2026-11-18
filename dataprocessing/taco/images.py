"""原图只读检查；官方坐标已对应显示方向，不旋转标注。"""
from __future__ import annotations
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from PIL import Image, ImageOps


def source_path(root: Path, file_name: str) -> Path:
    """参数为原图根目录和相对路径；返回安全绝对路径；越界抛 ValueError，无坐标变换。"""
    path = (root / file_name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"不安全路径: {file_name}")
    return path


def file_hash(path: Path) -> str:
    """参数为文件路径；返回 SHA256；读取失败抛 OSError，无坐标。"""
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def gps_coordinate(values: Any, reference: Any) -> float | None:
    """参数为 EXIF 度分秒和方位；返回十进制度或 None；无异常传播，无像素坐标。"""
    try:
        result = float(values[0]) + float(values[1])/60 + float(values[2])/3600
        return -result if reference in ('S', 'W') else result
    except (TypeError, IndexError, ValueError, ZeroDivisionError, OverflowError):
        return None


def inspect_image(record: dict[str, Any], root: Path) -> tuple[dict[str, Any], list[str]]:
    """参数为官方图像和根目录；返回元数据及错误；尺寸为 EXIF 显示像素，框不旋转；I/O 抛 OSError。"""
    path = source_path(root, record['file_name'])
    meta: dict[str, Any] = {'source_sha256': file_hash(path)}
    issues: list[str] = []
    with Image.open(path) as original:
        exif = original.getexif()
        sub = exif.get_ifd(34665) if 34665 in exif else {}
        gps = exif.get_ifd(34853) if 34853 in exif else {}
        rgb = ImageOps.exif_transpose(original).convert('RGB')
        if rgb.size != (record['width'], record['height']):
            issues.append('display_size_mismatch')
        orientation = exif.get(274, 1)
        # 非标准方向值由 Pillow 原样处理；尺寸匹配即可，不猜测旋转。
        meta.update(exif_orientation=orientation, display_size=list(rgb.size),
                    pixel_sha256=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest(),
                    camera=' '.join(str(exif.get(t, '')).strip() for t in (271,272)).strip(),
                    source_urls=[record[k] for k in ('flickr_url','coco_url') if record.get(k)])
        captured = str(sub.get(36867) or exif.get(306) or record.get('date_captured') or '')
        try:
            captured = datetime.strptime(captured, '%Y:%m:%d %H:%M:%S').strftime('%Y:%m:%d %H:%M:%S')
        except ValueError:
            captured = ''
        lat = gps_coordinate(gps.get(2),gps.get(1)); lon = gps_coordinate(gps.get(4),gps.get(3))
        meta.update(captured_at=captured, latitude=lat if lat is not None and -90<=lat<=90 else None,
                    longitude=lon if lon is not None and -180<=lon<=180 else None)
        gray = np.asarray(rgb.convert('L').resize((32,32), Image.Resampling.LANCZOS), dtype=np.float32)
        low = cv2.dct(gray)[:8,:8].flatten()
        bits = low > np.median(low[1:]); bits[0] = False
        meta['phash'] = f'{int.from_bytes(np.packbits(bits).tobytes()):016x}'
        small = np.asarray(rgb.convert('L').resize((9,8),Image.Resampling.LANCZOS))
        meta['dhash'] = f'{int.from_bytes(np.packbits(small[:,1:] > small[:,:-1]).tobytes()):016x}'
        hsv = cv2.cvtColor(np.asarray(rgb.resize((128,128))), cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv],[0,1],None,[16,4],[0,180,0,256]).flatten()
        norm = np.linalg.norm(hist)
        meta['hsv_hist'] = (hist / norm).astype(float).tolist()
    return meta, issues
