"""保留模型所需的 RGB 窗口、160 维 HSV 区域特征和 58 维整图特征。"""

import math

import cv2
import numpy as np
from PIL import Image, ImageOps # PIL图像是Python中表示、读取和处理图片的一种对象。 当前项目实际使用的是Pillow库


# 所有窗口都会缩放到64×64。这里固定五个统计区域及顺序：
# 整个窗口、左上、右上、左下、右下；训练和预测必须使用同一顺序。
_COLOR_REGIONS = (
    (slice(None), slice(None)),
    (slice(0, 32), slice(0, 32)),
    (slice(0, 32), slice(32, 64)),
    (slice(32, 64), slice(0, 32)),
    (slice(32, 64), slice(32, 64)),
)


def rgb_image(image: Image.Image | np.ndarray) -> np.ndarray:
    """
    PIL 图像按 EXIF 方向转正；数组须为 RGB uint8 HWC，原始输入保持不变。
    EXIF是照片附带的元数据
    """
    if isinstance(image, Image.Image):
        # 先按照片EXIF方向转正，再转RGB；摄像头数组应由调用方从BGR转成RGB。
        image = np.asarray(ImageOps.exif_transpose(image).convert("RGB"))
    if (not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3 or 0 in image.shape):
        raise ValueError("图像须为非空RGB uint8三通道数组或PIL图像")
    return image


def prepare_image(image: Image.Image | np.ndarray, long_edge: int = 640) -> np.ndarray:
    """使用面积插值按长边等比缩放；小图也会放大到指定尺寸。"""
    image = rgb_image(image)
    height, width = image.shape[:2]
    # 宽高使用同一比例，长边达到指定像素数；round与生成训练特征时保持一致。
    scale = int(long_edge) / max(height, width)
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    if size == (width, height):
        return image.copy()
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def generate_windows(shape, config) -> np.ndarray:
    """按固定顺序生成去重的 xyxy 窗口，右边界和下边界不包含在切片中。"""
    height, width = shape[:2]
    boxes = []
    seen = set()
    short_edge = min(height, width)
    # scale以短边为参照决定窗口面积；aspect=宽/高。
    # 宽乘sqrt(aspect)、高除sqrt(aspect)，改变形状而大致保持面积不变。
    for scale in config["scales"]:
        side = float(scale) * short_edge
        for aspect in config["aspects"]:
            root_aspect = math.sqrt(float(aspect))
            window_width = max(1, round(min(width, side * root_aspect)))
            window_height = max(1, round(min(height, side / root_aspect)))
            last_x, last_y = width - window_width, height - window_height
            # stride=0.8表示每次移动窗口宽/高的80%，相邻窗口有部分重叠。
            step_x = max(1, round(min(width, window_width * float(config["stride"]))))
            step_y = max(1, round(min(height, window_height * float(config["stride"]))))
            xs = list(range(0, last_x + 1, step_x))
            ys = list(range(0, last_y + 1, step_y))
            # 常规步长未必刚好落到边缘，额外补齐最右和最下的位置。
            if xs[-1] != last_x:
                xs.append(last_x)
            if ys[-1] != last_y:
                ys.append(last_y)
            for y in ys:
                for x in xs:
                    box = (x, y, x + window_width, y + window_height)
                    if box not in seen:
                        seen.add(box)
                        boxes.append(box)
    # 整图也作为一个候选，为后续汇总提供全局区域分数。
    whole = (0, 0, width, height)
    if whole not in seen:
        boxes.append(whole)
    return np.asarray(boxes, dtype=np.int32).reshape(-1, 4)


def extract_regions(frame: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """把N个窗口变成(N, 160)矩阵，每行是一个窗口的颜色比例。

    每个窗口分成整块和四象限，共5个区域；每区H16、S8、V8，合计32维。
    输出顺序固定为5×32，数值是像素比例，不是网络预测概率。
    """
    result = np.empty((len(boxes), 160), dtype=np.float32)
    for index, (left, top, right, bottom) in enumerate(boxes):
        patch = cv2.resize(frame[top:bottom, left:right], (64, 64),
                           interpolation=cv2.INTER_AREA)
        # OpenCV的uint8 HSV范围：H为0～179，S和V为0～255。
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        offset = 0
        for region in _COLOR_REGIONS:
            pixels = hsv[region]
            for channel, bins, upper in ((0, 16, 180), (1, 8, 256), (2, 8, 256)):
                histogram = cv2.calcHist([pixels], [channel], None, [bins], [0, upper]).ravel()
                # 用区域面积归一化计数，每条一维直方图之和为1。
                # 保持两次除法的顺序，与已保存权重所用的特征数值一致。
                result[index, offset:offset + bins] = histogram / pixels.shape[0] / pixels.shape[1]
                offset += bins
    return result


def _grid_slices(height: int, width: int):
    """按行优先产生4×4分区切片；极窄图片可重复像素，保持每块非空。"""
    for row in range(4):
        top = row * height // 4
        bottom = max(top + 1, (row + 1) * height // 4)
        for column in range(4):
            left = column * width // 4
            right = max(left + 1, (column + 1) * width // 4)
            yield slice(top, bottom), slice(left, right)


def extract_context(image: Image.Image | np.ndarray) -> np.ndarray:
    """把整图变成58维上下文，描述背景及局部差异。

    16个块各有11项统计；对每项作5种无序汇总，得到55维。
    再加入3项局部Lab颜色差异，共58维，不保留块的具体位置。
    """
    image = rgb_image(image)
    height, width = image.shape[:2]
    # 此分支独立使用长边384的图片，不能接着区域分支的640图再缩放。
    scale = 384 / max(height, width)
    resized = cv2.resize(image, (max(1, round(width * scale)), max(1, round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(resized, cv2.COLOR_RGB2HSV)
    # 浮点Lab的L表示明暗，a/b表示两组对立颜色；缩放后便于计算颜色距离。
    lab = cv2.cvtColor(resized.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    lab /= np.array([100, 128, 128], dtype=np.float32)
    # 灰度先归一化到0～1。Sobel是固定的亮度差分算子，不是可训练卷积层。
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255
    dx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3, scale=0.125,
                   borderType=cv2.BORDER_REFLECT_101)
    dy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3, scale=0.125,
                   borderType=cv2.BORDER_REFLECT_101)
    # 水平、垂直差分合成梯度幅度；Laplacian绝对值描述更细碎的亮度突变。
    magnitude = np.hypot(dx, dy)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3, scale=0.125,
                                    borderType=cv2.BORDER_REFLECT_101))
    # 颜色均值/标准差也使用统一量级；每块保留颜色、对比度、边缘和像素占比。
    hsv_unit = hsv.astype(np.float32) / np.array([180, 255, 255], dtype=np.float32)
    blocks = []
    distances = []
    lab_mean = lab.mean(axis=(0, 1))
    for region in _grid_slices(*gray.shape):
        pixels = hsv_unit[region].reshape(-1, 3)
        saturation, value = pixels[:, 1], pixels[:, 2]
        # 11项依次为：HSV均值3、标准差3、灰度标准差1、梯度均值1、
        # 亮度突变均值1、鲜艳像素比例1、明亮且低饱和度像素比例1。
        blocks.append(np.concatenate((
            pixels.mean(axis=0), pixels.std(axis=0),
            [gray[region].std(), magnitude[region].mean(), laplacian[region].mean(),
             (saturation > 0.5).mean(), ((value > 0.8) & (saturation < 0.2)).mean()],
        )))
        # 与整图平均颜色差多少：用来衡量某些局部区域是否特别显眼。
        distances.append(np.linalg.norm(lab[region].mean(axis=(0, 1)) - lab_mean))
    blocks = np.asarray(blocks)
    # 11项分别取均值、标准差、10%分位数、90%分位数和最大值，共55维。
    # Lab距离再取均值、90%分位数和最大值，共3维；拼接结果按固定顺序输出。
    return np.concatenate((
        blocks.mean(axis=0), blocks.std(axis=0), np.quantile(blocks, 0.1, axis=0),
        np.quantile(blocks, 0.9, axis=0), blocks.max(axis=0),
        [np.mean(distances), np.quantile(distances, 0.9), np.max(distances)],
    )).astype(np.float32)
