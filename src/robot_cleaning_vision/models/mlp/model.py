"""保留模型的两级 MLP 推理。

第一级：每个自动窗口的 160 维 HSV 特征 → 32 个隐藏单元 → 1 个区域分数。
第二级：30 维区域分数统计 + 58 维整图特征 → 8 个隐藏单元 → 1 个整图分数。

权重文件保存两级标准化参数、窗口配置和阈值；baseline1 命令行采用固定部署阈值。
"""

from pathlib import Path

import cv2
import numpy as np
import torch
from torch import nn

from .features import extract_context, extract_regions, generate_windows, prepare_image, rgb_image


DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[4] / "artifacts" / "mlp" / "model.pt"
# 训练评估与 baseline1 图片、摄像头入口共用此阈值。
DEPLOYMENT_THRESHOLD = 0.5

# 与训练时一致：两个网络均为“全连接 → ReLU → 全连接”。
# Sigmoid 在实际预测时调用，不放进 Sequential，便于明确区分原始输出和分数。


def _network(input_dim: int, hidden: int) -> nn.Sequential:
    """构造两级网络共用的结构，input_dim 和 hidden 决定每层的宽度。

    区域网络传入 (160, 32)，整图网络传入 (88, 8)。
    Linear 执行输入与权重的加权求和并加上偏置；ReLU 将负数置零。
    最后一个 Linear 只输出一个实数，允许为负，也允许大于 1。
    """
    return nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))


def _support3(probabilities, boxes):
    """计算相邻窗口对高分区域的支持程度，返回一个统计值。

    先按分数选择最高的 30 个候选，分数相同时用坐标稳定排序。
    逐个以候选为参照，查找与它交并比不低于 0.2 的窗口；
    至少有 3 个时，取其中最高 3 个分数的平均值，最后保留最大平均值。
    这里使用的是自动窗口之间的重叠，不使用真实目标框。
    """
    top = np.lexsort((boxes[:, 3], boxes[:, 2], boxes[:, 1], boxes[:, 0],
                      probabilities))[-min(30, len(probabilities)):]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    best = 0.0
    for index in top:
        box = boxes[index]
        lo = np.maximum(box[:2], boxes[:, :2])
        hi = np.minimum(box[2:], boxes[:, 2:])
        intersection = np.maximum(hi - lo, 0).prod(axis=1)
        # 交集面积 / 并集面积就是交并比，用来判断窗口是否在观察相近区域。
        area = (box[2] - box[0]) * (box[3] - box[1])
        overlap = intersection / np.maximum(area + areas - intersection, 1)
        scores = probabilities[overlap >= 0.2]
        if len(scores) >= 3:
            best = max(best, float(np.partition(scores, -3)[-3:].mean()))
    return best


def score_summary(probabilities, boxes, frame_size):
    """把数量不固定的区域分数压缩成固定顺序的 30 维向量。

    组成：最高分均值 5 维、分位数 6 维、均值/标准差 2 维、
    高分占比 5 维、按窗口大小分组的统计 6 维、面积/邻域统计 4 维、
    整图窗口分数与候选数量 2 维。frame_size 的顺序为（宽，高）。
    这些是固定计算规则，没有额外需要训练的网络参数。
    """
    p = np.asarray(probabilities, dtype=np.float32)
    boxes = np.asarray(boxes)
    order = np.argsort(p)
    area = ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])).astype(np.float64)
    relative = area / (frame_size[0] * frame_size[1])
    # 窗口面积除以整图面积，便于比较不同尺寸图片上的窗口大小。
    values = [p[order[-min(k, len(p)):]].mean() for k in (1, 3, 5, 10, 20)]
    values.extend(np.quantile(p, [.25, .5, .75, .9, .95, .99]))
    values.extend((p.mean(), p.std()))
    values.extend(np.mean(p >= t) for t in (.5, .7, .9, .95, .99))
    # 小、中、大三组各取最大分数和最高三个分数的平均值，共 6 维。
    for lo, hi in ((0, .02), (.02, .1), (.1, 1.00001)):
        selected = p[(relative >= lo) & (relative < hi)]
        if len(selected):
            values.extend((selected.max(), np.sort(selected)[-min(3, len(selected)):].mean()))
        else:
            values.extend((0.0, 0.0))
    # 最高分并列时取平均面积；高分小窗口占比包含第 20 高分处的全部并列窗口。
    highest_area = relative[p == p.max()].mean()
    cutoff = p[order[-min(20, len(p))]]
    values.extend((np.average(p, weights=area), highest_area,
                   np.mean(relative[p >= cutoff] < .02), _support3(p, boxes)))
    # 整图窗口也有一个区域分数；log1p 对候选数量作对数压缩。
    whole = ((boxes[:, 0] == 0) & (boxes[:, 1] == 0)
             & (boxes[:, 2] == frame_size[0]) & (boxes[:, 3] == frame_size[1]))
    values.extend((p[np.flatnonzero(whole)[0]], np.log1p(len(p))))
    return np.asarray(values, dtype=np.float32)


class MLPModel:
    """持有区域网络、整图网络及各自的标准化参数，供多帧重复预测。

    load() 只需调用一次。之后 predict() 复用内存中的权重，
    返回整图 FAIL 分数；PASS/FAIL 判定由公共后处理使用 threshold 完成。
    """

    model_id = "baseline1"
    model_name = "MLP"
    model_path = DEFAULT_MODEL_PATH

    def __init__(self, checkpoint):
        """从同一个权重文件恢复两级模型，不在预测时重新估计任何参数。"""
        region = checkpoint["region_checkpoint"]
        # 区域网络：160 个颜色特征 → 32 个隐藏单元 → 1 个原始输出。
        self.region_network = _network(160, 32)
        self.region_network.load_state_dict(region["state_dict"])
        self.region_network.eval()
        # 整图网络：30 个区域统计 + 58 个背景特征 → 8 个隐藏单元 → 1 个原始输出。
        self.network = _network(88, 8)
        self.network.load_state_dict(checkpoint["state_dict"])
        self.network.eval()
        # eval() 切换推理模式；下列均值和缩放系数均由训练数据预先拟合。
        # 两级网络的输入含义不同，因此分别使用各自的标准化参数。
        self.region_mean = np.asarray(region["mean"], dtype=np.float32)
        self.region_scale = np.asarray(region["scale"], dtype=np.float32)
        self.mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        self.scale = np.asarray(checkpoint["scale"], dtype=np.float32)
        # 候选尺度、长宽比、步长和最终阈值必须与保存的模型保持一致。
        self.proposal = dict(region["proposal"])
        self.threshold = float(checkpoint["threshold"])

    @classmethod
    def load(cls, path=DEFAULT_MODEL_PATH):
        """在 CPU 上加载选定模型；版本检查避免把其他结构按当前特征解释。"""
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if (saved["kind"] != "regional-score-stack-v1"
                or saved["summary_version"] != "region-score-summary-30-v1"
                or not saved["include_context"] or saved["context_version"] != "local-384-v1"
                or saved["region_checkpoint"]["mode"] != "color"):
            raise ValueError("Expected the selected HSV160/score30/local58 checkpoint")
        model = cls(saved)
        model.model_path = Path(path).resolve()
        return model

    def predict(self, image) -> float:
        """从 PIL 图像或 RGB 数组得到一个整图 FAIL 分数。

        数据形状依次为：
        N 个窗口 → (N, 160) → (N, 32) → (N, 1) → N 个区域分数；
        分数汇总 (30,) + 整图特征 (58,) → (1, 88) → (1, 8) → (1, 1)。
        N 随图片尺寸和候选配置变化，模型两级输入维数保持固定。
        """
        image = rgb_image(image)
        # 1. 转正并统一为 RGB；区域分支把长边等比缩放到模型指定的 640。
        frame = prepare_image(image, self.proposal["long_edge"])
        boxes = generate_windows(frame.shape, self.proposal)
        regions = extract_regions(frame, boxes)
        # 2. 每个窗口缩放至 64×64，计算 HSV160，再使用区域训练统计量标准化。
        normalized = (regions - self.region_mean) / self.region_scale
        # 3. 区域网络的最后一层输出称为 logit；Sigmoid 将其映射到 0～1。
        # inference_mode() 关闭梯度记录，区域分支一次处理全部窗口。
        with torch.inference_mode():
            probabilities = self.region_network(torch.from_numpy(normalized)).sigmoid().numpy().ravel()
        # 4. 汇总全部区域分数。整图 58 维特征从原始转正图片独立缩放到 384 提取，
        #    不从上面的 640 图片继续缩放，以保持与训练时完全一致。
        summary = score_summary(probabilities, boxes, (frame.shape[1], frame.shape[0]))
        features = np.r_[summary, extract_context(image)]
        # 5. 对拼接后的 88 维输入使用整图网络自己的标准化参数。
        normalized = (features - self.mean) / self.scale
        # 6. 第二个网络输出整图 logit，再经 Sigmoid 返回一个 Python 浮点分数。
        #    此处不直接比较阈值，外层调用链会用 self.threshold 作统一后处理。
        with torch.inference_mode():
            return float(self.network(torch.from_numpy(normalized[None, :])).sigmoid().numpy().ravel()[0])

    def predict_bgr(self, frame: np.ndarray) -> float:
        """将 OpenCV 的 BGR 帧 frame 转为 RGB，再计算整图 FAIL 概率。"""
        return self.predict(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
