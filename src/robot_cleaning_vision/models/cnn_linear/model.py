"""冻结 ResNet-50 特征、逻辑回归参数及离线模型加载。"""

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

from ...capture.camera import BGRFrame
from ...preprocessing import preprocess


ARTIFACT_ROOT = Path(__file__).resolve().parents[4] / "artifacts/cnn_linear"
DEFAULT_MODEL_PATH = ARTIFACT_ROOT / "model.pt"
CHECKPOINT_KIND = "resnet50-logistic-regression-v1"
WEIGHTS = ResNet50_Weights.IMAGENET1K_V2
LABELS = {"PASS": 0, "FAIL": 1}
SUPPORTED_INPUT_SIZES = (384, 512)
_PREPROCESSING = {
    "resize": "bilinear_center_letterbox",
    "fill": (124, 116, 104),
    "color": "RGB",
    "dtype": "float32",
    "scale": 1 / 255,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}


def compute_device() -> torch.device:
    """返回当前可用的 CUDA 设备，无 CUDA 时使用 CPU。"""
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class FeatureExtractor:
    """只读取全局平均池化后的 2048 维特征，固定参数和 BatchNorm 状态。"""

    def __init__(self, backbone: nn.Module, device: torch.device, *, input_size: int) -> None:
        """冻结 backbone，在 device 上使用 input_size×input_size 预处理提取整图特征。"""
        if input_size not in SUPPORTED_INPUT_SIZES:
            raise ValueError(f"不支持的 baseline2 输入尺寸：{input_size}")
        self.preprocessing = {**_PREPROCESSING, "target_size": (input_size, input_size)}
        if device.type == "cuda":
            # 统一单图与批量卷积的计算精度，避免 TF32 的分数偏差。
            torch.backends.cudnn.conv.fp32_precision = "ieee"
        self.backbone = backbone.requires_grad_(False).eval().to(device)
        self.device = device

    @classmethod
    def pretrained(cls, device: torch.device, *, input_size: int) -> "FeatureExtractor":
        """在 device 上载入官方 V2 骨干，移除原分类层并使用 input_size 方形输入。"""
        backbone = resnet50(weights=WEIGHTS)
        backbone.fc = nn.Identity()
        return cls(backbone, device, input_size=input_size)

    @torch.inference_mode()
    def extract(self, images: Iterable[Image.Image | BGRFrame]) -> np.ndarray:
        """将 images 中已转正的 PIL RGB 图或 OpenCV BGR 帧转换为 2048 维特征。"""
        config = self.preprocessing
        batch = np.stack([
            preprocess(
                image, target_size=config["target_size"], fill=config["fill"],
                color=config["color"], dtype=config["dtype"], scale=config["scale"],
                mean=config["mean"], std=config["std"],
            )[0]
            for image in images
        ])
        inputs = torch.from_numpy(batch).permute(0, 3, 1, 2).to(self.device)
        return self.backbone(inputs).cpu().numpy()


@dataclass(frozen=True)
class LinearHead:
    """保存 train 拟合的标准化参数和二元逻辑回归权重。"""

    mean: torch.Tensor
    scale: torch.Tensor
    coefficients: torch.Tensor
    intercept: float

    def probabilities(self, features: np.ndarray) -> np.ndarray:
        """按保存的参数将二维 features 矩阵转换为一维 FAIL 概率。"""
        normalized = features.astype(np.float64)
        normalized -= self.mean.numpy()
        normalized /= self.scale.numpy()
        logits = normalized @ self.coefficients.numpy()
        logits += self.intercept
        return torch.from_numpy(logits).sigmoid().numpy()


class CNNLinearModel:
    """提供带阈值的 baseline2，加载后可反复预测。"""

    model_id = "baseline2"
    model_name = "ResNet-50 + 逻辑回归"

    def __init__(
        self, extractor: FeatureExtractor, head: LinearHead, threshold: float, path: Path,
    ) -> None:
        """绑定特征提取器 extractor、逻辑回归 head、阈值 threshold 和权重路径 path。"""
        self.extractor = extractor
        self.head = head
        self.threshold = threshold
        self.model_path = path

    @classmethod
    def load(cls, path: Path | None = None) -> "CNNLinearModel":
        """从 path 离线加载包含输入尺寸的权重；省略 path 时加载默认部署权重。"""
        path = (DEFAULT_MODEL_PATH if path is None else Path(path)).resolve()
        saved = torch.load(path, map_location="cpu", weights_only=True)
        preprocessing = saved["preprocessing"]
        height, width = preprocessing["target_size"]
        if (saved["kind"] != CHECKPOINT_KIND or saved["pretrained_weights"] != WEIGHTS.name
                or height != width or height not in SUPPORTED_INPUT_SIZES
                or preprocessing != {**_PREPROCESSING, "target_size": (height, width)}
                or saved["labels"] != LABELS):
            raise ValueError(f"不兼容的 baseline2 模型文件：{path}")
        with torch.device("meta"):
            backbone = resnet50(weights=None)
            backbone.fc = nn.Identity()
        backbone.load_state_dict(saved["backbone"], assign=True)
        return cls(
            FeatureExtractor(backbone, compute_device(), input_size=height), LinearHead(**saved["linear_head"]),
            saved["threshold"], path,
        )

    def save(self) -> None:
        """将当前模型的全部推理参数写入 model_path，不覆盖已有文件。"""
        with self.model_path.open("xb") as stream:
            torch.save({
                "kind": CHECKPOINT_KIND,
                "pretrained_weights": WEIGHTS.name,
                "preprocessing": self.extractor.preprocessing,
                "labels": LABELS,
                "threshold": self.threshold,
                "backbone": self.extractor.backbone.state_dict(),
                "linear_head": vars(self.head),
            }, stream)

    def predict(self, image: Image.Image | BGRFrame) -> float:
        """预测一张 EXIF 已转正的 PIL RGB 图或一帧 OpenCV BGR 的 FAIL 概率。"""
        return float(self.head.probabilities(self.extractor.extract([image]))[0])

    def predict_bgr(self, frame: BGRFrame) -> float:
        """为摄像头和公共 BGR 推理入口计算 frame 的 FAIL 概率。"""
        return self.predict(frame)
