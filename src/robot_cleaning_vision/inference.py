"""在图片和摄像头会话中运行所选分类模型、绘制并保存预测。"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter
from typing import Protocol

import cv2
from PIL import Image, ImageOps
import numpy as np

from .capture.camera import BGRFrame, Camera
from .models.mlp.model import DEFAULT_MODEL_PATH, MLPModel
from .postprocessing import ImageResult, postprocess_classification
from .visualization import ResultRenderer

MODEL_IDS = ("baseline1", "baseline2")
DEFAULT_CAMERA: int | str = 0
DEFAULT_FONT = Path("C:/Windows/Fonts/msyh.ttc")
DEFAULT_OUTPUT = Path("artifacts/inference")


class Classifier(Protocol):
    """实时分类入口需要的模型身份、阈值与 BGR 预测方法。"""

    model_id: str
    model_name: str
    model_path: Path
    threshold: float

    def predict_bgr(self, frame: BGRFrame) -> float:
        """返回 OpenCV BGR 帧 frame 的整图 FAIL 概率。"""
        ...


@dataclass
class FramePrediction:
    """同一帧的原始 BGR、中文渲染结果和实际推理信息。"""

    original: BGRFrame
    annotated: BGRFrame
    result: ImageResult
    threshold: float
    elapsed_ms: float
    model_id: str
    model_name: str
    model_path: Path


def classify_frame(
    frame: BGRFrame, model: Classifier, renderer: ResultRenderer, *,
    started_at: float | None = None,
) -> FramePrediction:
    """用 model 推理 BGR 原图 frame，再用 renderer 绘制到副本上。

    未提供 started_at 时从本函数入口计时。绘图、显示和磁盘保存不计入推理耗时。
    """
    if started_at is None:
        started_at = perf_counter()
    probability = model.predict_bgr(frame)
    threshold = float(model.threshold)
    result = postprocess_classification(probability, threshold=threshold)
    elapsed_ms = (perf_counter() - started_at) * 1000
    annotated = renderer.draw(
        frame, result, model_name=f"{model.model_id} · {model.model_name}", elapsed_ms=elapsed_ms,
    )
    return FramePrediction(
        frame, annotated, result, threshold, elapsed_ms,
        model.model_id, model.model_name, model.model_path,
    )


def save_prediction(
    prediction: FramePrediction, output: Path, *, source: str | int,
) -> Path:
    """独占新建一次结果目录，保存当前显示帧，绝不重新读帧或重新推理。"""
    output.mkdir(parents=True, exist_ok=True)
    directory = Path(mkdtemp(prefix=datetime.now().strftime("%Y%m%d_%H%M%S_"), dir=output))
    for filename, frame in (
        ("original.png", prediction.original), ("result.png", prediction.annotated),
    ):
        with (directory / filename).open("xb") as stream:
            Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).save(stream, format="PNG")
    payload = {
        "task": "classification",
        "model_id": prediction.model_id,
        "model_name": prediction.model_name,
        "model_path": str(prediction.model_path),
        "source": source,
        "fail_probability": prediction.result.image_score,
        "threshold": prediction.threshold,
        "decision": prediction.result.decision,
        "elapsed_ms": prediction.elapsed_ms,
    }
    with (directory / "prediction.json").open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        f"已保存（{prediction.model_id} | {prediction.result.decision} | "
        f"FAIL 概率：{prediction.result.image_score:.6f}）：\n"
        f"  原图：{(directory / 'original.png').resolve()}\n"
        f"  预测：{(directory / 'prediction.json').resolve()}\n"
        f"  中文结果：{(directory / 'result.png').resolve()}",
        flush=True,
    )
    return directory


def run_image(
    path: Path, model: Classifier, renderer: ResultRenderer, output: Path,
) -> Path:
    """按 EXIF 转正原始图片，真实推理并保存 PNG 和 JSON。"""
    started_at = perf_counter()
    with Image.open(path) as image:
        rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    prediction = classify_frame(frame, model, renderer, started_at=started_at)
    return save_prediction(prediction, output, source=str(path.resolve()))


def run_camera(
    source: int | str, model: Classifier, renderer: ResultRenderer, output: Path,
) -> None:
    """按键保存实际显示帧；所有退出路径释放相机和窗口。

    Camera 未区分 EOF 与读帧故障，故本地视频结束也明确报告 CameraError，
    不吞掉其他采集故障；本地视频 smoke 可在结束前按 Q/Esc 或关闭窗口。
    """
    camera = Camera(source)
    try:
        camera.open()
        window_name = f"Robot Cleaning Vision - {model.model_id} | S/Space: save | Q/Esc: quit"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        print("S / 空格：保存当前帧；Q / Esc / 关闭窗口：退出。", flush=True)
        while True:
            started_at = perf_counter()
            frame = camera.read()
            prediction = classify_frame(frame, model, renderer, started_at=started_at)
            cv2.imshow(window_name, prediction.annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S"), 32):
                save_prediction(prediction, output, source=source)
    finally:
        try:
            camera.release()
        finally:
            cv2.destroyAllWindows()


def run(
    *, model_id: str, input_path: Path | None = None, camera: int | str = DEFAULT_CAMERA,
    font: Path = DEFAULT_FONT, output: Path = DEFAULT_OUTPUT,
    model_path: Path | None = None,
) -> None:
    """按 model_id 选择模型，每次会话只加载一次 model_path 权重及中文字体。

    input_path 指定原图；省略时读取 camera。font 为字体路径，output 为结果目录。
    model_path 省略时使用所选模型的默认权重。
    """
    if model_id == "baseline1":
        model = MLPModel.load(model_path if model_path is not None else DEFAULT_MODEL_PATH)
        model.threshold = 0.5  # 保留 baseline1 实测入口使用的阈值。
    elif model_id == "baseline2":
        from .models.cnn_linear.model import CNNLinearModel

        model = CNNLinearModel.load(model_path)
    else:
        raise ValueError(f"未知分类模型：{model_id}")
    print(f"模型：{model.model_id} · {model.model_name} | 权重：{model.model_path}", flush=True)
    renderer = ResultRenderer(font)
    if input_path is not None:
        run_image(input_path, model, renderer, output)
    else:
        run_camera(camera, model, renderer, output)
