"""唯一 MLP 的图片与摄像头实测入口；所有图像边界均明确区分 RGB/BGR。"""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

import cv2
from PIL import Image, ImageOps
import numpy as np

from .capture.camera import BGRFrame, Camera
from .models.mlp.model import DEFAULT_MODEL_PATH, MLPModel
from .postprocessing import ImageResult, postprocess_classification
from .visualization import ResultRenderer

# 常用配置；命令行可覆盖采集源、字体和输出根目录，模型及阈值固定随权重文件。
DEFAULT_CAMERA: int | str = 0
DEFAULT_FONT = Path("C:/Windows/Fonts/msyh.ttc")
DEFAULT_OUTPUT = Path("artifacts/inference")
WINDOW_NAME = "Robot Cleaning Vision - MLP | S/Space: save | Q/Esc: quit"
MODEL_NAME = "MLP"


@dataclass
class FramePrediction:
    """同一帧的原始 BGR、中文渲染结果和实际推理信息。"""

    original: BGRFrame
    annotated: BGRFrame
    result: ImageResult
    threshold: float
    elapsed_ms: float


def classify_frame(
    frame: BGRFrame, model: MLPModel, renderer: ResultRenderer, *,
    started_at: float | None = None,
) -> FramePrediction:
    """BGR 转 RGB 后推理，不修改原图；耗时从本帧采集/解码到后处理完成。

    未提供 started_at 时从本函数入口计时。绘图、显示和磁盘保存不计入推理耗时。
    """
    if started_at is None:
        started_at = perf_counter()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    probability = model.predict(rgb)
    threshold = float(model.threshold)
    result = postprocess_classification(probability, threshold=threshold)
    elapsed_ms = (perf_counter() - started_at) * 1000
    annotated = renderer.draw(
        frame, result, model_name=MODEL_NAME, elapsed_ms=elapsed_ms,
    )
    return FramePrediction(frame, annotated, result, threshold, elapsed_ms)


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
        "model_name": MODEL_NAME,
        "model_path": str(DEFAULT_MODEL_PATH),
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
        f"已保存（{prediction.result.decision}）：\n"
        f"  原图：{(directory / 'original.png').resolve()}\n"
        f"  预测：{(directory / 'prediction.json').resolve()}\n"
        f"  中文结果：{(directory / 'result.png').resolve()}",
        flush=True,
    )
    return directory


def run_image(
    path: Path, model: MLPModel, renderer: ResultRenderer, output: Path,
) -> Path:
    """按 EXIF 转正原始图片，真实推理并保存 PNG 和 JSON。"""
    started_at = perf_counter()
    with Image.open(path) as image:
        rgb = np.array(ImageOps.exif_transpose(image).convert("RGB"))
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    prediction = classify_frame(frame, model, renderer, started_at=started_at)
    return save_prediction(prediction, output, source=str(path.resolve()))


def run_camera(
    source: int | str, model: MLPModel, renderer: ResultRenderer, output: Path,
) -> None:
    """按键保存实际显示帧；所有退出路径释放相机和窗口。

    Camera 未区分 EOF 与读帧故障，故本地视频结束也明确报告 CameraError，
    不吞掉其他采集故障；本地视频 smoke 可在结束前按 Q/Esc 或关闭窗口。
    """
    camera = Camera(source)
    try:
        camera.open()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        print("S / 空格：保存当前帧；Q / Esc / 关闭窗口：退出。", flush=True)
        while True:
            started_at = perf_counter()
            frame = camera.read()
            prediction = classify_frame(frame, model, renderer, started_at=started_at)
            cv2.imshow(WINDOW_NAME, prediction.annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S"), 32):
                save_prediction(prediction, output, source=source)
    finally:
        try:
            camera.release()
        finally:
            cv2.destroyAllWindows()


def run(
    *, input_path: Path | None = None, camera: int | str = DEFAULT_CAMERA,
    font: Path = DEFAULT_FONT, output: Path = DEFAULT_OUTPUT,
) -> None:
    """每次运行只加载一次固定最优模型及中文字体。"""
    model = MLPModel.load()
    model.threshold = 0.5  # 实测判定阈值，50%
    renderer = ResultRenderer(font)
    if input_path is not None:
        run_image(input_path, model, renderer, output)
    else:
        run_camera(camera, model, renderer, output)
