"""清洁机器人视觉系统命令行入口。"""

import argparse # 读取命令行参数
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from .capture.camera import CameraError
from .capture.preview import run_capture
from .capture.storage import SaveError
from .inference import DEFAULT_CAMERA, DEFAULT_FONT, DEFAULT_OUTPUT, MODEL_IDS, run
from .postprocessing import postprocess_classification, postprocess_detection
from .preprocessing import preprocess
from .visualization import ResultRenderer


def _camera_source(value: str) -> int | str:
    """区分命令行设备索引与视频流地址。

    参数：value 为命令行 str；可转换为整数的内容作为设备索引，其余保留为
    OpenCV 视频流地址或文件路径，相对文件路径以进程工作目录为基准。
    返回：int 设备索引或 str 地址/路径。异常：无预期异常。
    """
    try:
        return int(value)
    except ValueError:
        return value


def _positive_pixels(value: str) -> int:
    """解析正整数像素尺寸。

    参数：value 为宽或高的命令行 str，单位像素，无默认值。
    返回：正整数 int。异常：非整数或非正数抛出 argparse.ArgumentTypeError。
    """
    try:
        pixels = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("尺寸必须为正整数（像素）。") from error
    if pixels <= 0:
        raise argparse.ArgumentTypeError("尺寸必须为正整数（像素）。")
    return pixels


def _run_offline(args: argparse.Namespace) -> None:
    """按 args 指定的输入、画布尺寸和阈值回放已有预测并保存标注图。"""
    if args.output.suffix.lower() != ".png":
        raise ValueError("输出必须为 .png 文件")
    if args.output.exists():
        raise FileExistsError(f"输出文件已存在，拒绝覆盖：{args.output}")
    with args.predictions.open(encoding="utf-8") as stream:
        prediction = json.load(stream)
    if not isinstance(prediction, dict):
        raise ValueError("预测 JSON 必须为对象")
    task = prediction.get("task")
    if task not in ("classification", "detection"):
        raise ValueError("预测 task 必须为 classification 或 detection")
    model_name = prediction.get("model_name")
    if not isinstance(model_name, str) or not model_name.strip():
        raise ValueError("预测 model_name 必须为非空字符串")
    required = (
        ("fail_probability",) if task == "classification"
        else ("boxes", "class_ids", "scores")
    )
    missing = [key for key in required if key not in prediction]
    if missing:
        raise ValueError(f"预测 JSON 缺少字段：{', '.join(missing)}")

    renderer = ResultRenderer(args.font)
    with Image.open(args.input) as source:
        frame = cv2.cvtColor(np.array(source.convert("RGB")), cv2.COLOR_RGB2BGR)
    # 离线预测已由外部模型产生；此处复用真实预处理得到匹配的坐标变换。
    _, transform = preprocess(frame, target_size=tuple(args.target_size))
    if task == "classification":
        probability = prediction["fail_probability"]
        if not isinstance(probability, (int, float)) or isinstance(probability, bool):
            raise ValueError("fail_probability 必须为数值")
        result = postprocess_classification(probability, threshold=args.threshold)
    else:
        for key in required:
            if not isinstance(prediction[key], list):
                raise ValueError(f"{key} 必须为 JSON 数组")
        boxes = np.asarray(prediction["boxes"], dtype=np.float64)
        if boxes.shape == (0,):
            boxes = boxes.reshape(0, 4)
        result = postprocess_detection(
            boxes, prediction["class_ids"], prediction["scores"], transform,
            threshold=args.threshold,
        )
        if prediction.get("model_id") == "baseline3":
            model_name = f"baseline3 · {model_name}"
    annotated = renderer.draw(
        frame, result, model_name=model_name, elapsed_ms=args.elapsed_ms,
    )
    # 独占创建；即使处理期间同名文件出现，也不覆盖调用方的文件。
    with args.output.open("xb") as stream:
        Image.fromarray(annotated[:, :, ::-1]).save(stream, format="PNG")
    print(f"已生成：{args.output.resolve()} | {result.decision}")


def main() -> None:
    """解析统一命令行并执行采集、已有预测处理或真实模型推理。

    参数：无，从进程命令行读取参数，相对路径基于进程工作目录。
    返回：None。异常：帮助或参数错误由 argparse 抛出 SystemExit；运行错误
    报告原因并以状态码 1 退出，键盘中断以状态码 130 退出。
    """

    # 定义允许的命令行参数
    parser = argparse.ArgumentParser(
        prog="robot-cleaning-vision",
        description="清洁机器人视觉系统。",
        epilog="架构与各模块说明见 docs/架构设计.md。",
    )
    subparsers = parser.add_subparsers(dest="command", title="子命令")
    capture = subparsers.add_parser(
        "capture", help="实时采集、预览并保存原始图片和元数据。",
        description="同一物理布置共用 scene_id，改变物体布置后请指定新的 scene_id。",
    )
    capture.add_argument(
        "--camera", type=_camera_source, default=0, metavar="SOURCE",
        help="摄像头整数索引（默认 0），或 OpenCV 可读取的视频流地址/视频文件路径。",
    )
    capture.add_argument(
        "--output", default="data/raw/local", metavar="DIR",
        help="输出根目录，相对进程工作目录；默认 data/raw/local，CSV 位于此目录。",
    )
    capture.add_argument(
        "--scene-id", required=True, metavar="SCENE",
        help="物理场景的单级目录名；同一布置重复使用，改变布置后更换。",
    )
    capture.add_argument(
        "--width", type=_positive_pixels, metavar="PIXELS",
        help="请求采集宽度（正整数像素），默认使用设备配置；实际宽度以读帧为准。",
    )
    capture.add_argument(
        "--height", type=_positive_pixels, metavar="PIXELS",
        help="请求采集高度（正整数像素），默认使用设备配置；实际高度以读帧为准。",
    )
    process = subparsers.add_parser(
        "process", help="读取原图和已有预测 JSON，后处理并输出中文标注 PNG。",
        description="不执行模型推理；检测框必须对应相同原图的指定 letterbox 画布。",
    )
    process.add_argument("--input", type=Path, required=True, help="原始图片路径。")
    process.add_argument(
        "--predictions", type=Path, required=True, help="UTF-8 模型原始预测 JSON。",
    )
    process.add_argument(
        "--target-size", type=_positive_pixels, nargs=2, required=True,
        metavar=("HEIGHT", "WIDTH"), help="产生预测时的预处理画布尺寸，顺序为高、宽。",
    )
    process.add_argument(
        "--threshold", type=float, required=True, help="分类 FAIL 概率或检测得分阈值，[0, 1]。",
    )
    process.add_argument("--font", type=Path, required=True, help="中文字体文件路径。")
    process.add_argument(
        "--elapsed-ms", type=float, help="外部最近完成帧的毫秒耗时；省略则显示 —。",
    )
    process.add_argument(
        "--output", type=Path, required=True, help="新标注 PNG 路径，父目录须存在，不覆盖。",
    )
    inference = subparsers.add_parser(
        "run", help="运行分类或目标检测模型，识别图片或实时摄像头画面。",
    )
    inference.add_argument(
        "--model", choices=MODEL_IDS, required=True, help="指定本次推理使用的视觉模型。",
    )
    inference.add_argument(
        "--model-path", type=Path, help="指定所选模型的权重文件；省略时使用默认权重。",
    )
    inference.add_argument(
        "--input", type=Path, help="原始图片路径；省略时打开摄像头。",
    )
    inference.add_argument(
        "--camera", type=_camera_source, default=DEFAULT_CAMERA, metavar="SOURCE",
        help="摄像头整数索引（默认 0），或视频流地址/本地视频路径。",
    )
    inference.add_argument(
        "--font", type=Path, default=DEFAULT_FONT, help=f"中文字体，默认 {DEFAULT_FONT}。",
    )
    inference.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT, metavar="DIR",
        help=f"输出根目录，默认 {DEFAULT_OUTPUT}；每次保存新建目录，不覆盖。",
    )

    # 解析命令行参数
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        if args.command == "capture":
            run_capture(
                args.camera, args.output, args.scene_id, width=args.width, height=args.height,
            )
        elif args.command == "run":
            run(
                input_path=args.input, camera=args.camera, font=args.font, output=args.output,
                model_id=args.model, model_path=args.model_path,
            )
        else:
            _run_offline(args)
    except KeyboardInterrupt:
        parser.exit(130, "操作已中断，已打开的资源已释放。\n")
    except (CameraError, SaveError, ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"{args.command} 失败：{error}\n")


if __name__ == "__main__":
    main()
