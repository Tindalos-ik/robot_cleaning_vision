"""清洁机器人视觉系统命令行入口。"""

import argparse

import cv2

from .capture.camera import CameraError
from .capture.preview import run_capture
from .capture.storage import SaveError


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


def main() -> None:
    """解析统一命令行并执行采集；没有子命令时保留帮助输出。

    参数：无，从进程命令行读取参数，相对路径基于进程工作目录。
    返回：None。异常：帮助或参数错误由 argparse 抛出 SystemExit；运行错误
    报告原因并以状态码 1 退出，键盘中断以状态码 130 退出。
    """
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
    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return
    try:
        run_capture(
            args.camera, args.output, args.scene_id, width=args.width, height=args.height,
        )
    except KeyboardInterrupt:
        parser.exit(130, "采集已中断，设备与窗口已释放。\n")
    except (CameraError, SaveError, ValueError, OSError, cv2.error) as error:
        parser.exit(1, f"采集失败：{error}\n")


if __name__ == "__main__":
    main()
