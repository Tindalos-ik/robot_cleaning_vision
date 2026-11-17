"""不依赖窗口或按键的 OpenCV 采集接口。"""

from types import TracebackType

import sys
import cv2
import numpy as np
from numpy.typing import NDArray

# 声明 一个 BGR 图像帧类型，一个数组，元素为 uint8，形状为 (高, 宽, 3)，3 是 BGR 通道数
BGRFrame = NDArray[np.uint8]


class CameraError(RuntimeError):
    """设备打开或读帧失败。"""


class Camera:
    """管理单个采集设备；与窗口无关，可供后续模型主循环复用。"""

    def __init__(
        self, source: int | str = 0, *, width: int | None = None,
        height: int | None = None,
    ) -> None:
        """配置设备但不打开。

        参数：source 为 int 设备索引（默认 0），或 str 视频流地址/视频文件路径；
        相对文件路径以进程工作目录为基准。
        width、height 分别为期望宽、高，
        单位像素，默认 None 表示采用设备默认值，设备可能忽略设置。
        返回：None。
        """

        self.source = source
        self.width = width
        self.height = height
        self._device: cv2.VideoCapture | None = None

    def open(self) -> "Camera":
        """打开配置的设备并请求分辨率，已打开时复用设备。

        参数：无，使用构造时的来源和宽、高（像素）。
        返回：当前 Camera 实例。异常：打开或配置失败抛出 CameraError；
        失败前已创建的设备也会释放。

        Windows 的整数摄像头索引使用 DirectShow，避免部分设备在 MSMF 下无法读帧；
        视频文件、网络流和其他系统仍由 OpenCV 自动选择后端。
        """
        if self._device is not None:
            return self
        device = cv2.VideoCapture()
        backend = cv2.CAP_ANY
        if sys.platform == "win32" and isinstance(self.source, int):
            backend = cv2.CAP_DSHOW
        try:
            if not device.open(self.source, backend):
                raise CameraError(f"无法打开摄像头或视频流：{self.source!r}")
            if self.width is not None:
                device.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            if self.height is not None:
                device.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        except BaseException as error:
            device.release()
            if isinstance(error, cv2.error):
                raise CameraError(f"打开或配置采集源 {self.source!r} 失败：{error}") from error
            raise
        self._device = device
        return self

    def read(self) -> BGRFrame:
        """读取一帧原始图像，不缩放、不叠字、不转换通道。

        参数：无，须先 open()。
        返回：numpy.ndarray，shape=(height, width, 3)，
        通道顺序 BGR，dtype=uint8；宽、高以本次实际帧为准，单位像素。
        异常：未打开、读取失败（包括流结束）或帧格式不符时抛出 CameraError。
        读帧失败会立即释放设备；正常读取后由调用者 release() 或 with 释放。
        """
        if self._device is None:
            raise CameraError("采集设备尚未打开，请先调用 open()。")
        try:
            success, frame = self._device.read()
            if not success or frame is None or frame.size == 0:
                raise CameraError(f"读取采集源 {self.source!r} 失败：未获得图像或视频流已结束。")
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
                raise CameraError(f"采集源 {self.source!r} 未返回 H×W×3、uint8 的 BGR 图像。")
        except BaseException as error:
            self.release()
            if isinstance(error, cv2.error):
                raise CameraError(f"读取采集源 {self.source!r} 失败：{error}") from error
            raise
        return frame

    def release(self) -> None:
        """释放设备，重复调用安全，便于统一处理正常与异常退出。

        参数：无。返回：None。异常：底层 OpenCV 释放异常原样传播。
        """
        device, self._device = self._device, None
        if device is not None:
            device.release()

    def __enter__(self) -> "Camera":
        """进入上下文并打开设备。

        参数：无。返回：已打开的 Camera。异常：open() 的 CameraError 原样传播。
        """
        return self.open()

    def __exit__(
        self, exc_type: type[BaseException] | None,
        exc_value: BaseException | None, traceback: TracebackType | None,
    ) -> None:
        """退出上下文时释放设备，不吞掉主循环异常。

        参数：exc_type 为异常类型或 None；exc_value 为异常对象或 None；
        traceback 为回溯对象或 None，均由 with 传入，不影响释放行为。
        返回：None，原异常继续传播。异常：release() 的底层异常原样传播。
        """
        self.release()
