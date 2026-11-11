"""单线程预览与按键调度；采集和保存接口不依赖此模块。"""

from datetime import datetime
from pathlib import Path
import sys

import cv2

from .camera import BGRFrame, Camera
from .storage import ImageStore, SaveError

WINDOW_NAME = "Robot Cleaning Vision - Capture"


def draw_preview(frame: BGRFrame, scene_id: str, saved_count: int, status: str) -> BGRFrame:
    """在副本上绘制场景、本次成功数量、操作提示及最近保存状态。

    参数：frame 为原始 BGR ndarray，shape=(height, width, 3)，dtype=uint8，
    宽、高单位像素；scene_id 为场景字符串；saved_count 为本次成功保存的 int
    数量；status 为英文状态字符串，无额外单位或默认值。
    返回：相同 shape、dtype 的独立 BGR 副本，输入不被修改。OpenCV 内置字体
    仅支持 ASCII，非 ASCII 场景在画面中转义，窗口标题另显示完整场景名。
    异常：OpenCV 绘制错误抛出 cv2.error；内存不足抛出 MemoryError。
    """
    preview = frame.copy()
    scene_text = scene_id.encode("ascii", errors="backslashreplace").decode("ascii")
    lines = (
        f"Scene: {scene_text} | Saved: {saved_count}",
        "S / Space: save | Q / Esc: quit",
        status,
    )
    cv2.rectangle(preview, (0, 0), (preview.shape[1] - 1, 86), (0, 0, 0), -1)
    for index, line in enumerate(lines):
        cv2.putText(
            preview, line, (8, 23 + index * 26), cv2.FONT_HERSHEY_SIMPLEX,
            0.55, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return preview


def run_capture(
    source: int | str, output: str | Path, scene_id: str, *,
    width: int | None = None, height: int | None = None,
) -> int:
    """持续读取和显示新帧，按键保存当前显示帧，所有退出路径均释放资源。

    参数：source 为 int 摄像头索引，或 str OpenCV 视频流地址/文件路径；
    output 为 str/Path 输出根目录，相对路径以进程工作目录为基准；scene_id
    为同一物理布置共用的 str 场景名；width、height 为请求宽、高（像素），
    默认 None 使用设备默认尺寸，实际保存尺寸以读帧为准。
    返回：int，本次成功保存数量（不含历史记录）。S/s 或空格保存，Q/q、Esc
    或关闭窗口退出。保存错误打印原因并允许继续操作，成功则打印完整路径。
    异常：打开/读取失败抛出 CameraError；目录准备失败抛出 SaveError；参数
    错误抛出 ValueError；显示失败抛出 cv2.error；中断抛出 KeyboardInterrupt。
    """
    store = ImageStore(output, scene_id, source)
    camera = Camera(source, width=width, height=height)
    saved_count = 0
    status = "Ready"
    try:
        camera.open()
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        while True:
            frame = camera.read()
            captured_at = datetime.now().astimezone()
            preview = draw_preview(frame, scene_id, saved_count, status)
            cv2.setWindowTitle(WINDOW_NAME, f"{WINDOW_NAME} | {scene_id} | Saved: {saved_count}")
            cv2.imshow(WINDOW_NAME, preview)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("s"), ord("S"), 32):
                try:
                    # 在下一次 read() 前保存本轮原图，避免按键与显示帧错位。
                    record = store.save(frame, captured_at=captured_at)
                except SaveError as error:
                    status = "Save failed (see terminal)"
                    print(f"保存失败：{error}", file=sys.stderr, flush=True)
                else:
                    saved_count += 1
                    status = f"Saved #{saved_count}"
                    print(f"已保存：{store.output_dir / record.file_name}", flush=True)
    finally:
        # 不依赖垃圾回收；读帧、绘制、键盘中断等异常同样走释放路径。
        try:
            camera.release()
        finally:
            cv2.destroyAllWindows()
    return saved_count
