"""摄像头后端调整后，视频输入仍须正常解码、保持BGR并在结束时释放。"""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import cv2
import numpy as np

from robot_cleaning_vision.capture.camera import Camera, CameraError


class CameraVideoTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / 'frames.avi'
        writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*'MJPG'), 5, (64, 48))
        self.assertTrue(writer.isOpened())
        for color in ((255, 0, 0), (0, 0, 255)):
            writer.write(np.full((48, 64, 3), color, dtype=np.uint8))
        writer.release()

    def test_file_input_keeps_frame_order_and_bgr_channels(self):
        with Camera(str(self.path)) as camera:
            blue = camera.read()
            red = camera.read()
        self.assertEqual(blue.shape, (48, 64, 3))
        self.assertEqual(blue.dtype, np.uint8)
        self.assertGreater(blue[:, :, 0].mean(), 240)
        self.assertLess(blue[:, :, 2].mean(), 15)
        self.assertGreater(red[:, :, 2].mean(), 240)
        self.assertLess(red[:, :, 0].mean(), 15)

    def test_end_of_video_reports_error_and_releases_file(self):
        camera = Camera(str(self.path)).open()
        try:
            camera.read()
            camera.read()
            with self.assertRaises(CameraError):
                camera.read()
            # 设备释放后视频可被删除，且再次读取按“未打开”处理。
            self.path.unlink()
            with self.assertRaisesRegex(CameraError, '尚未打开'):
                camera.read()
        finally:
            camera.release()


if __name__ == '__main__':
    unittest.main()
