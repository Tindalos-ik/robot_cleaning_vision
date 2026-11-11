"""验证原图保存、跨次追加和写入失败时的数据完整性。"""

import csv
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from uuid import UUID

import cv2
import numpy as np

from robot_cleaning_vision.capture.storage import ImageStore, SaveError


class ImageStoreTests(unittest.TestCase):
    """使用生成的 BGR 图像和临时目录，不依赖摄像头或桌面。"""

    def test_lossless_images_and_records_survive_new_session(self) -> None:
        """验证两次采集保留原像素和记录。

        参数：self 为 unittest 测试实例，无单位。返回：None。
        异常：断言失败抛出 AssertionError，临时文件错误原样传播。
        """
        with TemporaryDirectory() as directory:
            root = Path(directory)
            frame = np.arange(97 * 131 * 3, dtype=np.uint32).astype(np.uint8).reshape(97, 131, 3)
            timestamp = datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
            first = ImageStore(root, "场景_001", 0).save(frame, captured_at=timestamp)
            original_csv = (root / "images.csv").read_bytes()
            second = ImageStore(root, "场景_001", "rtsp://example/camera").save(
                255 - frame, captured_at=timestamp,
            )
            self.assertNotEqual(first.image_id, second.image_id)
            self.assertTrue((root / "images.csv").read_bytes().startswith(original_csv))
            with (root / "images.csv").open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual([row["image_id"] for row in rows], [first.image_id, second.image_id])
            self.assertEqual([row["camera_source"] for row in rows], ["0", "rtsp://example/camera"])
            for row, expected in zip(rows, (frame, 255 - frame), strict=True):
                self.assertEqual(row["scene_id"], "场景_001")
                self.assertEqual(row["file_name"], f"场景_001/{row['image_id']}")
                self.assertEqual((int(row["width"]), int(row["height"])), (131, 97))
                self.assertEqual(datetime.fromisoformat(row["captured_at"]), timestamp)
                image = cv2.imdecode(np.fromfile(root / row["file_name"], dtype=np.uint8), cv2.IMREAD_COLOR)
                np.testing.assert_array_equal(image, expected)

    def test_name_collision_never_overwrites_or_deletes_existing_image(self) -> None:
        """验证即使随机文件名碰撞，也不会覆盖或误删原图。

        参数：self 为 unittest 测试实例，无单位。返回：None。
        异常：断言失败抛出 AssertionError，临时文件错误原样传播。
        """
        with TemporaryDirectory() as directory, patch(
            "robot_cleaning_vision.capture.storage.uuid4", return_value=UUID(int=1),
        ):
            store = ImageStore(directory, "scene_001", 0)
            timestamp = datetime(2026, 1, 2, tzinfo=timezone.utc)
            frame = np.full((20, 30, 3), (1, 2, 3), dtype=np.uint8)
            first = store.save(frame, captured_at=timestamp)
            original = (store.output_dir / first.file_name).read_bytes()
            original_csv = store.csv_path.read_bytes()
            with self.assertRaises(SaveError):
                store.save(255 - frame, captured_at=timestamp)
            self.assertEqual((store.output_dir / first.file_name).read_bytes(), original)
            self.assertEqual(store.csv_path.read_bytes(), original_csv)

    def test_invalid_csv_removes_unregistered_image_and_keeps_old_records(self) -> None:
        """验证已有 CSV 不兼容时保留记录并清理本次图片。

        参数：self 为 unittest 测试实例，无单位。返回：None。
        异常：断言失败抛出 AssertionError，临时文件错误原样传播。
        """
        with TemporaryDirectory() as directory:
            store = ImageStore(directory, "scene_001", 0)
            store.csv_path.write_bytes(b"legacy,header\r\nold,record\r\n")
            before = store.csv_path.read_bytes()
            with self.assertRaisesRegex(SaveError, "CSV 表头不匹配"):
                store.save(np.zeros((20, 30, 3), dtype=np.uint8))
            self.assertEqual(store.csv_path.read_bytes(), before)
            self.assertEqual(list(store.scene_dir.iterdir()), [])

    def test_partial_csv_write_rolls_back_without_damaging_previous_images(self) -> None:
        """验证 CSV 部分写入失败后回退，旧图片和旧记录保持不变。

        参数：self 为 unittest 测试实例，无单位。返回：None。
        异常：断言失败抛出 AssertionError，临时文件错误原样传播。
        """
        def fail_after_partial_write(writer: csv.DictWriter, row: dict[str, object]) -> None:
            """模拟磁盘在写出部分 CSV 后故障。

            参数：writer 为实际 DictWriter；row 为待写元数据字典，无单位。
            返回：不返回。异常：始终抛出 OSError 以覆盖追加回退路径。
            """
            writer.writer.writerow(["partial"])
            raise OSError("simulated disk failure")

        with TemporaryDirectory() as directory:
            store = ImageStore(directory, "scene_001", 0)
            frame = np.zeros((20, 30, 3), dtype=np.uint8)
            first = store.save(frame)
            before = store.csv_path.read_bytes()
            original = (store.output_dir / first.file_name).read_bytes()
            with patch.object(csv.DictWriter, "writerow", autospec=True, side_effect=fail_after_partial_write):
                with self.assertRaisesRegex(SaveError, "simulated disk failure"):
                    store.save(frame)
            self.assertEqual(store.csv_path.read_bytes(), before)
            self.assertEqual((store.output_dir / first.file_name).read_bytes(), original)
            self.assertEqual([path.name for path in store.scene_dir.iterdir()], [first.image_id])


if __name__ == "__main__":
    unittest.main()
