"""字幕可靠保存测试。"""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from wd_subtitler.subtitle_writer import SubtitleSaveError, save_subtitles


class SubtitleWriterTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base = Path(self.temp_dir.name) / "测试视频"
        self.segments = [
            {"start": 0.0, "end": 1.2, "text": "原文一", "trans": "译文一"},
            {"start": 1.3, "end": 2.5, "text": "原文二", "trans": ""},
        ]

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_已有字幕不会被覆盖(self):
        original = self.base.with_suffix(".srt")
        original.write_text("旧字幕", encoding="utf-8")

        result = save_subtitles(self.base, self.segments, "srt")

        self.assertEqual("旧字幕", original.read_text(encoding="utf-8"))
        self.assertEqual("测试视频 (1).srt", result.path.name)
        self.assertTrue(result.path.exists())

    def test_取消翻译保存为_partial并标记未翻译行(self):
        result = save_subtitles(
            self.base,
            self.segments,
            "srt",
            with_translation=True,
            partial=True,
        )

        content = result.path.read_text(encoding="utf-8")
        self.assertEqual("测试视频.partial.srt", result.path.name)
        self.assertIn("译文一", content)
        self.assertIn("[未翻译] 原文二", content)

    def test_原子提交失败不留下正式字幕或临时文件(self):
        with patch(
            "wd_subtitler.subtitle_writer._publish_without_overwrite",
            side_effect=OSError("磁盘错误"),
        ):
            with self.assertRaises(SubtitleSaveError):
                save_subtitles(self.base, self.segments, "srt")

        self.assertFalse(self.base.with_suffix(".srt").exists())
        self.assertEqual([], list(Path(self.temp_dir.name).glob("*.tmp")))

    def test_并发保存同名字幕时自动编号且互不覆盖(self):
        results = []
        errors = []

        def write_once():
            try:
                results.append(save_subtitles(self.base, self.segments, "srt"))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=write_once) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual([], errors)
        self.assertEqual({"测试视频.srt", "测试视频 (1).srt"}, {item.path.name for item in results})


if __name__ == "__main__":
    unittest.main()
