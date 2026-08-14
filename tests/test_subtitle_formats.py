"""字幕时间戳格式化测试。"""

import unittest

from wd_subtitler.subtitle_formats import format_timestamp


class SubtitleFormatTests(unittest.TestCase):
    def test_支持三种时间戳格式(self):
        self.assertEqual("01:01:01,250", format_timestamp(3661.25, "srt"))
        self.assertEqual("[61:01.25]", format_timestamp(3661.25, "lrc"))
        self.assertEqual("01:01:01.250", format_timestamp(3661.25, "vtt"))

    def test_无效和负数时间会归零(self):
        self.assertEqual("00:00:00,000", format_timestamp("无效", "srt"))
        self.assertEqual("00:00:00,000", format_timestamp(-1, "srt"))


if __name__ == "__main__":
    unittest.main()
