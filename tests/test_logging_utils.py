"""统一日志格式回归测试。"""

import unittest

from wd_subtitler.logging_utils import (
    display_width,
    format_log_message,
    format_phase_title,
    infer_log_level,
    normalize_log_message,
)


class LoggingUtilsTests(unittest.TestCase):
    def test_单行日志包含固定时间和中文级别(self):
        result = format_log_message("开始识别", "ASR", timestamp="10:20:30")

        self.assertEqual("[10:20:30] [识别] 开始识别", result)

    def test_多行正文按显示宽度对齐(self):
        result = format_log_message("术语提取完成：\n夏希=なつき", "CONTEXT", timestamp="10:20:30")
        first, second = result.splitlines()
        prefix = "[10:20:30] [上下文] "

        self.assertTrue(first.startswith(prefix))
        self.assertEqual(" " * display_width(prefix) + "夏希=なつき", second)

    def test_旧装饰符和多余空白会被清理(self):
        self.assertEqual(["模型加载完成"], normalize_log_message("  ✅ 模型加载完成  "))

    def test_服务消息可以推断级别(self):
        self.assertEqual("SUCCESS", infer_log_level("✅ 模型加载完成", "ASR"))
        self.assertEqual("WARNING", infer_log_level("⚠️ 改用 CPU", "ASR"))
        self.assertEqual("ASR", infer_log_level("识别进度 50%", "ASR"))

    def test_阶段标题宽度固定(self):
        title = format_phase_title("阶段一 · 语音识别", width=40)

        self.assertIn("阶段一 · 语音识别", title)
        self.assertEqual(40, display_width(title))


if __name__ == "__main__":
    unittest.main()
