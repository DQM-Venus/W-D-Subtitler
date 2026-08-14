"""质量模式界面说明回归测试。"""

import unittest

from wd_subtitler.quality_modes import get_quality_mode_description


class QualityModeDescriptionTests(unittest.TestCase):
    def test_快速模式说明包含速度和复核范围(self):
        text = get_quality_mode_description("快速", True)

        self.assertIn("速度优先", text)
        self.assertIn("仅复核可疑片段", text)
        self.assertIn("局部识别一次", text)

    def test_高质量模式说明包含质量和复核范围(self):
        text = get_quality_mode_description("高质量", True)

        self.assertIn("准确性优先", text)
        self.assertIn("独立识别完整音频", text)
        self.assertIn("局部识别一次", text)

    def test_关闭复核后说明会同步变化(self):
        text = get_quality_mode_description("高质量", False)

        self.assertIn("复核已关闭", text)


if __name__ == "__main__":
    unittest.main()
