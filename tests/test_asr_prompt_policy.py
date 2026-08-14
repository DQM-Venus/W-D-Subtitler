"""ASR 热词策略回归测试。"""

import unittest

from wd_subtitler.asr_prompt_policy import (
    get_primary_prompt_options,
    get_review_prompt_options,
    normalize_hotwords,
)


class ASRPromptPolicyTests(unittest.TestCase):
    def test_空热词不会生成任何提示(self):
        self.assertEqual(
            {"initial_prompt": None, "hotwords": None},
            get_primary_prompt_options(""),
        )

    def test_Kotoba仅使用用户填写的热词(self):
        self.assertEqual(
            {"initial_prompt": None, "hotwords": "夏希 吹奏楽部"},
            get_primary_prompt_options("  夏希\n吹奏楽部  "),
        )

    def test_Large_v3不继承热词(self):
        self.assertEqual(
            {"initial_prompt": None, "hotwords": None},
            get_review_prompt_options(),
        )

    def test_热词长度受到限制(self):
        self.assertEqual(500, len(normalize_hotwords("夏" * 600)))


if __name__ == "__main__":
    unittest.main()
