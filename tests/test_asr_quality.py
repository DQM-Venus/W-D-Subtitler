"""ASR 质量评分回归测试。"""

import unittest

from wd_subtitler.asr_service import WhisperASRService


class ASRQualityTests(unittest.TestCase):
    def test_高质量片段无需复核(self):
        result = WhisperASRService._evaluate_segment_quality({
            "avg_logprob": -0.3,
            "no_speech_prob": 0.05,
            "compression_ratio": 1.2,
            "decoding_temperature": 0.0,
        })

        self.assertEqual(100, result["quality_score"])
        self.assertFalse(result["needs_review"])
        self.assertFalse(result["low_confidence"])

    def test_边缘片段进入复核但不标记听不清(self):
        result = WhisperASRService._evaluate_segment_quality({
            "avg_logprob": -0.95,
            "no_speech_prob": 0.1,
            "compression_ratio": 1.4,
            "decoding_temperature": 0.0,
        })

        self.assertEqual(70, result["quality_score"])
        self.assertTrue(result["needs_review"])
        self.assertFalse(result["low_confidence"])
        self.assertIn("识别概率偏低", result["review_reasons"])

    def test_多项风险会标记低置信度(self):
        result = WhisperASRService._evaluate_segment_quality({
            "avg_logprob": -1.4,
            "no_speech_prob": 0.8,
            "compression_ratio": 2.7,
            "decoding_temperature": 0.4,
        })

        self.assertEqual(0, result["quality_score"])
        self.assertTrue(result["needs_review"])
        self.assertTrue(result["low_confidence"])
        self.assertGreaterEqual(len(result["review_reasons"]), 4)

    def test_单项严重风险会标记低置信度(self):
        result = WhisperASRService._evaluate_segment_quality({
            "avg_logprob": -1.4,
            "no_speech_prob": 0.1,
            "compression_ratio": 1.2,
            "decoding_temperature": 0.0,
        })

        self.assertEqual(55, result["quality_score"])
        self.assertTrue(result["low_confidence"])

    def test_离散裁剪区间能正确分组(self):
        timestamps = [1.0, 2.0, 2.2, 3.0]

        self.assertEqual(0, WhisperASRService._find_clip_group(1.1, 1.8, timestamps))
        self.assertEqual(1, WhisperASRService._find_clip_group(2.3, 2.8, timestamps))
        self.assertIsNone(WhisperASRService._find_clip_group(4.0, 5.0, timestamps))

    def test_常见礼貌语不会仅凭文本命中被删除(self):
        service = object.__new__(WhisperASRService)

        self.assertFalse(service._is_hallucination(
            "ありがとうございました",
            avg_logprob=-0.2,
            no_speech_prob=0.05,
            compression_ratio=1.1,
        ))

    def test_常见礼貌语伴随多项声学风险会被过滤(self):
        service = object.__new__(WhisperASRService)

        self.assertTrue(service._is_hallucination(
            "ありがとうございました",
            avg_logprob=-1.2,
            no_speech_prob=0.8,
            compression_ratio=1.1,
        ))


if __name__ == "__main__":
    unittest.main()
