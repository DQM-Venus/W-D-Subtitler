"""Large-v3 可疑片段复核回归测试。"""

import unittest

from wd_subtitler.asr_review import (
    apply_review_candidates,
    build_review_clips,
    flatten_clip_timestamps,
    apply_third_candidates,
    map_full_review_candidates,
    map_review_candidates,
    normalize_asr_text,
)


class ASRReviewTests(unittest.TestCase):
    def test_复核区间不会跨越相邻字幕(self):
        segments = [
            {"start": 1.0, "end": 2.0, "needs_review": False},
            {"start": 2.1, "end": 3.0, "needs_review": True},
            {"start": 3.1, "end": 4.0, "needs_review": False},
        ]

        clips = build_review_clips(segments, padding=0.5)

        self.assertEqual([{"segment_index": 1, "start": 2.0, "end": 3.1}], clips)
        self.assertEqual([2.0, 3.1], flatten_clip_timestamps(clips))

    def test_高质量模式可以强制复核高置信度片段(self):
        segments = [{
            "start": 1.0,
            "end": 2.0,
            "needs_review": False,
            "review_requested": True,
        }]

        clips = build_review_clips(segments)

        self.assertEqual(1, len(clips))

    def test_Large_v3候选按时间映射回原字幕(self):
        clips = [
            {"segment_index": 0, "start": 1.0, "end": 2.0},
            {"segment_index": 2, "start": 5.0, "end": 6.0},
        ]
        review_segments = [
            {
                "start": 1.1,
                "end": 1.5,
                "text": "夏希",
                "quality_score": 90,
                "avg_logprob": -0.2,
                "no_speech_prob": 0.1,
                "compression_ratio": 1.1,
                "decoding_temperature": 0.0,
            },
            {
                "start": 1.5,
                "end": 1.9,
                "text": "先輩",
                "quality_score": 85,
                "avg_logprob": -0.3,
                "no_speech_prob": 0.2,
                "compression_ratio": 1.2,
                "decoding_temperature": 0.0,
            },
        ]

        mapped = map_review_candidates(clips, review_segments)

        self.assertEqual("夏希先輩", mapped[0]["text"])
        self.assertEqual(85, mapped[0]["quality_score"])
        self.assertNotIn(2, mapped)

    def test_明显更好的Large_v3候选会替换(self):
        segments = [{
            "start": 1.0,
            "end": 2.0,
            "text": "心接にしてくれますね",
            "quality_score": 45,
            "needs_review": True,
            "low_confidence": True,
            "review_reasons": ["识别概率很低"],
        }]
        candidates = {0: {
            "text": "親切にしてくれますね",
            "quality_score": 90,
            "avg_logprob": -0.2,
            "no_speech_prob": 0.1,
            "compression_ratio": 1.2,
            "decoding_temperature": 0.0,
        }}

        stats = apply_review_candidates(segments, candidates)

        self.assertEqual("心接にしてくれますね", segments[0]["text"])
        self.assertTrue(segments[0]["needs_ai_review"])
        self.assertEqual(0, stats["replaced"])

    def test_接近分数的冲突候选留给AI(self):
        segments = [{
            "start": 1.0,
            "end": 2.0,
            "text": "候補一",
            "quality_score": 70,
            "needs_review": True,
            "low_confidence": False,
        }]
        candidates = {0: {"text": "候補二", "quality_score": 75}}

        stats = apply_review_candidates(segments, candidates)

        self.assertEqual("候補一", segments[0]["text"])
        self.assertTrue(segments[0]["needs_ai_review"])
        self.assertEqual(1, stats["needs_ai"])

    def test_仅标点不同视为一致(self):
        self.assertEqual(
            normalize_asr_text("なつき、先輩！"),
            normalize_asr_text("なつき先輩"),
        )

    def test_整段复核候选可以跨两个首轮片段对齐(self):
        segments = [
            {"start": 0.0, "end": 1.0},
            {"start": 1.0, "end": 2.0},
        ]
        review = [{
            "start": 0.0,
            "end": 2.0,
            "text": "夏希先輩",
            "quality_score": 90,
        }]

        mapped = map_full_review_candidates(segments, review)

        self.assertEqual("夏希", mapped[0]["text"])
        self.assertEqual("先輩", mapped[1]["text"])

    def test_第三次候选与Large一致时形成共识(self):
        segments = [{
            "text": "夏木先輩",
            "primary_candidate": {"text": "夏木先輩"},
            "review_candidate": {"text": "夏希先輩"},
            "needs_ai_review": True,
            "needs_review": True,
        }]

        stats = apply_third_candidates(
            segments,
            {0: {"text": "夏希先輩", "quality_score": 80}},
        )

        self.assertEqual("夏希先輩", segments[0]["text"])
        self.assertFalse(segments[0]["needs_ai_review"])
        self.assertEqual(1, stats["resolved"])


if __name__ == "__main__":
    unittest.main()
