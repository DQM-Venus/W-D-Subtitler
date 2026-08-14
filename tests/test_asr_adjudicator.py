"""AI 候选裁决测试。"""

import json
import unittest

from wd_subtitler.asr_adjudicator import adjudicate_segments, build_adjudication_batches


def make_segment(primary, review, index):
    return {
        "start": float(index),
        "end": float(index + 1),
        "text": primary,
        "primary_candidate": {"text": primary, "quality_score": 60},
        "review_candidate": {"text": review, "quality_score": 65},
        "needs_ai_review": True,
        "needs_review": True,
    }


class ASRAdjudicatorTests(unittest.TestCase):
    def test_API异常不会提前修改候选状态(self):
        segments = [make_segment("夏希先輩", "夏木先生", 0)]

        with self.assertRaises(RuntimeError):
            adjudicate_segments(
                segments,
                lambda messages: (_ for _ in ()).throw(RuntimeError("网络失败")),
            )

        self.assertEqual("夏希先輩", segments[0]["text"])
        self.assertTrue(segments[0]["needs_ai_review"])

    def test_三种裁决均能应用(self):
        segments = [
            make_segment("候補一", "候補壱", 0),
            make_segment("候補二", "候補弐", 1),
            make_segment("夏木先輩", "夏希先生", 2),
        ]

        def api_call(messages):
            return "\n".join([
                json.dumps({"id": "S00001", "action": "primary", "reason": "首轮自然"}, ensure_ascii=False),
                json.dumps({"id": "S00002", "action": "review", "reason": "复核清晰"}, ensure_ascii=False),
                json.dumps({"id": "S00003", "action": "corrected", "corrected_text": "夏希先輩", "reason": "结合上下文纠正敬称"}, ensure_ascii=False),
            ])

        stats = adjudicate_segments(segments, api_call)

        self.assertEqual(3, stats["resolved"])
        self.assertEqual(1, stats["corrected"])
        self.assertEqual("候補一", segments[0]["text"])
        self.assertEqual("候補弐", segments[1]["text"])
        self.assertEqual("夏希先輩", segments[2]["text"])
        self.assertEqual("ai-corrected", segments[2]["asr_source"])

    def test_缺失项只重试一次(self):
        segments = [
            make_segment("候補一", "候補壱", 0),
            make_segment("候補二", "候補弐", 1),
        ]
        calls = []

        def api_call(messages):
            calls.append(messages)
            if len(calls) == 1:
                return '{"id":"S00001","action":"primary","reason":"确定"}'
            return '{"id":"S00002","action":"review","reason":"补充"}'

        stats = adjudicate_segments(segments, api_call)

        self.assertEqual(2, len(calls))
        self.assertEqual(2, stats["resolved"])

    def test_非法修正版会保留Kotoba并等待人工复核(self):
        segments = [make_segment("夏希先輩", "夏木先生", 0)]

        stats = adjudicate_segments(
            segments,
            lambda messages: '{"id":"S00001","action":"corrected","corrected_text":"hello","reason":"错误返回"}',
        )

        self.assertEqual(1, stats["failed"])
        self.assertEqual("夏希先輩", segments[0]["text"])
        self.assertTrue(segments[0]["needs_ai_review"])

    def test_批次同时受条数与字符数限制(self):
        segments = [make_segment("長い候補" * 20, "別候補" * 20, i) for i in range(25)]

        batches = build_adjudication_batches(segments, max_items=20, max_chars=1000)

        self.assertGreater(len(batches), 2)
        self.assertTrue(all(len(batch) <= 20 for batch in batches))

    def test_裁决请求包含声学风险和第三候选(self):
        segments = [make_segment("夏木先輩", "夏希先生", 0)]
        segments[0].update({
            "avg_logprob": -1.1,
            "no_speech_prob": 0.6,
            "review_reasons": ["识别概率偏低"],
            "third_candidate": {"text": "夏希先輩", "quality_score": 82},
        })

        item = build_adjudication_batches(segments)[0][0]

        self.assertEqual(-1.1, item["primary"]["avg_logprob"])
        self.assertEqual("夏希先輩", item["third"]["text"])
        self.assertEqual(1.0, item["duration_seconds"])


if __name__ == "__main__":
    unittest.main()
