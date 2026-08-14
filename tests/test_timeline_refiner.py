"""轻量时间轴精修测试。"""

import unittest

from wd_subtitler.timeline_refiner import refine_timeline, split_segments_at_speech_gaps


class TimelineRefinerTests(unittest.TestCase):
    def test_VAD静音点可以拆分无标点长句(self):
        segments = [{
            "start": 0.0,
            "end": 4.0,
            "text": "これはとても長い字幕なので途中の無音で自然に二つへ分割する必要があります",
        }]
        intervals = [
            {"start": 0.0, "end": 1.7},
            {"start": 2.3, "end": 4.0},
        ]

        count = split_segments_at_speech_gaps(segments, intervals, max_chars=20)

        self.assertEqual(1, count)
        self.assertEqual(2, len(segments))
        self.assertEqual(2.0, segments[0]["end"])
        self.assertEqual(2.0, segments[1]["start"])
    def test_语音边界会收紧且保持不重叠(self):
        segments = [
            {"start": 0.0, "end": 2.0, "text": "第一句"},
            {"start": 2.1, "end": 4.0, "text": "第二句"},
        ]
        speech = [
            {"start": 0.3, "end": 1.7},
            {"start": 2.35, "end": 3.7},
        ]

        stats = refine_timeline(segments, speech)

        self.assertEqual(2, stats.refined)
        self.assertEqual((0.3, 1.7), (segments[0]["start"], segments[0]["end"]))
        self.assertEqual((2.35, 3.7), (segments[1]["start"], segments[1]["end"]))
        self.assertLessEqual(segments[0]["end"] + 0.02, segments[1]["start"])

    def test_没有语音区间时保留原时间轴(self):
        segments = [{"start": 1.0, "end": 2.0, "text": "字幕"}]

        stats = refine_timeline(segments, [])

        self.assertEqual(0, stats.refined)
        self.assertEqual(1, stats.skipped)
        self.assertEqual((1.0, 2.0), (segments[0]["start"], segments[0]["end"]))

    def test_精修结果不得短于最低时长(self):
        segments = [{"start": 1.0, "end": 1.5, "text": "字幕"}]
        speech = [{"start": 1.2, "end": 1.3}]

        stats = refine_timeline(segments, speech)

        self.assertEqual(0, stats.refined)
        self.assertGreaterEqual(segments[0]["end"] - segments[0]["start"], 0.3)


if __name__ == "__main__":
    unittest.main()
