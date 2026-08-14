"""Kotoba 与 Large-v3 共用工作音频测试。"""

import threading
import unittest
from pathlib import Path

from wd_subtitler.processing_models import AudioTrackInfo, MediaTask, ProcessingOptions
from wd_subtitler.processing_pipeline import SubtitleProcessingPipeline


class RecordingASRService:
    def __init__(self, segments):
        self.paths = []
        self.segments = segments

    def transcribe(self, path, **kwargs):
        self.paths.append(path)
        return [dict(segment) for segment in self.segments]


class SharedWorkingAudioTests(unittest.TestCase):
    def test_Kotoba与Large_v3接收同一工作音频(self):
        primary = RecordingASRService([{
            "start": 0.0,
            "end": 1.0,
            "text": "夏木先輩",
            "quality_score": 50,
            "needs_review": True,
            "low_confidence": True,
        }])
        review = RecordingASRService([{
            "start": 0.0,
            "end": 1.0,
            "text": "夏希先輩",
            "quality_score": 90,
            "avg_logprob": -0.2,
            "no_speech_prob": 0.1,
            "compression_ratio": 1.0,
            "decoding_temperature": 0.0,
        }])
        task = MediaTask(
            source_path=Path("测试视频.mkv"),
            tracks=(AudioTrackInfo(0, 0, "jpn", "aac", 2),),
            selected_track_index=0,
            working_audio_path=Path("统一工作音频.wav"),
        )
        options = ProcessingOptions(
            trans_key="",
            base_url="",
            asr_hotwords="",
            do_translate=False,
            use_context=False,
            export_fmt="srt",
            model_precision="float16",
            quality_mode="快速",
            large_v3_review=True,
            ai_asr_arbitration=False,
            timeline_refinement=True,
        )

        pipeline = SubtitleProcessingPipeline(
            options=options,
            tasks=[task],
            asr_service=primary,
            translation_service=None,
            cancel_event=threading.Event(),
            log_callback=lambda *_: None,
            progress_callback=lambda *_: None,
        )
        pipeline._asr_file_weight = 50.0
        pipeline._run_primary_asr(task)
        task.segments[0]["review_requested"] = True
        pipeline._run_large_v3_review(task, review)

        expected = str(task.working_audio_path)
        self.assertEqual([expected], primary.paths)
        self.assertEqual([expected], review.paths)


if __name__ == "__main__":
    unittest.main()
