"""翻译失败与取消时的部分字幕保存测试。"""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from wd_subtitler.checkpoint_store import CheckpointStore
from wd_subtitler.processing_models import AudioTrackInfo, MediaTask, ProcessingOptions
from wd_subtitler.processing_pipeline import (
    SubtitleProcessingPipeline,
    TranslationCancelled,
    TranslationPartialError,
)
from wd_subtitler.translation_service import APIErrorInfo, TranslationAPIError
from wd_subtitler.subtitle_writer import SubtitleSaveError


class FakeASR:
    def close(self):
        pass

    def force_terminate(self):
        pass


class FailingTranslationService:
    model = "测试模型"

    def translate_batch(self, *args, **kwargs):
        raise TranslationAPIError(APIErrorInfo("network_error", "网络失败", retryable=True))


class CancellingTranslationService:
    model = "测试模型"

    def __init__(self, event):
        self.event = event

    def translate_batch(self, batch, *args, **kwargs):
        self.event.set()
        return ["已完成译文" for _ in batch]


def options():
    return ProcessingOptions(
        trans_key="密钥",
        base_url="https://api.deepseek.com",
        asr_hotwords="",
        do_translate=True,
        use_context=False,
        export_fmt="srt",
        model_precision="float16",
        quality_mode="快速",
        large_v3_review=False,
        ai_asr_arbitration=False,
        timeline_refinement=False,
    )


class PartialTranslationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        source = root / "测试.wav"
        source.write_bytes("音频".encode("utf-8"))
        self.task = MediaTask(
            source_path=source,
            tracks=(AudioTrackInfo(0, 0, "jpn", "pcm", 1),),
            selected_track_index=0,
            segments=[
                {"start": 0.0, "end": 1.0, "text": "原文一", "trans": "译文一"},
                {"start": 1.1, "end": 2.0, "text": "原文二", "trans": ""},
            ],
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def make_pipeline(self, translation, event=None):
        return SubtitleProcessingPipeline(
            options=options(),
            tasks=[self.task],
            asr_service=FakeASR(),
            translation_service=translation,
            cancel_event=event or threading.Event(),
            log_callback=lambda *_: None,
            progress_callback=lambda *_: None,
            checkpoint_store=CheckpointStore(Path(self.temp_dir.name) / "checkpoints"),
        )

    def test_API失败会保存部分字幕(self):
        pipeline = self.make_pipeline(FailingTranslationService())

        with self.assertRaises(TranslationPartialError) as raised:
            pipeline._translate_and_save(self.task, "", "签名", True, start_index=1)

        result = raised.exception.partial_result
        self.assertTrue(result.path.exists())
        content = result.path.read_text(encoding="utf-8")
        self.assertIn("译文一", content)
        self.assertIn("[未翻译] 原文二", content)

    def test_批次结束后取消会保存部分字幕(self):
        event = threading.Event()
        pipeline = self.make_pipeline(CancellingTranslationService(event), event)

        with self.assertRaises(TranslationCancelled) as raised:
            pipeline._translate_and_save(self.task, "", "签名", True)

        self.assertTrue(raised.exception.partial_result.path.exists())

    def test_翻译和部分字幕保存错误会同时保留(self):
        pipeline = self.make_pipeline(FailingTranslationService())

        with patch(
            "wd_subtitler.processing_pipeline.save_subtitles",
            side_effect=SubtitleSaveError("磁盘已满"),
        ):
            with self.assertRaises(TranslationPartialError) as raised:
                pipeline._translate_and_save(self.task, "", "签名", True, start_index=1)

        self.assertIsInstance(raised.exception.translation_error, TranslationAPIError)
        self.assertIsInstance(raised.exception.save_error, SubtitleSaveError)
        self.assertIn("网络失败", str(raised.exception))
        self.assertIn("磁盘已满", str(raised.exception))


if __name__ == "__main__":
    unittest.main()
