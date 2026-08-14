"""流水线实际跳过已恢复 ASR 阶段的测试。"""

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from wd_subtitler.checkpoint_store import CheckpointStore
from wd_subtitler.processing_control import CheckpointStage
from wd_subtitler.processing_models import AudioTrackInfo, MediaTask, ProcessingOptions
from wd_subtitler.processing_pipeline import SubtitleProcessingPipeline


class RejectingASR:
    def __init__(self):
        self.compute_type = "float16"
        self.model_loaded = False
        self.model = None
        self.called = False

    def transcribe(self, *args, **kwargs):
        self.called = True
        raise AssertionError("恢复时不应重新执行 ASR")

    def close(self):
        pass

    def force_terminate(self):
        pass


class TranslationStub:
    model = "测试模型"


class PipelineResumeTests(unittest.TestCase):
    def test_恢复ASR断点后不重新调用模型(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "测试.wav"
            source.write_bytes(b"audio")
            task = MediaTask(
                source_path=source,
                tracks=(AudioTrackInfo(0, 0, "jpn", "pcm", 1),),
                selected_track_index=0,
                segments=[{"start": 0.0, "end": 1.0, "text": "恢复文本", "trans": ""}],
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
                large_v3_review=False,
                ai_asr_arbitration=False,
                timeline_refinement=False,
            )
            store = CheckpointStore(root / "checkpoints")
            store.save_stage(task, CheckpointStage.ASR_COMPLETE, options, "测试模型")
            task.resume_plan = store.find_resume(task, options, "测试模型")
            task.segments = []
            service = RejectingASR()
            pipeline = SubtitleProcessingPipeline(
                options=options,
                tasks=[task],
                asr_service=service,
                translation_service=TranslationStub(),
                cancel_event=threading.Event(),
                log_callback=lambda *_: None,
                progress_callback=lambda *_: None,
                checkpoint_store=store,
            )

            with patch("wd_subtitler.processing_pipeline.prepare_working_audio") as prepare:
                prepare.side_effect = lambda media_task, *_: setattr(
                    media_task, "working_audio_path", source
                ) or source
                outcome = pipeline.run()

            self.assertFalse(service.called)
            self.assertEqual(["恢复文本"], [item["text"] for item in task.segments])
            self.assertEqual(1, len(outcome.summary["success_files"]))


if __name__ == "__main__":
    unittest.main()
