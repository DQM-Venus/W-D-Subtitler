"""全流程断点存储与恢复判定测试。"""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from wd_subtitler.checkpoint_store import CheckpointStore, context_input_signature
from wd_subtitler.processing_control import CheckpointStage
from wd_subtitler.processing_models import AudioTrackInfo, MediaTask, ProcessingOptions


def make_options(**changes):
    values = {
        "trans_key": "不可写入断点的密钥",
        "base_url": "https://api.deepseek.com",
        "asr_hotwords": "夏希",
        "do_translate": True,
        "use_context": True,
        "export_fmt": "srt",
        "model_precision": "float16",
        "quality_mode": "快速",
        "large_v3_review": True,
        "ai_asr_arbitration": False,
        "timeline_refinement": True,
    }
    values.update(changes)
    return ProcessingOptions(**values)


class CheckpointStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.source = root / "测试.wav"
        self.source.write_bytes(b"a" * 2048)
        self.task = MediaTask(
            source_path=self.source,
            tracks=(AudioTrackInfo(0, 0, "jpn", "pcm", 1),),
            selected_track_index=0,
            segments=[{"start": 0.0, "end": 1.0, "text": "こんにちは", "trans": ""}],
        )
        self.store = CheckpointStore(root / "checkpoints")

    def tearDown(self):
        self.temp_dir.cleanup()

    def save_through_timeline(self, options):
        for stage in (
            CheckpointStage.ASR_COMPLETE,
            CheckpointStage.REVIEW_COMPLETE,
            CheckpointStage.ARBITRATION_COMPLETE,
            CheckpointStage.TIMELINE_COMPLETE,
        ):
            self.store.save_stage(self.task, stage, options, "模型")

    def test_阶段断点可恢复且不保存密钥(self):
        options = make_options()
        self.store.save_stage(self.task, CheckpointStage.ASR_COMPLETE, options, "模型")
        plan = self.store.find_resume(self.task, options, "模型")

        self.assertEqual(CheckpointStage.ASR_COMPLETE, plan.stage)
        raw = plan.checkpoint_path.read_text(encoding="utf-8")
        self.assertNotIn(options.trans_key, raw)

    def test_导出格式和密钥变化不影响恢复但热词变化会使识别失效(self):
        options = make_options()
        self.save_through_timeline(options)

        compatible = self.store.find_resume(
            self.task,
            make_options(export_fmt="lrc", trans_key="另一密钥"),
            "模型",
        )
        incompatible = self.store.find_resume(
            self.task, make_options(asr_hotwords="另一热词"), "模型"
        )

        self.assertEqual(CheckpointStage.TIMELINE_COMPLETE, compatible.stage)
        self.assertIsNone(incompatible)

    def test_AI关闭时修改BaseURL只使翻译断点失效(self):
        original = make_options(ai_asr_arbitration=False)
        self.save_through_timeline(original)

        plan = self.store.find_resume(
            self.task,
            make_options(ai_asr_arbitration=False, base_url="https://example.com"),
            "模型",
        )

        self.assertEqual(CheckpointStage.TIMELINE_COMPLETE, plan.stage)

    def test_时间轴选项变化会回退到裁决阶段(self):
        original = make_options(timeline_refinement=True)
        self.save_through_timeline(original)

        plan = self.store.find_resume(
            self.task, make_options(timeline_refinement=False), "模型"
        )

        self.assertEqual(CheckpointStage.ARBITRATION_COMPLETE, plan.stage)

    def test_AI启用状态变化会回退到复核阶段(self):
        original = make_options(ai_asr_arbitration=False)
        self.save_through_timeline(original)

        plan = self.store.find_resume(
            self.task, make_options(ai_asr_arbitration=True), "模型"
        )

        self.assertEqual(CheckpointStage.REVIEW_COMPLETE, plan.stage)

    def test_成功后删除断点(self):
        options = make_options()
        path = self.store.save_stage(
            self.task, CheckpointStage.ASR_COMPLETE, options, "模型"
        )

        self.store.delete_completed(self.task)

        self.assertFalse(path.exists())

    def test_翻译断点需要上下文签名一致(self):
        options = make_options()
        signature = context_input_signature([self.task])
        self.save_through_timeline(options)
        self.task.segments[0]["trans"] = "你好"
        self.store.save_translation_progress(
            self.task, 1, options, "模型", signature, "术语表", True
        )

        translated = self.store.find_resume(
            self.task, options, "模型", expected_context_signature=signature
        )
        timeline_only = self.store.find_resume(
            self.task, options, "模型", expected_context_signature="已变化"
        )

        self.assertEqual(CheckpointStage.TRANSLATING, translated.stage)
        self.assertEqual(1, translated.translated_count)
        self.assertEqual(CheckpointStage.TIMELINE_COMPLETE, timeline_only.stage)

    def test_源文件变化后拒绝恢复(self):
        options = make_options()
        self.store.save_stage(self.task, CheckpointStage.ASR_COMPLETE, options, "模型")
        self.source.write_bytes(b"b" * 2048)

        self.assertIsNone(self.store.find_resume(self.task, options, "模型"))

    def test_超过七天的断点会清理(self):
        options = make_options()
        path = self.store.save_stage(
            self.task, CheckpointStage.ASR_COMPLETE, options, "模型"
        )
        data = json.loads(path.read_text(encoding="utf-8"))
        data["updated_at"] = (
            datetime.now(timezone.utc) - timedelta(days=8)
        ).isoformat()
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

        self.assertEqual(1, self.store.cleanup_expired(days=7))
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
