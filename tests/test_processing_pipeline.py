"""后台处理流水线资源释放测试。"""

import threading
import unittest

from wd_subtitler.processing_pipeline import SubtitleProcessingPipeline


class FakeASRService:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class FakeWorkspace:
    def __init__(self):
        self.closed = False

    def close(self):
        self.closed = True


class ProcessingPipelineTests(unittest.TestCase):
    def test_关闭流水线会释放ASR和临时媒体(self):
        asr_service = FakeASRService()
        workspace = FakeWorkspace()
        pipeline = SubtitleProcessingPipeline(
            options=None,
            tasks=[],
            asr_service=asr_service,
            translation_service=None,
            cancel_event=threading.Event(),
            log_callback=lambda *_: None,
            progress_callback=lambda *_: None,
        )
        pipeline.workspace = workspace

        pipeline.close()

        self.assertTrue(asr_service.closed)
        self.assertTrue(workspace.closed)
        self.assertIsNone(pipeline.workspace)


if __name__ == "__main__":
    unittest.main()
