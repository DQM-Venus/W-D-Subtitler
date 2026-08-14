"""GUI 安全关闭状态机测试。"""

import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from wd_subtitler.gui import SubtitleToolApp
from wd_subtitler.processing_control import ShutdownState


class FakeRoot:
    def __init__(self):
        self.destroyed = False

    def destroy(self):
        self.destroyed = True


class FakeThread:
    def is_alive(self):
        return True


class FakeASR:
    def __init__(self):
        self.forced = False

    def force_terminate(self):
        self.forced = True


class GUIShutdownTests(unittest.TestCase):
    def make_app(self):
        app = SubtitleToolApp.__new__(SubtitleToolApp)
        app.root = FakeRoot()
        app.cancel_event = threading.Event()
        app.shutdown_state = ShutdownState.RUNNING
        app._worker_thread = FakeThread()
        app._active_pipeline = None
        app.asr_service = FakeASR()
        app._closing = False
        app.save_config_to_file = lambda: None
        app._set_controls_enabled = lambda *_: None
        app.log = lambda *_: None
        return app

    def test_首次关闭只请求取消不会关闭ASR或销毁窗口(self):
        app = self.make_app()

        app.on_close()

        self.assertEqual(ShutdownState.CANCELLING, app.shutdown_state)
        self.assertTrue(app.cancel_event.is_set())
        self.assertFalse(app.asr_service.forced)
        self.assertFalse(app.root.destroyed)

    def test_再次关闭确认后强制终止并销毁窗口(self):
        app = self.make_app()
        app.shutdown_state = ShutdownState.CANCELLING

        with patch("wd_subtitler.gui.messagebox.askyesno", return_value=True):
            app.on_close()

        self.assertTrue(app.asr_service.forced)
        self.assertEqual(ShutdownState.CLOSED, app.shutdown_state)
        self.assertTrue(app.root.destroyed)

    def test_恢复弹窗三种选择分别继续重置和取消(self):
        app = self.make_app()
        plan = SimpleNamespace(
            stage=SimpleNamespace(value="ASR_COMPLETE"),
            updated_at=__import__("datetime").datetime.now().astimezone(),
        )
        task = type("Task", (), {"source_path": type("Path", (), {"name": "测试.wav"})(), "resume_plan": None})()
        app.trans_service = type("Translation", (), {"model": "模型"})()
        discarded = []
        app.checkpoint_store = type("Store", (), {
            "find_resume": lambda *_: plan,
            "discard": lambda _, item: discarded.append(item),
        })()
        with patch("wd_subtitler.gui.messagebox.askyesnocancel", return_value=True):
            self.assertTrue(app._resolve_resume_plans([task], object()))
            self.assertIs(plan, task.resume_plan)

        task.resume_plan = None
        with patch("wd_subtitler.gui.messagebox.askyesnocancel", return_value=False):
            self.assertTrue(app._resolve_resume_plans([task], object()))
            self.assertEqual([task], discarded)

        with patch("wd_subtitler.gui.messagebox.askyesnocancel", return_value=None):
            self.assertFalse(app._resolve_resume_plans([task], object()))

    def test_后台清理完成事件会关闭等待中的窗口(self):
        app = self.make_app()
        app.shutdown_state = ShutdownState.CANCELLING
        outcome = type("Outcome", (), {"cleanup_completed": True})()

        closed = app._handle_pipeline_finished(outcome)

        self.assertTrue(closed)
        self.assertTrue(app.root.destroyed)
        self.assertEqual(ShutdownState.CLOSED, app.shutdown_state)


if __name__ == "__main__":
    unittest.main()
