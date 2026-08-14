"""ASR 子进程隔离回归测试。"""

import os
import time
import unittest

from wd_subtitler.asr_process_service import ASRCancelled, ASRProcessError, ASRProcessService


def crash_on_word_timestamps_worker(connection):
    """模拟词级时间戳触发原生进程崩溃。"""
    command = connection.recv()
    if command["word_timestamps"]:
        os._exit(91)
    connection.send({
        "type": "result",
        "segments": [{
            "start": 0.0,
            "end": 1.0,
            "text": "测试字幕",
            "trans": "",
            "words": None,
            "low_confidence": False,
        }],
        "device": command["config"]["device"],
        "compute_type": command["config"]["compute_type"],
    })


def echo_decode_options_worker(connection):
    """把收到的高质量解码参数编码进测试结果。"""
    command = connection.recv()
    connection.send({
        "type": "result",
        "segments": [{
            "start": 0.0,
            "end": 1.0,
            "text": (
                f"{command['beam_size']}|{command['best_of']}|"
                f"{command['patience']}|{command['temperature']}|"
                f"{command['condition_on_previous_text']}|{command['hotwords']}|"
                f"{command['clip_timestamps']}"
            ),
            "trans": "",
            "words": None,
            "low_confidence": False,
        }],
        "device": command["config"]["device"],
        "compute_type": command["config"]["compute_type"],
    })


def hanging_worker(connection):
    """模拟底层推理挂起且无法消费取消消息。"""
    connection.recv()
    time.sleep(10)


class ASRProcessServiceTests(unittest.TestCase):
    def test_挂起推理会在限定时间内强制取消(self):
        started_at = time.monotonic()
        service = ASRProcessService(worker_target=hanging_worker)
        try:
            with self.assertRaises(ASRCancelled):
                service.transcribe(
                    "无需存在的测试文件.wav",
                    cancel_check=lambda: time.monotonic() - started_at > 0.1,
                )
        finally:
            service.close()

        self.assertLess(time.monotonic() - started_at, 2.5)

    def test_原生崩溃后关闭词级时间戳重试(self):
        messages = []
        service = ASRProcessService(worker_target=crash_on_word_timestamps_worker)
        try:
            result = service.transcribe(
                "无需存在的测试文件.wav",
                progress_callback=lambda message, progress_pct=None: messages.append(message),
            )
        finally:
            service.close()

        self.assertEqual("测试字幕", result[0]["text"])
        self.assertTrue(any("关闭词级时间戳" in message for message in messages))

    def test_普通子进程异常会返回主进程(self):
        service = ASRProcessService()
        try:
            with self.assertRaises(ASRProcessError) as raised:
                service.transcribe("不存在的测试音频.wav")
        finally:
            service.close()

        self.assertIn("文件不存在", str(raised.exception))

    def test_高质量解码参数会传入子进程(self):
        service = ASRProcessService(worker_target=echo_decode_options_worker)
        try:
            result = service.transcribe(
                "无需存在的测试文件.wav",
                beam_size=5,
                best_of=5,
                patience=1.2,
                temperature=(0.0, 0.2, 0.4),
                condition_on_previous_text=True,
                hotwords="夏希 先辈",
                clip_timestamps=[1.0, 2.0, 5.0, 6.0],
            )
        finally:
            service.close()

        self.assertEqual(
            "5|5|1.2|(0.0, 0.2, 0.4)|True|夏希 先辈|[1.0, 2.0, 5.0, 6.0]",
            result[0]["text"],
        )

    def test_强制终止会按取消处理而不是原生崩溃(self):
        service = ASRProcessService(worker_target=hanging_worker)
        errors = []

        def run_transcribe():
            try:
                service.transcribe("无需存在的测试文件.wav")
            except Exception as exc:
                errors.append(exc)

        import threading

        thread = threading.Thread(target=run_transcribe)
        thread.start()
        time.sleep(0.2)
        service.force_terminate()
        thread.join(timeout=2)

        self.assertFalse(thread.is_alive())
        self.assertEqual(1, len(errors))
        self.assertIsInstance(errors[0], ASRCancelled)


if __name__ == "__main__":
    unittest.main()
