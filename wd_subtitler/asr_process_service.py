"""隔离原生 ASR 推理的子进程客户端。"""

import multiprocessing
import threading
import time

from .asr_service import WhisperASRService
from .asr_worker import run_asr_worker
from .processing_control import ProcessingCancelled


class ASRProcessError(RuntimeError):
    """ASR 子进程返回了可捕获的 Python 异常。"""


class ASRWorkerCrashed(RuntimeError):
    """ASR 子进程在未返回结果时异常退出。"""

    def __init__(self, exit_code):
        self.exit_code = exit_code
        super().__init__(f"ASR 子进程异常退出，退出码：{exit_code}")


class ASRCancelled(ProcessingCancelled):
    """ASR 已按用户请求取消。"""


class ASRProcessService:
    """通过常驻子进程执行 ASR，并在原生崩溃后自动降级。"""

    def __init__(
        self,
        model_name=None,
        device=None,
        compute_type=None,
        worker_target=run_asr_worker,
    ):
        self.model_name = model_name or WhisperASRService.DEFAULT_MODEL
        self.device = device or WhisperASRService.DEFAULT_DEVICE
        self.compute_type = compute_type or WhisperASRService.DEFAULT_COMPUTE_TYPE
        self.model = None
        self.model_loaded = False

        self._context = multiprocessing.get_context("spawn")
        self._worker_target = worker_target
        self._process = None
        self._connection = None
        self._disable_word_timestamps = False
        self._force_cpu = False
        self._state_lock = threading.RLock()
        self._forced_termination = threading.Event()

    def _start_worker(self):
        with self._state_lock:
            if self._process is not None and self._process.is_alive():
                return
            self._cleanup_worker()
            parent_connection, child_connection = self._context.Pipe(duplex=True)
            process = self._context.Process(
                target=self._worker_target,
                args=(child_connection,),
                name="字幕工具-ASR子进程",
                daemon=True,
            )
            process.start()
            child_connection.close()
            self._connection = parent_connection
            self._process = process

    def _cleanup_worker(self):
        with self._state_lock:
            connection = self._connection
            process = self._process
            self._connection = None
            self._process = None
        if connection is not None:
            try:
                connection.close()
            except OSError:
                pass
        if process is not None:
            process.join(timeout=0.2)
            if process.is_alive():
                process.terminate()
                process.join(timeout=2)
        self.model = None
        self.model_loaded = False

    def _run_once(
        self,
        fpath,
        language,
        initial_prompt,
        beam_size,
        best_of,
        patience,
        temperature,
        condition_on_previous_text,
        hotwords,
        clip_timestamps,
        vad_filter,
        word_timestamps,
        progress_callback,
        cancel_check,
        device,
        compute_type,
    ):
        self._start_worker()
        connection = self._connection
        process = self._process
        if self._forced_termination.is_set() or connection is None or process is None:
            raise ASRCancelled("ASR 已取消")
        command = {
            "type": "transcribe",
            "fpath": fpath,
            "language": language,
            "initial_prompt": initial_prompt,
            "beam_size": beam_size,
            "best_of": best_of,
            "patience": patience,
            "temperature": temperature,
            "condition_on_previous_text": condition_on_previous_text,
            "hotwords": hotwords,
            "clip_timestamps": clip_timestamps,
            "vad_filter": vad_filter,
            "word_timestamps": word_timestamps,
            "config": {
                "model_name": self.model_name,
                "device": device,
                "compute_type": compute_type,
            },
        }
        connection.send(command)
        cancel_sent = False
        cancel_started_at = None

        while True:
            if cancel_check and cancel_check() and not cancel_sent:
                try:
                    connection.send({"type": "cancel"})
                    cancel_sent = True
                    cancel_started_at = time.monotonic()
                except (BrokenPipeError, EOFError, OSError):
                    self._cleanup_worker()
                    raise ASRCancelled("ASR 已取消")

            if (
                cancel_sent
                and cancel_started_at is not None
                and time.monotonic() - cancel_started_at >= 1.5
            ):
                self._cleanup_worker()
                raise ASRCancelled("ASR 已取消，推理子进程已强制停止")

            try:
                has_message = connection.poll(0.1)
            except (EOFError, BrokenPipeError, OSError):
                if self._forced_termination.is_set() or (cancel_check and cancel_check()):
                    raise ASRCancelled("ASR 已取消")
                process.join(timeout=0.2)
                raise ASRWorkerCrashed(process.exitcode)

            if has_message:
                try:
                    message = connection.recv()
                except (EOFError, BrokenPipeError, OSError):
                    message = None

                if message is None:
                    if self._forced_termination.is_set() or (cancel_check and cancel_check()):
                        raise ASRCancelled("ASR 已取消")
                    process.join(timeout=0.2)
                    raise ASRWorkerCrashed(process.exitcode)

                message_type = message.get("type")
                if message_type == "progress":
                    if progress_callback:
                        progress_callback(
                            message["message"],
                            progress_pct=message.get("progress_pct"),
                        )
                    continue
                if message_type == "error":
                    raise ASRProcessError(
                        f"{message['message']}\n\n子进程堆栈：\n{message['traceback']}"
                    )
                if message_type == "cancelled":
                    self._cleanup_worker()
                    raise ASRCancelled("ASR 已取消")
                if message_type == "result":
                    if cancel_sent:
                        self._cleanup_worker()
                        raise ASRCancelled("ASR 已取消")
                    self.model = True
                    self.model_loaded = True
                    return message["segments"]

            if not process.is_alive():
                process.join(timeout=0.2)
                if self._forced_termination.is_set() or (cancel_check and cancel_check()):
                    raise ASRCancelled("ASR 已取消")
                raise ASRWorkerCrashed(process.exitcode)

            time.sleep(0.02)

    def transcribe(
        self,
        fpath,
        language="ja",
        initial_prompt=None,
        beam_size=2,
        best_of=5,
        patience=1.0,
        temperature=0,
        condition_on_previous_text=False,
        hotwords=None,
        clip_timestamps="0",
        vad_filter=True,
        word_timestamps=True,
        progress_callback=None,
        cancel_check=None,
    ):
        """在隔离进程中识别，并在原生崩溃时逐级降级。"""
        self._forced_termination.clear()
        attempts = []
        preferred_word_timestamps = word_timestamps and not self._disable_word_timestamps
        preferred_device = "cpu" if self._force_cpu else self.device
        preferred_compute_type = "int8" if self._force_cpu else self.compute_type
        attempts.append((preferred_device, preferred_compute_type, preferred_word_timestamps))

        if preferred_word_timestamps:
            attempts.append((preferred_device, preferred_compute_type, False))
        if preferred_device == "cuda":
            attempts.append(("cpu", "int8", False))

        unique_attempts = list(dict.fromkeys(attempts))
        last_crash = None
        for attempt_index, (device, compute_type, use_word_timestamps) in enumerate(unique_attempts):
            if attempt_index > 0 and progress_callback:
                if device == "cpu":
                    progress_callback("⚠️ ASR 子进程再次崩溃，切换到 CPU int8 模式重试")
                else:
                    progress_callback("⚠️ 词级时间戳对齐导致子进程崩溃，关闭词级时间戳后重试")

            try:
                result = self._run_once(
                    fpath=fpath,
                    language=language,
                    initial_prompt=initial_prompt,
                    beam_size=beam_size,
                    best_of=best_of,
                    patience=patience,
                    temperature=temperature,
                    condition_on_previous_text=condition_on_previous_text,
                    hotwords=hotwords,
                    clip_timestamps=clip_timestamps,
                    vad_filter=vad_filter,
                    word_timestamps=use_word_timestamps,
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                    device=device,
                    compute_type=compute_type,
                )
                if not use_word_timestamps and device != "cpu":
                    self._disable_word_timestamps = True
                if device == "cpu":
                    self._disable_word_timestamps = True
                    self._force_cpu = True
                return result
            except ASRWorkerCrashed as exc:
                if self._forced_termination.is_set() or (cancel_check and cancel_check()):
                    raise ASRCancelled("ASR 已取消") from exc
                last_crash = exc
                self._cleanup_worker()

        raise last_crash or ASRWorkerCrashed(None)

    def close(self):
        """关闭常驻 ASR 子进程。"""
        with self._state_lock:
            connection = self._connection
            process = self._process
        if connection is not None and process is not None and process.is_alive():
            try:
                connection.send({"type": "shutdown"})
                process.join(timeout=2)
            except (BrokenPipeError, EOFError, OSError):
                pass
        self._cleanup_worker()

    def force_terminate(self):
        """线程安全地强制终止当前 ASR 子进程。"""
        self._forced_termination.set()
        self._cleanup_worker()
