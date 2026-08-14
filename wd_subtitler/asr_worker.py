"""ASR 子进程入口。

该模块只负责进程间通信和服务生命周期。将 CTranslate2 推理放在独立
进程中后，即使底层原生库发生访问冲突，也不会拖垮 Tkinter 主进程。
"""

import faulthandler
import traceback

from .asr_service import WhisperASRService
from .runtime_paths import ensure_log_dir


def run_asr_worker(connection):
    """接收 ASR 命令并在当前子进程中执行。"""
    service = None
    service_config = None
    crash_log_path = ensure_log_dir() / "asr_crash.log"
    crash_log = crash_log_path.open("a", encoding="utf-8", buffering=1)
    faulthandler.enable(file=crash_log, all_threads=True)

    try:
        while True:
            command = connection.recv()
            command_type = command.get("type")

            if command_type == "shutdown":
                return
            if command_type != "transcribe":
                continue

            config = command["config"]
            current_config = (
                config["model_name"],
                config["device"],
                config["compute_type"],
            )
            if service is None or current_config != service_config:
                service = WhisperASRService(
                    model_name=config["model_name"],
                    device=config["device"],
                    compute_type=config["compute_type"],
                )
                service_config = current_config

            cancelled = False

            def progress_callback(message, progress_pct=None):
                connection.send({
                    "type": "progress",
                    "message": message,
                    "progress_pct": progress_pct,
                })

            def cancel_check():
                nonlocal cancelled
                while connection.poll():
                    pending = connection.recv()
                    if pending.get("type") == "cancel":
                        cancelled = True
                return cancelled

            try:
                segments = service.transcribe(
                    command["fpath"],
                    language=command["language"],
                    initial_prompt=command.get("initial_prompt"),
                    beam_size=command["beam_size"],
                    best_of=command["best_of"],
                    patience=command["patience"],
                    temperature=command["temperature"],
                    condition_on_previous_text=command["condition_on_previous_text"],
                    hotwords=command.get("hotwords"),
                    clip_timestamps=command.get("clip_timestamps", "0"),
                    vad_filter=command["vad_filter"],
                    word_timestamps=command["word_timestamps"],
                    progress_callback=progress_callback,
                    cancel_check=cancel_check,
                )
                if cancelled:
                    connection.send({"type": "cancelled"})
                else:
                    connection.send({
                        "type": "result",
                        "segments": segments,
                        "device": service.device,
                        "compute_type": service.compute_type,
                    })
            except Exception as exc:
                connection.send({
                    "type": "error",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                })
    except (EOFError, BrokenPipeError, OSError):
        return
    finally:
        faulthandler.disable()
        crash_log.close()
        connection.close()
