"""W-D Subtitler 应用入口。"""

import faulthandler
import multiprocessing
import sys
import threading
import time
import traceback

from wd_subtitler.logging_utils import format_log_message, format_phase_title
from wd_subtitler.runtime_paths import ensure_log_dir


class Tee:
    """同时向控制台和日志文件写入文本。"""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                pass

    def flush(self):
        for stream in self.streams:
            try:
                stream.flush()
            except Exception:
                pass


def _prepare_windows_console():
    """双击运行且没有控制台时，在 Windows 上分配控制台。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        if not ctypes.windll.kernel32.GetConsoleWindow():
            ctypes.windll.kernel32.AllocConsole()
            sys.stdout = open("CONOUT$", "w", encoding="utf-8")
            sys.stderr = open("CONOUT$", "w", encoding="utf-8")
    except Exception:
        pass


def _setup_logging():
    """初始化本次运行的控制台与文件日志。"""
    log_path = ensure_log_dir() / "app.log"
    log_file = log_path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = Tee(sys.__stdout__, log_file)
    sys.stderr = Tee(sys.__stderr__, log_file)
    faulthandler.enable(file=log_file, all_threads=True)

    print(format_log_message(format_phase_title("字幕工具箱启动"), "INFO"))
    print(format_log_message(f"Python：{sys.version}", "INFO"))
    print(format_log_message(f"可执行文件：{sys.executable}", "INFO"))
    print(format_log_message(f"日志文件：{log_path}", "INFO"))
    return log_file


def _thread_excepthook(args):
    """记录线程中未处理的异常。"""
    print(format_log_message(f"线程异常：{args.thread.name}", "ERROR"), flush=True)
    traceback.print_exception(args.exc_type, args.exc_value, args.exc_traceback, file=sys.stderr)


def main():
    """创建窗口并进入 Tk 主循环。"""
    multiprocessing.freeze_support()
    _prepare_windows_console()
    log_file = _setup_logging()
    threading.excepthook = _thread_excepthook
    try:
        import tkinter as tk

        from wd_subtitler.gui import SubtitleToolApp

        root = tk.Tk()
        SubtitleToolApp(root)
        print(format_log_message("GUI 初始化完成，进入主循环", "INFO"), flush=True)
        root.mainloop()
        print(format_log_message("主循环正常退出", "INFO"), flush=True)
    except Exception:
        print(format_log_message("程序发生致命错误", "ERROR"), flush=True)
        traceback.print_exc(file=sys.stderr)
        try:
            input("按回车键退出")
        except Exception:
            time.sleep(10)
    finally:
        log_file.close()


if __name__ == "__main__":
    main()
