"""W-D Subtitler 图形界面。"""

import json
import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

from .asr_process_service import ASRProcessService
from .checkpoint_store import CheckpointStore
from .asr_prompt_policy import normalize_hotwords
from .logging_utils import format_log_message
from .media_service import MediaProcessingError, create_media_task
from .processing_models import MediaTask, ProcessingOptions
from .processing_pipeline import SubtitleProcessingPipeline
from .processing_control import ShutdownState
from .quality_modes import get_quality_mode_description
from .runtime_paths import APP_ICON, CONFIG_FILE
from .translation_service import DeepSeekService


class SubtitleToolApp:
    """字幕工具主窗口。"""

    def __init__(self, root):
        self.root = root
        self.root.title("W-D Subtitler")
        self.root.geometry("1060x850")
        if APP_ICON.exists():
            try:
                self.root.iconbitmap(str(APP_ICON))
            except tk.TclError:
                pass

        self.media_tasks = []
        self.asr_service = ASRProcessService()
        self.trans_service = DeepSeekService()
        self.cancel_event = threading.Event()
        self.ui_events = queue.Queue()
        self._closing = False
        self._worker_thread = None
        self._active_pipeline = None
        self.shutdown_state = ShutdownState.RUNNING
        self.checkpoint_store = CheckpointStore()
        self.checkpoint_store.cleanup_expired(days=7)
        self.config = {
            "trans_key": "",
            "base_url": "https://api.deepseek.com",
            "asr_hotwords": "",
            "do_translate": True,
            "use_context": True,
            "export_fmt": "lrc",
            "model_precision": "float16",
            "quality_mode": "快速",
            "large_v3_review": True,
            "ai_asr_arbitration": False,
            "timeline_refinement": True,
        }
        self.load_config()
        self.create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.after(50, self._drain_ui_events)

    def load_config(self):
        """加载本地配置文件。"""
        if not CONFIG_FILE.exists():
            return
        try:
            with CONFIG_FILE.open("r", encoding="utf-8") as config_file:
                loaded = json.load(config_file)
            for key in self.config:
                if key in loaded:
                    self.config[key] = loaded[key]
            if not self.config["asr_hotwords"] and loaded.get("context_prompt"):
                self.config["asr_hotwords"] = loaded["context_prompt"]
        except Exception as exc:
            print(format_log_message(f"配置文件加载失败：{exc}", "WARNING"), flush=True)

    def _collect_options(self):
        """在 Tk 主线程生成后台使用的不可变快照。"""
        return ProcessingOptions(
            trans_key=self.entry_trans_key.get().strip(),
            base_url=self.entry_base_url.get().strip(),
            asr_hotwords=normalize_hotwords(self.entry_asr_hotwords.get()),
            do_translate=self.var_do_trans.get(),
            use_context=self.var_use_context.get(),
            export_fmt=self.combo_export.get(),
            model_precision=self.combo_precision.get(),
            quality_mode=self.combo_quality.get(),
            large_v3_review=self.var_large_v3_review.get(),
            ai_asr_arbitration=self.var_ai_arbitration.get(),
            timeline_refinement=self.var_timeline_refinement.get(),
        )

    def save_config_to_file(self, options=None):
        """保存主线程已经读取到的界面配置。"""
        try:
            options = options or self._collect_options()
            current_conf = {
                "trans_key": options.trans_key,
                "base_url": options.base_url,
                "asr_hotwords": options.asr_hotwords,
                "do_translate": options.do_translate,
                "use_context": options.use_context,
                "export_fmt": options.export_fmt,
                "model_precision": options.model_precision,
                "quality_mode": options.quality_mode,
                "large_v3_review": options.large_v3_review,
                "ai_asr_arbitration": options.ai_asr_arbitration,
                "timeline_refinement": options.timeline_refinement,
            }
            CONFIG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with CONFIG_FILE.open("w", encoding="utf-8", newline="\n") as config_file:
                json.dump(current_conf, config_file, indent=4, ensure_ascii=False)
                config_file.write("\n")
        except Exception as exc:
            print(format_log_message(f"配置文件保存失败：{exc}", "WARNING"), flush=True)

    def on_close(self):
        """先等待后台安全清理；再次关闭时允许用户确认强退。"""
        try:
            self.save_config_to_file()
        except tk.TclError:
            pass
        if not self._worker_thread or not self._worker_thread.is_alive():
            self.shutdown_state = ShutdownState.CLOSED
            self._closing = True
            self.root.destroy()
            return
        if self.shutdown_state == ShutdownState.RUNNING:
            self.shutdown_state = ShutdownState.CANCELLING
            self.cancel_event.set()
            self._set_controls_enabled(False)
            self.log("正在等待当前请求完成并安全退出；再次关闭可选择强制退出。", "WARNING")
            return
        if self.shutdown_state in {ShutdownState.CANCELLING, ShutdownState.CLEANING}:
            if messagebox.askyesno(
                "确认强制退出",
                "后台任务尚未完成清理。强制退出可能失去尚未保存的结果，是否继续？",
            ):
                self.asr_service.force_terminate()
                if self._active_pipeline is not None:
                    self._active_pipeline.force_terminate()
                self.shutdown_state = ShutdownState.CLOSED
                self._closing = True
                self.root.destroy()

    def _set_controls_enabled(self, enabled):
        """统一切换可交互控件状态。"""
        state = "normal" if enabled else "disabled"
        for widget in self.root.winfo_children():
            self._set_widget_tree_state(widget, state)

    def _set_widget_tree_state(self, widget, state):
        try:
            if isinstance(widget, (ttk.Button, ttk.Checkbutton, ttk.Entry, ttk.Combobox)):
                widget.configure(state=state)
        except tk.TclError:
            pass
        for child in widget.winfo_children():
            self._set_widget_tree_state(child, state)

    def create_widgets(self):
        """创建全部界面控件。"""
        frame_top = ttk.LabelFrame(self.root, text="1. 媒体与音轨", padding=10)
        frame_top.pack(fill="x", padx=10, pady=5)
        button_row = ttk.Frame(frame_top)
        button_row.pack(fill="x")
        ttk.Button(button_row, text="添加视频/音频", command=self.add_files).pack(side="left", padx=5)
        ttk.Button(button_row, text="清空", command=self.clear_files).pack(side="left", padx=5)
        self.lbl_count = ttk.Label(button_row, text="0 个文件")
        self.lbl_count.pack(side="left", padx=15)
        ttk.Label(button_row, text="所选文件音轨：").pack(side="left", padx=(30, 5))
        self.combo_audio_track = ttk.Combobox(button_row, width=42, state="readonly")
        self.combo_audio_track.pack(side="left", fill="x", expand=True, padx=5)
        self.combo_audio_track.bind("<<ComboboxSelected>>", self._change_selected_track)

        self.media_tree = ttk.Treeview(
            frame_top,
            columns=("file", "track"),
            show="headings",
            height=4,
            selectmode="browse",
        )
        self.media_tree.heading("file", text="文件")
        self.media_tree.heading("track", text="识别音轨")
        self.media_tree.column("file", width=570, anchor="w")
        self.media_tree.column("track", width=390, anchor="w")
        self.media_tree.pack(fill="x", padx=5, pady=5)
        self.media_tree.bind("<<TreeviewSelect>>", self._update_track_selector)

        frame_asr = ttk.LabelFrame(self.root, text="2. 语音识别", padding=10)
        frame_asr.pack(fill="x", padx=10, pady=5)
        row_asr = ttk.Frame(frame_asr)
        row_asr.pack(fill="x", pady=2)
        ttk.Label(row_asr, text="模型精度：").pack(side="left", padx=5)
        precision_values = ["float16", "int8_float16", "int8"]
        self.combo_precision = ttk.Combobox(
            row_asr,
            values=precision_values,
            width=12,
            state="readonly",
        )
        precision = self.config.get("model_precision", "float16")
        self.combo_precision.set(precision if precision in precision_values else "float16")
        self.combo_precision.pack(side="left", padx=5)
        ttk.Label(row_asr, text="质量模式：").pack(side="left", padx=(20, 5))
        self.combo_quality = ttk.Combobox(
            row_asr,
            values=["快速", "高质量"],
            width=10,
            state="readonly",
        )
        self.combo_quality.set(self.config.get("quality_mode", "快速"))
        self.combo_quality.pack(side="left", padx=5)
        self.combo_quality.bind("<<ComboboxSelected>>", self.update_quality_description)
        ttk.Label(
            row_asr,
            text="float16 适合 GPU；int8 更省显存并适合 CPU",
            foreground="gray",
        ).pack(side="left", padx=15)
        self.lbl_quality_help = ttk.Label(
            frame_asr,
            foreground="#555555",
            wraplength=1000,
            justify="left",
        )
        self.lbl_quality_help.pack(fill="x", padx=5, pady=(4, 2))

        row_hotwords = ttk.Frame(frame_asr)
        row_hotwords.pack(fill="x", pady=4)
        ttk.Label(row_hotwords, text="日文热词：").pack(side="left", padx=5)
        self.entry_asr_hotwords = ttk.Entry(row_hotwords, width=55)
        self.entry_asr_hotwords.insert(0, self.config.get("asr_hotwords", ""))
        self.entry_asr_hotwords.pack(side="left", padx=5)
        ttk.Label(
            row_hotwords,
            text="只填写确认准确的人名或专有术语",
            foreground="gray",
        ).pack(side="left", padx=5)

        row_quality = ttk.Frame(frame_asr)
        row_quality.pack(fill="x", pady=2)
        self.var_large_v3_review = tk.BooleanVar(value=self.config.get("large_v3_review", True))
        ttk.Checkbutton(
            row_quality,
            text="使用 Whisper Large-v3 复核",
            variable=self.var_large_v3_review,
            command=self.update_quality_description,
        ).pack(side="left", padx=5)
        self.var_ai_arbitration = tk.BooleanVar(value=self.config.get("ai_asr_arbitration", False))
        ttk.Checkbutton(
            row_quality,
            text="AI 候选裁决",
            variable=self.var_ai_arbitration,
        ).pack(side="left", padx=15)
        self.var_timeline_refinement = tk.BooleanVar(value=self.config.get("timeline_refinement", True))
        ttk.Checkbutton(
            row_quality,
            text="时间轴精修",
            variable=self.var_timeline_refinement,
        ).pack(side="left", padx=15)
        ttk.Label(
            row_quality,
            text="AI 裁决会产生额外 API 请求；时间轴精修失败会自动保留原时间轴",
            foreground="gray",
        ).pack(side="left", padx=5)
        self.update_quality_description()

        frame_trans = ttk.LabelFrame(self.root, text="3. AI 翻译", padding=10)
        frame_trans.pack(fill="x", padx=10, pady=5)
        row_trans = ttk.Frame(frame_trans)
        row_trans.pack(fill="x", pady=2)
        self.var_do_trans = tk.BooleanVar(value=self.config["do_translate"])
        ttk.Checkbutton(row_trans, text="启用翻译", variable=self.var_do_trans).pack(side="left")
        ttk.Label(row_trans, text="API Key：").pack(side="left", padx=5)
        self.entry_trans_key = ttk.Entry(row_trans, width=40, show="*")
        self.entry_trans_key.insert(0, self.config["trans_key"])
        self.entry_trans_key.pack(side="left", padx=5)
        ttk.Label(row_trans, text="Base URL：").pack(side="left", padx=(15, 5))
        self.entry_base_url = ttk.Entry(row_trans, width=36)
        self.entry_base_url.insert(0, self.config["base_url"])
        self.entry_base_url.pack(side="left", padx=5)

        frame_act = ttk.LabelFrame(self.root, text="4. 执行", padding=10)
        frame_act.pack(fill="both", expand=True, padx=10, pady=5)
        row_export = ttk.Frame(frame_act)
        row_export.pack(fill="x", pady=2)
        ttk.Label(row_export, text="导出格式：").pack(side="left", padx=5)
        self.combo_export = ttk.Combobox(
            row_export,
            values=["lrc", "srt"],
            width=10,
            state="readonly",
        )
        self.combo_export.set(self.config.get("export_fmt", "lrc"))
        self.combo_export.pack(side="left", padx=5)
        self.var_use_context = tk.BooleanVar(value=self.config["use_context"])
        ttk.Checkbutton(
            row_export,
            text="启用术语/大纲分析",
            variable=self.var_use_context,
        ).pack(side="left", padx=20)

        row_buttons = ttk.Frame(frame_act)
        row_buttons.pack(fill="x", pady=5)
        self.btn_run = ttk.Button(row_buttons, text="开始处理", command=self.start_thread)
        self.btn_run.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.btn_cancel = ttk.Button(
            row_buttons,
            text="取消",
            command=self.cancel_process,
            state="disabled",
        )
        self.btn_cancel.pack(side="right", padx=(5, 0))
        self.progress = ttk.Progressbar(frame_act, mode="determinate")
        self.progress.pack(fill="x")
        self.log_area = scrolledtext.ScrolledText(frame_act, height=15)
        self.log_area.pack(fill="both", expand=True, pady=5)
        for tag, color in {
            "PHASE": "orange",
            "TRANS": "blue",
            "ERROR": "red",
            "WARNING": "#B36B00",
            "SUCCESS": "green",
            "CONTEXT": "purple",
            "ASR": "#006400",
            "SUMMARY": "#008B8B",
        }.items():
            self.log_area.tag_config(tag, foreground=color)

    def _queue_ui(self, event_type, *payload):
        self.ui_events.put((event_type, payload))

    def _drain_ui_events(self):
        """仅在 Tk 主线程消费后台事件。"""
        if self._closing:
            return
        try:
            while True:
                event_type, payload = self.ui_events.get_nowait()
                if event_type == "log":
                    self.log(*payload)
                elif event_type == "progress":
                    self.progress.configure(value=payload[0])
                elif event_type == "dialog":
                    level, title, message = payload
                    getattr(messagebox, f"show{level}")(title, message)
                elif event_type == "running":
                    running = payload[0]
                    self.btn_run.config(state="disabled" if running else "normal")
                    self.btn_cancel.config(state="normal" if running else "disabled")
                elif event_type == "pipeline_finished":
                    if self._handle_pipeline_finished(payload[0]):
                        return
        except queue.Empty:
            pass
        self.root.after(50, self._drain_ui_events)

    def _handle_pipeline_finished(self, outcome):
        """处理后台清理完成事件；返回窗口是否已经关闭。"""
        if self.shutdown_state == ShutdownState.CANCELLING:
            self.shutdown_state = ShutdownState.CLEANING
        if self.shutdown_state == ShutdownState.CLEANING and outcome.cleanup_completed:
            self.shutdown_state = ShutdownState.CLOSED
            self._closing = True
            self.root.destroy()
            return True
        return False

    def thread_safe_log(self, message, tag="INFO"):
        self._queue_ui("log", message, tag)

    def thread_safe_progress(self, value):
        self._queue_ui("progress", value)

    def log(self, message, tag="INFO"):
        formatted = format_log_message(message, tag)
        if tag == "PHASE" and self.log_area.index("end-1c") != "1.0":
            self.log_area.insert("end", "\n")
        self.log_area.insert("end", formatted + "\n", tag)
        self.log_area.see("end")
        if tag == "PHASE":
            print(flush=True)
        print(formatted, flush=True)

    def update_quality_description(self, event=None):
        mode = self.combo_quality.get() or "快速"
        self.lbl_quality_help.config(
            text=get_quality_mode_description(mode, self.var_large_v3_review.get())
        )

    def add_files(self):
        paths = filedialog.askopenfilenames(
            filetypes=[
                ("音视频", "*.mp4 *.mkv *.mp3 *.wav *.m4a *.flac *.ogg *.aac *.wma *.avi *.mov *.webm"),
                ("所有文件", "*.*"),
            ]
        )
        known = {str(task.source_path) for task in self.media_tasks}
        failures = []
        for path in paths:
            if path in known:
                continue
            try:
                task = create_media_task(path)
                self.media_tasks.append(task)
                known.add(path)
                self._insert_media_task(task)
            except MediaProcessingError as exc:
                failures.append(str(exc))
        self.lbl_count.config(text=f"{len(self.media_tasks)} 个文件")
        if failures:
            messagebox.showwarning("部分媒体无法添加", "\n".join(failures[:10]))

    def _insert_media_task(self, task):
        self.media_tree.insert(
            "",
            "end",
            iid=str(len(self.media_tasks) - 1),
            values=(task.source_path.name, task.selected_track.display_name),
        )

    def clear_files(self):
        self.media_tasks.clear()
        for item in self.media_tree.get_children():
            self.media_tree.delete(item)
        self.combo_audio_track.set("")
        self.combo_audio_track["values"] = ()
        self.lbl_count.config(text="0 个文件")

    def _selected_task(self):
        selection = self.media_tree.selection()
        if not selection:
            return None
        return self.media_tasks[int(selection[0])]

    def _update_track_selector(self, event=None):
        task = self._selected_task()
        if not task:
            return
        self.combo_audio_track["values"] = [track.display_name for track in task.tracks]
        self.combo_audio_track.set(task.selected_track.display_name)

    def _change_selected_track(self, event=None):
        task = self._selected_task()
        if not task:
            return
        selected_name = self.combo_audio_track.get()
        for track in task.tracks:
            if track.display_name == selected_name:
                task.selected_track_index = track.stream_index
                self.media_tree.set(self.media_tree.selection()[0], "track", track.display_name)
                break

    def cancel_process(self):
        if self.cancel_event.is_set():
            return
        self.cancel_event.set()
        self.log(
            "正在取消；ASR 最多等待 1.5 秒。如正在请求翻译 API，将等待当前请求结束。",
            "WARNING",
        )

    def _validate_start(self, options):
        if not self.media_tasks:
            return "请先添加音视频文件"
        if options.ai_asr_arbitration and not options.large_v3_review:
            return "AI 候选裁决依赖 Large-v3，请先启用 Large-v3 复核"
        if (options.do_translate or options.ai_asr_arbitration) and not options.trans_key:
            return "翻译或 AI 候选裁决需要 DeepSeek API Key"
        if (options.do_translate or options.ai_asr_arbitration) and not options.base_url:
            return "请输入 DeepSeek Base URL"
        return ""

    def start_thread(self):
        options = self._collect_options()
        error = self._validate_start(options)
        if error:
            messagebox.showerror("无法开始", error)
            return
        tasks = [
            MediaTask(
                source_path=task.source_path,
                tracks=task.tracks,
                selected_track_index=task.selected_track_index,
            )
            for task in self.media_tasks
        ]
        if not self._resolve_resume_plans(tasks, options):
            return
        self.save_config_to_file(options)
        self.cancel_event.clear()
        self.btn_run.config(state="disabled")
        self.btn_cancel.config(state="normal")
        self.progress.configure(value=0)
        self._worker_thread = threading.Thread(
            target=self._run_pipeline,
            args=(options, tasks),
            daemon=True,
            name="字幕处理后台线程",
        )
        self._worker_thread.start()

    def _resolve_resume_plans(self, tasks, options):
        """询问用户继续、重新处理或取消；返回是否启动任务。"""
        resume_plans = []
        for task in tasks:
            plan = self.checkpoint_store.find_resume(
                task, options, self.trans_service.model
            )
            if plan:
                resume_plans.append((task, plan))
        if resume_plans:
            details = "\n".join(
                f"• {task.source_path.name}：{plan.stage.value}，"
                f"更新时间 {plan.updated_at.astimezone().strftime('%Y-%m-%d %H:%M')}"
                for task, plan in resume_plans
            )
            choice = messagebox.askyesnocancel(
                "发现可恢复断点",
                f"发现以下未完成任务：\n{details}\n\n"
                "选择“是”继续处理，“否”删除断点并重新处理，“取消”不启动。",
            )
            if choice is None:
                return False
            if choice:
                for task, plan in resume_plans:
                    task.resume_plan = plan
            else:
                for task, _ in resume_plans:
                    self.checkpoint_store.discard(task)
        return True

    def _run_pipeline(self, options, tasks):
        """后台线程入口；只通过事件队列更新界面。"""
        pipeline = SubtitleProcessingPipeline(
            options=options,
            tasks=tasks,
            asr_service=self.asr_service,
            translation_service=self.trans_service,
            cancel_event=self.cancel_event,
            log_callback=self.thread_safe_log,
            progress_callback=self.thread_safe_progress,
            checkpoint_store=self.checkpoint_store,
        )
        self._active_pipeline = pipeline
        try:
            outcome = pipeline.run()
        finally:
            self._active_pipeline = None
        self.thread_safe_progress(0 if outcome.cancelled else 100)
        self._queue_ui("running", False)
        self._queue_ui("pipeline_finished", outcome)
        if outcome.cancelled:
            self.thread_safe_log("任务已取消，临时媒体已清理", "WARNING")
        elif outcome.fatal_error:
            self.thread_safe_log("任务因致命错误终止，请查看日志", "ERROR")
            self._queue_ui(
                "dialog",
                "error",
                "处理失败",
                f"处理发生致命错误：\n{outcome.fatal_error}",
            )
        elif outcome.summary["failed_files"]:
            level = "warning" if outcome.summary["success_files"] else "error"
            self._queue_ui("dialog", level, "任务结束", "部分文件处理失败，请查看日志。")
        else:
            self.thread_safe_log("全部任务处理完成", "SUCCESS")
            self._queue_ui("dialog", "info", "完成", "处理完毕！")
