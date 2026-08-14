"""字幕处理后台流水线。

本模块不依赖 Tkinter。后台线程只通过回调上报日志和进度，从而让 GUI
专注于界面状态与用户输入。
"""

import threading
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .asr_process_service import ASRCancelled, ASRProcessService
from .asr_prompt_policy import get_primary_prompt_options, get_review_prompt_options
from .asr_review import (
    LARGE_V3_MODEL,
    apply_review_candidates,
    apply_third_candidates,
    build_review_clips,
    flatten_clip_timestamps,
    map_full_review_candidates,
    map_review_candidates,
)
from .logging_utils import format_phase_title, infer_log_level
from .media_service import MediaWorkspace, prepare_working_audio
from .processing_models import MediaTask, ProcessingOptions
from .processing_control import CheckpointStage, ProcessingCancelled, STAGE_ORDER
from .checkpoint_store import CheckpointStore, context_input_signature
from .subtitle_writer import SubtitleSaveError, save_subtitles
from .timeline_refiner import (
    detect_speech_intervals,
    refine_timeline_from_audio,
    split_segments_at_speech_gaps,
)
from .translation_service import DeepSeekService, TranslationAPIError


LogCallback = Callable[[str, str], None]
ProgressCallback = Callable[[float], None]


@dataclass(frozen=True)
class ProcessingOutcome:
    """后台流水线的最终结果。"""

    summary: dict
    cancelled: bool = False
    fatal_error: str = ""
    partial_files: tuple[str, ...] = ()
    cleanup_completed: bool = False


class TranslationPartialError(RuntimeError):
    """翻译失败，并携带部分字幕保存结果。"""

    def __init__(self, translation_error, partial_result=None, save_error=None):
        self.translation_error = translation_error
        self.partial_result = partial_result
        self.save_error = save_error
        message = f"翻译失败：{translation_error}"
        if save_error:
            message += f"；部分字幕保存失败：{save_error}"
        super().__init__(message)


class TranslationCancelled(ProcessingCancelled):
    """翻译取消，并携带已经保存的部分结果。"""

    def __init__(self, partial_result=None, save_error=None):
        self.partial_result = partial_result
        self.save_error = save_error
        message = "翻译已取消"
        if save_error:
            message += f"；部分字幕保存失败：{save_error}"
        super().__init__(message)


def create_empty_summary(total_files: int) -> dict:
    """创建单次任务的统计容器。"""
    return {
        "total_files": total_files,
        "success_files": [],
        "failed_files": [],
        "partial_files": [],
        "total_asr_lines": 0,
        "total_translated_lines": 0,
        "total_low_conf": 0,
        "total_needs_review": 0,
        "large_reviewed": 0,
        "large_replaced": 0,
        "large_agreed": 0,
        "ai_requested": 0,
        "ai_resolved": 0,
        "ai_corrected": 0,
        "ai_failed": 0,
        "timeline_refined": 0,
        "timeline_skipped": 0,
        "timeline_fallback_files": 0,
        "output_files": [],
    }


def format_processing_summary(summary: dict, total_duration: int) -> str:
    """将统计数据格式化为规整的摘要文本。"""
    minutes, seconds = divmod(total_duration, 60)
    lines = [
        f"总耗时      {minutes} 分 {seconds} 秒",
        f"处理文件    {summary['total_files']} 个（成功 {len(summary['success_files'])}，失败 {len(summary['failed_files'])}）",
        f"识别字幕    {summary['total_asr_lines']} 行",
    ]
    if summary["total_translated_lines"]:
        lines.append(f"完成翻译    {summary['total_translated_lines']} 行")
    if summary["large_reviewed"]:
        lines.append(
            f"Large-v3    {summary['large_reviewed']} 行（一致 {summary['large_agreed']}，替换 {summary['large_replaced']}）"
        )
    if summary["ai_requested"]:
        lines.append(
            f"AI 裁决     {summary['ai_requested']} 行（解决 {summary['ai_resolved']}，"
            f"修正 {summary['ai_corrected']}，失败 {summary['ai_failed']}）"
        )
    if summary["timeline_refined"] or summary["timeline_skipped"]:
        lines.append(
            f"时间轴精修  调整 {summary['timeline_refined']} 行，"
            f"保留 {summary['timeline_skipped']} 行，回退 {summary['timeline_fallback_files']} 个文件"
        )
    if summary["total_needs_review"]:
        lines.append(f"仍需复核    {summary['total_needs_review']} 行")
    if summary["failed_files"]:
        lines.append(f"失败文件    {', '.join(summary['failed_files'])}")
    if summary["partial_files"]:
        lines.append(f"部分结果    {', '.join(Path(path).name for path in summary['partial_files'])}")
    if summary["output_files"]:
        lines.append("输出文件")
        lines.extend(f"  - {Path(path).name}" for path in summary["output_files"])
    return "\n".join(lines)


class SubtitleProcessingPipeline:
    """执行媒体预处理、识别、复核、翻译和保存。"""

    def __init__(
        self,
        options: ProcessingOptions,
        tasks: list[MediaTask],
        asr_service: ASRProcessService,
        translation_service: DeepSeekService,
        cancel_event: threading.Event,
        log_callback: LogCallback,
        progress_callback: ProgressCallback,
        checkpoint_store: CheckpointStore | None = None,
    ):
        self.options = options
        self.tasks = tasks
        self.asr_service = asr_service
        self.translation_service = translation_service
        self.cancel_event = cancel_event
        self.log = log_callback
        self.report_progress = progress_callback
        self.workspace = None
        self.checkpoint_store = checkpoint_store or CheckpointStore()
        self._active_asr_service = asr_service
        self._asr_file_base_progress = 0.0
        self._asr_file_weight = 0.0
        self._valid_tasks: list[MediaTask] = []

    def phase(self, title: str) -> None:
        self.log(format_phase_title(title), "PHASE")

    def _asr_progress(self, message, progress_pct=None):
        self.log(message, infer_log_level(message, "ASR"))
        if progress_pct is not None:
            value = self._asr_file_base_progress + progress_pct / 100.0 * self._asr_file_weight
            self.report_progress(value)

    def _run_primary_asr(self, task: MediaTask) -> None:
        name = task.source_path.name
        hotwords_note = (
            f"；日文热词：{self.options.asr_hotwords[:30]}"
            if self.options.asr_hotwords else ""
        )
        self.log(f"开始识别：{name}{hotwords_note}", "ASR")
        decode_options = (
            {
                "beam_size": 5,
                "best_of": 5,
                "patience": 1.2,
                "temperature": (0.0, 0.2, 0.4),
                "condition_on_previous_text": False,
            }
            if self.options.quality_mode == "高质量"
            else {
                "beam_size": 2,
                "best_of": 5,
                "patience": 1.0,
                "temperature": 0,
                "condition_on_previous_text": False,
            }
        )
        started_at = time.time()
        task.segments = self.asr_service.transcribe(
            str(task.working_audio_path),
            language="ja",
            **decode_options,
            **get_primary_prompt_options(self.options.asr_hotwords),
            word_timestamps=False,
            progress_callback=self._asr_progress,
            cancel_check=self.cancel_event.is_set,
        )
        review_count = sum(1 for segment in task.segments if segment.get("needs_review"))
        self.log(
            f"识别完成：{name}（{len(task.segments)} 行，耗时 {int(time.time() - started_at)} 秒；"
            f"待复核 {review_count} 行）",
            "SUCCESS",
        )

    def _run_large_v3_review(self, task: MediaTask, review_service) -> dict:
        full_audio = self.options.quality_mode == "高质量"
        clips = [] if full_audio else build_review_clips(task.segments)
        if not full_audio and not clips:
            return {"reviewed": 0, "replaced": 0, "agreed": 0, "needs_ai": 0, "missing": 0}
        self.log(
            f"开始 Large-v3 {'独立全音频识别' if full_audio else '局部复核'}："
            f"{task.source_path.name}"
            + ("" if full_audio else f"（{len(clips)} 个片段）"),
            "ASR",
        )
        review_segments = review_service.transcribe(
            str(task.working_audio_path),
            language="ja",
            **get_review_prompt_options(),
            beam_size=5,
            best_of=5,
            patience=1.2,
            temperature=(0.0, 0.2, 0.4),
            condition_on_previous_text=False,
            clip_timestamps="0" if full_audio else flatten_clip_timestamps(clips),
            word_timestamps=False,
            progress_callback=self._asr_progress,
            cancel_check=self.cancel_event.is_set,
        )
        mapped = (
            map_full_review_candidates(task.segments, review_segments)
            if full_audio
            else map_review_candidates(clips, review_segments)
        )
        stats = apply_review_candidates(task.segments, mapped)
        self.log(
            "Large-v3 复核完成："
            f"一致 {stats['agreed']}；替换 {stats['replaced']}；"
            f"待 AI {stats['needs_ai']}；无候选 {stats['missing']}",
            "SUCCESS",
        )
        return stats

    def _run_third_asr(self, task: MediaTask, review_service) -> dict:
        clips = build_review_clips(task.segments, padding=0.8, selection_key="needs_ai_review")
        if not clips:
            return {"reviewed": 0, "resolved": 0, "selected_review": 0, "unresolved": 0}
        self.log(
            f"开始第三次局部识别：{task.source_path.name}（{len(clips)} 个分歧片段）",
            "ASR",
        )
        third_segments = review_service.transcribe(
            str(task.working_audio_path),
            language="ja",
            **get_review_prompt_options(),
            beam_size=5,
            best_of=5,
            patience=1.0,
            temperature=(0.2, 0.4, 0.6),
            condition_on_previous_text=False,
            clip_timestamps=flatten_clip_timestamps(clips),
            word_timestamps=False,
            progress_callback=self._asr_progress,
            cancel_check=self.cancel_event.is_set,
        )
        stats = apply_third_candidates(
            task.segments,
            map_review_candidates(clips, third_segments),
        )
        self.log(
            f"第三次识别完成：形成共识 {stats['resolved']}；"
            f"仍待 AI {stats['unresolved']}",
            "SUCCESS" if not stats["unresolved"] else "WARNING",
        )
        return stats

    def _translate_and_save(
        self,
        task: MediaTask,
        context_summary: str,
        context_signature: str,
        context_ready: bool,
        start_index: int = 0,
    ):
        segments = task.segments
        index = max(0, min(start_index, len(segments)))
        previous = [
            {"original": segment.get("text", ""), "translated": segment.get("trans", "")}
            for segment in segments[max(0, index - 5):index]
            if segment.get("trans") and segment.get("trans") != "(翻译失败)"
        ]
        batch_number = 0
        while index < len(segments) and not self.cancel_event.is_set():
            batch_end = index
            characters = 0
            while batch_end < len(segments):
                text = segments[batch_end]["text"]
                if characters + len(text) > 1500 and batch_end > index:
                    break
                characters += len(text)
                batch_end += 1
            batch = segments[index:batch_end]
            next_preview = [
                segments[next_index]["text"]
                for next_index in range(batch_end, min(batch_end + 3, len(segments)))
            ]
            try:
                translations = self.translation_service.translate_batch(
                    batch,
                    self.options.trans_key,
                    self.options.base_url,
                    context_summary,
                    prev_context=previous[-5:] or None,
                    next_preview=next_preview,
                    cancel_check=self.cancel_event.is_set,
                )
            except ProcessingCancelled:
                try:
                    partial_result = save_subtitles(
                        task.output_base, segments, self.options.export_fmt,
                        with_translation=True, partial=True,
                    )
                except SubtitleSaveError as save_error:
                    raise TranslationCancelled(save_error=save_error) from save_error
                raise TranslationCancelled(partial_result=partial_result)
            except TranslationAPIError as translation_error:
                try:
                    partial_result = save_subtitles(
                        task.output_base, segments, self.options.export_fmt,
                        with_translation=True, partial=True,
                    )
                except SubtitleSaveError as save_error:
                    raise TranslationPartialError(
                        translation_error, save_error=save_error
                    ) from translation_error
                raise TranslationPartialError(
                    translation_error, partial_result=partial_result
                ) from translation_error
            for offset, translation in enumerate(translations):
                final_translation = translation or "(翻译失败)"
                segments[index + offset]["trans"] = final_translation
                previous.append({
                    "original": batch[offset]["text"],
                    "translated": "" if final_translation == "(翻译失败)" else final_translation,
                })
            previous = previous[-10:]
            index = batch_end
            batch_number += 1
            self._save_translation_checkpoint(
                task, index, context_signature, context_summary, context_ready
            )
            self.log(
                f"翻译进度：{task.source_path.name}；{index}/{len(segments)} 行；第 {batch_number} 批",
                "TRANS",
            )

        partial = self.cancel_event.is_set()
        result = save_subtitles(
            task.output_base,
            segments,
            self.options.export_fmt,
            with_translation=True,
            partial=partial,
        )
        self.log(
            f"{'部分翻译已保存' if partial else '翻译并保存完成'}：{result.path.name}",
            "WARNING" if partial else "SUCCESS",
        )
        if partial:
            raise TranslationCancelled(partial_result=result)
        return result

    def run(self) -> ProcessingOutcome:
        """执行完整流水线，并确保释放子进程和临时媒体。"""
        started_at = time.time()
        summary = create_empty_summary(len(self.tasks))
        fatal_error = ""
        try:
            self._run_internal(summary)
        except ProcessingCancelled as exc:
            self.cancel_event.set()
            self._save_cancelled_partials(summary)
            self.log(str(exc), "WARNING")
        except Exception as exc:
            fatal_error = str(exc)
            self.log(f"处理发生致命错误：{exc}", "ERROR")
            self.log(traceback.format_exc(), "ERROR")
        finally:
            self.close()
        self.phase("处理摘要")
        self.log(format_processing_summary(summary, int(time.time() - started_at)), "SUMMARY")
        return ProcessingOutcome(
            summary=summary,
            cancelled=self.cancel_event.is_set(),
            fatal_error=fatal_error,
            partial_files=tuple(summary["partial_files"]),
            cleanup_completed=True,
        )

    def close(self) -> None:
        """释放 ASR 子进程和本次任务的临时媒体。"""
        self.asr_service.close()
        if self.workspace is not None:
            self.workspace.close()
            self.workspace = None

    def force_terminate(self) -> None:
        """强制终止当前正在工作的 ASR 服务。"""
        self.cancel_event.set()
        self._active_asr_service.force_terminate()

    def _save_cancelled_partials(self, summary: dict) -> None:
        """取消发生在任意后期阶段时，尽量保留已有字幕文本。"""
        if not self.options.do_translate:
            return
        existing_sources = {
            Path(path).name.split(".partial", 1)[0]
            for path in summary["partial_files"]
        }
        for task in self._valid_tasks:
            if not task.segments or task.output_base.name in existing_sources:
                continue
            try:
                result = save_subtitles(
                    task.output_base,
                    task.segments,
                    self.options.export_fmt,
                    with_translation=True,
                    partial=True,
                )
                summary["partial_files"].append(str(result.path))
                summary["output_files"].append(str(result.path))
                if task.source_path.name not in summary["failed_files"]:
                    summary["failed_files"].append(task.source_path.name)
                self.log(f"取消后的部分结果已保存：{result.path.name}", "WARNING")
            except SubtitleSaveError as exc:
                self.log(f"取消后无法保存部分字幕：{task.source_path.name}；{exc}", "ERROR")

    @staticmethod
    def _has_stage(task: MediaTask, stage: CheckpointStage) -> bool:
        return bool(
            task.resume_plan
            and STAGE_ORDER[task.resume_plan.stage] >= STAGE_ORDER[stage]
        )

    def _save_stage(self, task: MediaTask, stage: CheckpointStage) -> None:
        try:
            self.checkpoint_store.save_stage(
                task, stage, self.options, self.translation_service.model
            )
        except OSError as exc:
            self.log(f"断点保存失败：{task.source_path.name}；{exc}", "WARNING")

    def _save_translation_checkpoint(
        self,
        task: MediaTask,
        translated_count: int,
        context_signature: str,
        context_summary: str,
        context_ready: bool,
    ) -> None:
        """断点写入失败不应破坏已完成的翻译批次。"""
        try:
            self.checkpoint_store.save_translation_progress(
                task,
                translated_count,
                self.options,
                self.translation_service.model,
                context_signature,
                context_summary,
                context_ready,
            )
        except OSError as exc:
            self.log(
                f"翻译断点保存失败：{task.source_path.name}；{exc}", "WARNING"
            )

    def _run_internal(self, summary: dict) -> None:
        self.asr_service.compute_type = self.options.model_precision
        self.asr_service.model_loaded = False
        self.asr_service.model = None
        self.workspace = MediaWorkspace()

        self.phase("阶段一 · 媒体预处理与语音识别")
        valid_tasks = []
        self._asr_file_weight = 50.0 / max(len(self.tasks), 1)
        for index, task in enumerate(self.tasks):
            if self.cancel_event.is_set():
                break
            try:
                self.log(
                    f"正在预处理：{task.source_path.name}；{task.selected_track.display_name}",
                    "ASR",
                )
                prepare_working_audio(task, self.workspace, self.cancel_event.is_set)
                self._asr_file_base_progress = index * self._asr_file_weight
                if self._has_stage(task, CheckpointStage.ASR_COMPLETE):
                    task.segments = [dict(segment) for segment in task.resume_plan.segments]
                    self.log(f"已从断点恢复识别结果：{task.source_path.name}", "SUCCESS")
                else:
                    self._run_primary_asr(task)
                    self._save_stage(task, CheckpointStage.ASR_COMPLETE)
                valid_tasks.append(task)
                self._valid_tasks = valid_tasks
                summary["total_asr_lines"] += len(task.segments)
            except InterruptedError:
                self.cancel_event.set()
                break
            except ASRCancelled:
                self.cancel_event.set()
                raise
            except Exception as exc:
                summary["failed_files"].append(task.source_path.name)
                self.log(f"识别失败：{task.source_path.name}；{exc}", "ERROR")
            self.report_progress((index + 1) * self._asr_file_weight)

        if self.cancel_event.is_set() or not valid_tasks:
            return
        valid_tasks.sort(key=lambda task: task.source_path.name)
        review_all = self.options.large_v3_review and self.options.quality_mode == "高质量"
        for task in valid_tasks:
            for segment in task.segments:
                segment["review_requested"] = review_all or segment.get("needs_review", False)

        review_pending = [
            task for task in valid_tasks
            if not self._has_stage(task, CheckpointStage.REVIEW_COMPLETE)
        ]
        review_failed = set()
        if self.options.large_v3_review and review_pending:
            total_targets = sum(
                1 for task in review_pending for segment in task.segments
                if segment.get("review_requested")
            )
            if total_targets:
                self.phase("阶段二 · Whisper Large-v3 复核")
                self.asr_service.close()
                review_service = ASRProcessService(
                    model_name=LARGE_V3_MODEL,
                    compute_type=self.options.model_precision,
                )
                self._active_asr_service = review_service
                try:
                    for task in review_pending:
                        if self.cancel_event.is_set():
                            break
                        try:
                            stats = self._run_large_v3_review(task, review_service)
                            summary["large_reviewed"] += stats["reviewed"]
                            summary["large_replaced"] += stats["replaced"]
                            summary["large_agreed"] += stats["agreed"]
                            if any(
                                segment.get("needs_ai_review")
                                for segment in task.segments
                            ):
                                third_stats = self._run_third_asr(task, review_service)
                                summary["large_replaced"] += third_stats["selected_review"]
                        except ASRCancelled:
                            self.cancel_event.set()
                            raise
                        except Exception as exc:
                            review_failed.add(str(task.source_path.resolve()))
                            self.log(
                                f"Large-v3 复核失败：{task.source_path.name}；保留 Kotoba；{exc}",
                                "WARNING",
                            )
                finally:
                    review_service.close()
                    self._active_asr_service = self.asr_service
        for task in valid_tasks:
            if self.cancel_event.is_set():
                break
            if self._has_stage(task, CheckpointStage.REVIEW_COMPLETE):
                task.segments = [dict(segment) for segment in task.resume_plan.segments]
            elif str(task.source_path.resolve()) not in review_failed:
                self._save_stage(task, CheckpointStage.REVIEW_COMPLETE)

        arbitration_failed = set()
        if self.options.ai_asr_arbitration and not self.cancel_event.is_set():
            self.phase("阶段三 · AI 候选裁决")
            for task in valid_tasks:
                if self.cancel_event.is_set():
                    break
                if self._has_stage(task, CheckpointStage.ARBITRATION_COMPLETE):
                    task.segments = [dict(segment) for segment in task.resume_plan.segments]
                    continue
                if not any(segment.get("needs_ai_review") for segment in task.segments):
                    continue
                try:
                    stats = self.translation_service.adjudicate_asr(
                        task.segments,
                        self.options.trans_key,
                        self.options.base_url,
                        cancel_check=self.cancel_event.is_set,
                    )
                    for key in ("requested", "resolved", "corrected", "failed"):
                        summary[f"ai_{key}"] += stats[key]
                    self.log(
                        f"AI 裁决完成：{task.source_path.name}；"
                        f"解决 {stats['resolved']}；失败 {stats['failed']}",
                        "SUCCESS" if not stats["failed"] else "WARNING",
                    )
                except TranslationAPIError as exc:
                    arbitration_failed.add(str(task.source_path.resolve()))
                    failed = sum(1 for segment in task.segments if segment.get("needs_ai_review"))
                    summary["ai_requested"] += failed
                    summary["ai_failed"] += failed
                    self.log(f"AI 裁决失败 [{exc.info.code}]：{exc}；保留 Kotoba", "WARNING")

        for task in valid_tasks:
            if self.cancel_event.is_set():
                break
            if self._has_stage(task, CheckpointStage.ARBITRATION_COMPLETE):
                task.segments = [dict(segment) for segment in task.resume_plan.segments]
            elif str(task.source_path.resolve()) not in arbitration_failed:
                self._save_stage(task, CheckpointStage.ARBITRATION_COMPLETE)

        if self.options.timeline_refinement and not self.cancel_event.is_set():
            self.phase("阶段四 · 时间轴精修")
            for task in valid_tasks:
                if self._has_stage(task, CheckpointStage.TIMELINE_COMPLETE):
                    task.segments = [dict(segment) for segment in task.resume_plan.segments]
                    continue
                try:
                    task.speech_intervals = detect_speech_intervals(task.working_audio_path)
                except Exception:
                    task.speech_intervals = []
                split_count = split_segments_at_speech_gaps(
                    task.segments, task.speech_intervals
                )
                if split_count:
                    self.log(
                        f"VAD 辅助断句：{task.source_path.name}；新增 {split_count} 个切分点",
                        "SUCCESS",
                    )
                stats = refine_timeline_from_audio(
                    task.segments,
                    task.working_audio_path,
                    speech_intervals=task.speech_intervals,
                )
                summary["timeline_refined"] += stats.refined
                summary["timeline_skipped"] += stats.skipped
                summary["timeline_fallback_files"] += int(stats.fallback)
                self.log(
                    f"时间轴精修：{task.source_path.name}；调整 {stats.refined}；"
                    f"保留 {stats.skipped}" + ("；已整体回退" if stats.fallback else ""),
                    "WARNING" if stats.fallback else "SUCCESS",
                )

        for task in valid_tasks:
            if self.cancel_event.is_set():
                break
            if self._has_stage(task, CheckpointStage.TIMELINE_COMPLETE):
                task.segments = [dict(segment) for segment in task.resume_plan.segments]
            else:
                self._save_stage(task, CheckpointStage.TIMELINE_COMPLETE)

        summary["total_low_conf"] = sum(
            1 for task in valid_tasks for segment in task.segments if segment.get("low_confidence")
        )
        summary["total_needs_review"] = sum(
            1 for task in valid_tasks for segment in task.segments
            if segment.get("needs_review") or segment.get("needs_ai_review")
        )
        if self.cancel_event.is_set():
            return

        context_signature = context_input_signature(valid_tasks)
        for task in valid_tasks:
            upgraded = self.checkpoint_store.find_resume(
                task,
                self.options,
                self.translation_service.model,
                expected_context_signature=context_signature,
            )
            if upgraded and upgraded.stage == CheckpointStage.TRANSLATING:
                task.resume_plan = upgraded
                task.segments = [dict(segment) for segment in upgraded.segments]

        context_summary = ""
        context_ready = not (self.options.do_translate and self.options.use_context)
        resumed_contexts = {
            task.resume_plan.context_summary
            for task in valid_tasks
            if task.resume_plan
            and task.resume_plan.stage == CheckpointStage.TRANSLATING
            and task.resume_plan.context_input_signature == context_signature
            and task.resume_plan.context_summary
        }
        if len(resumed_contexts) == 1:
            context_summary = resumed_contexts.pop()
            context_ready = True
            self.log("已从断点恢复翻译参考信息", "SUCCESS")
        if self.options.do_translate and self.options.use_context:
            if context_summary:
                pass
            elif len(valid_tasks) > 1:
                self.phase("阶段五 · 全局翻译总纲分析")
                combined = "\n".join(
                    f"--- {task.source_path.name} ---\n"
                    + "\n".join(segment["text"] for segment in task.segments)
                    for task in valid_tasks
                )
                try:
                    context_summary = self.translation_service.analyze_full_text(
                        combined, self.options.trans_key, self.options.base_url,
                        cancel_check=self.cancel_event.is_set,
                    )
                    context_ready = True
                except TranslationAPIError as exc:
                    self.log(f"总纲生成失败 [{exc.info.code}]：{exc}", "WARNING")
            else:
                self.phase("阶段五 · 术语提取")
                text = "\n".join(segment["text"] for segment in valid_tasks[0].segments)
                try:
                    context_summary = self.translation_service.extract_terms(
                        text, self.options.trans_key, self.options.base_url,
                        cancel_check=self.cancel_event.is_set,
                    )
                    context_ready = True
                except TranslationAPIError as exc:
                    self.log(f"术语提取失败 [{exc.info.code}]：{exc}", "WARNING")

        self.phase("最终阶段 · 翻译与字幕保存" if self.options.do_translate else "最终阶段 · 字幕保存")
        for index, task in enumerate(valid_tasks):
            if self.cancel_event.is_set():
                break
            try:
                if self.options.do_translate:
                    self.log(f"开始翻译：{task.source_path.name}", "TRANS")
                    start_index = (
                        task.resume_plan.translated_count
                        if task.resume_plan
                        and task.resume_plan.stage == CheckpointStage.TRANSLATING
                        else 0
                    )
                    if start_index:
                        self.log(
                            f"从断点继续翻译：已完成 {start_index}/{len(task.segments)} 行",
                            "SUCCESS",
                        )
                    result = self._translate_and_save(
                        task, context_summary, context_signature, context_ready, start_index
                    )
                    summary["total_translated_lines"] += sum(
                        1 for segment in task.segments
                        if segment.get("trans") and segment["trans"] != "(翻译失败)"
                    )
                else:
                    result = save_subtitles(
                        task.output_base,
                        task.segments,
                        self.options.export_fmt,
                        with_translation=False,
                    )
                    self.log(f"字幕已保存：{result.path.name}", "SUCCESS")
                summary["success_files"].append(
                    task.source_path.name + ("（部分）" if result.partial else "")
                )
                summary["output_files"].append(str(result.path))
                self.checkpoint_store.delete_completed(task)
            except TranslationCancelled as exc:
                summary["failed_files"].append(task.source_path.name)
                if exc.partial_result:
                    summary["partial_files"].append(str(exc.partial_result.path))
                    summary["output_files"].append(str(exc.partial_result.path))
                    self.log(f"部分结果已保存：{exc.partial_result.path.name}", "WARNING")
                self.cancel_event.set()
                raise
            except TranslationPartialError as exc:
                summary["failed_files"].append(task.source_path.name)
                if exc.partial_result:
                    summary["partial_files"].append(str(exc.partial_result.path))
                    summary["output_files"].append(str(exc.partial_result.path))
                    self.log(f"翻译失败，部分结果已保存：{exc.partial_result.path.name}", "WARNING")
                self.log(str(exc), "ERROR")
            except (SubtitleSaveError, TranslationAPIError, OSError) as exc:
                summary["failed_files"].append(task.source_path.name)
                self.log(f"处理失败：{task.source_path.name}；{exc}", "ERROR")
            self.report_progress(55 + (index + 1) / len(valid_tasks) * 45)
