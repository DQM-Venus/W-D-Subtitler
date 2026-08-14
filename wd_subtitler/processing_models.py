"""处理任务使用的不可变配置与媒体数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .processing_control import CheckpointStage


@dataclass(frozen=True)
class AudioTrackInfo:
    """媒体中的一条音轨。"""

    stream_index: int
    audio_index: int
    language: str
    codec: str
    channels: int

    @property
    def display_name(self):
        language = self.language or "未知语言"
        codec = self.codec or "未知编码"
        channels = f"{self.channels} 声道" if self.channels else "声道未知"
        return f"音轨 {self.audio_index + 1}｜{language}｜{codec}｜{channels}"


@dataclass
class MediaTask:
    """单个输入媒体在一次处理任务中的状态。"""

    source_path: Path
    tracks: tuple[AudioTrackInfo, ...]
    selected_track_index: int
    working_audio_path: Path | None = None
    segments: list[dict] = field(default_factory=list)
    speech_intervals: list[dict] = field(default_factory=list)
    resume_plan: ResumePlan | None = None

    @property
    def selected_track(self):
        return next(
            track for track in self.tracks
            if track.stream_index == self.selected_track_index
        )

    @property
    def output_base(self):
        return self.source_path.with_suffix("")


@dataclass(frozen=True)
class ProcessingOptions:
    """点击开始时从界面生成的不可变配置快照。"""

    trans_key: str
    base_url: str
    asr_hotwords: str
    do_translate: bool
    use_context: bool
    export_fmt: str
    model_precision: str
    quality_mode: str
    large_v3_review: bool
    ai_asr_arbitration: bool
    timeline_refinement: bool


@dataclass(frozen=True)
class SubtitleSaveResult:
    """字幕文件成功落盘后的结果。"""

    path: Path
    partial: bool
    written_lines: int


@dataclass(frozen=True)
class CheckpointSnapshot:
    """磁盘断点的公开摘要。"""

    path: Path
    source_path: Path
    stage: CheckpointStage
    updated_at: datetime
    translated_count: int = 0


@dataclass(frozen=True)
class ResumePlan:
    """当前配置可以安全复用的处理结果。"""

    checkpoint_path: Path
    stage: CheckpointStage
    segments: list[dict]
    updated_at: datetime
    translated_count: int = 0
    context_input_signature: str = ""
    context_summary: str = ""


@dataclass(frozen=True)
class ASRDecision:
    """AI 对单个 ASR 分歧片段作出的裁决。"""

    segment_id: str
    action: str
    final_text: str
    source: str
    reason: str
