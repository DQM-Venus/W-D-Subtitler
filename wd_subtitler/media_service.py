"""媒体探测、音轨选择与统一音频预处理。"""

import shutil
import tempfile
import wave
from pathlib import Path

import av

from .processing_models import AudioTrackInfo, MediaTask


class MediaProcessingError(RuntimeError):
    """媒体无法探测或解码。"""


class MediaWorkspace:
    """管理一次任务产生的临时音频目录。"""

    def __init__(self):
        self.path = Path(tempfile.mkdtemp(prefix="wd_subtitler_"))
        self._closed = False

    def close(self):
        if not self._closed:
            shutil.rmtree(self.path, ignore_errors=True)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def probe_audio_tracks(path):
    """返回媒体内全部音轨，并优先选择带日语标签的音轨。"""
    source = Path(path)
    if not source.exists():
        raise MediaProcessingError(f"文件不存在：{source}")

    try:
        with av.open(str(source), mode="r", metadata_errors="ignore") as container:
            tracks = []
            for audio_index, stream in enumerate(container.streams.audio):
                metadata = stream.metadata or {}
                language = str(metadata.get("language", "")).strip().lower()
                codec_context = stream.codec_context
                codec = getattr(codec_context, "name", "") or ""
                channels = getattr(codec_context, "channels", 0) or 0
                tracks.append(AudioTrackInfo(
                    stream_index=stream.index,
                    audio_index=audio_index,
                    language=language,
                    codec=codec,
                    channels=channels,
                ))
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise MediaProcessingError(f"无法读取媒体：{source.name}；{exc}") from exc

    if not tracks:
        raise MediaProcessingError(f"媒体中没有可用音轨：{source.name}")

    return tuple(tracks)


def choose_default_track(tracks):
    """优先选择日语音轨，否则选择第一条音轨。"""
    if not tracks:
        raise MediaProcessingError("没有可选择的音轨")
    for track in tracks:
        if track.language in {"ja", "jpn"}:
            return track.stream_index
    return tracks[0].stream_index


def create_media_task(path):
    tracks = probe_audio_tracks(path)
    return MediaTask(
        source_path=Path(path),
        tracks=tracks,
        selected_track_index=choose_default_track(tracks),
    )


def prepare_working_audio(task, workspace, cancel_check=None):
    """把所选音轨解码为可供全部后续阶段复用的 16 kHz 单声道 WAV。"""
    output_path = workspace.path / f"{len(list(workspace.path.glob('*.wav'))):04d}.wav"
    selected = task.selected_track
    try:
        with av.open(str(task.source_path), mode="r", metadata_errors="ignore") as container:
            stream = container.streams[selected.stream_index]
            resampler = av.audio.resampler.AudioResampler(
                format="s16",
                layout="mono",
                rate=16000,
            )
            with wave.open(str(output_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                for packet in container.demux(stream):
                    if cancel_check and cancel_check():
                        raise InterruptedError("媒体预处理已取消")
                    for frame in packet.decode():
                        for converted in resampler.resample(frame):
                            wav_file.writeframes(converted.to_ndarray().tobytes())
                for converted in resampler.resample(None):
                    wav_file.writeframes(converted.to_ndarray().tobytes())
    except InterruptedError:
        if output_path.exists():
            output_path.unlink()
        raise
    except (av.FFmpegError, OSError, ValueError) as exc:
        if output_path.exists():
            output_path.unlink()
        raise MediaProcessingError(
            f"音轨解码失败：{task.source_path.name}；{exc}"
        ) from exc

    if not output_path.exists() or output_path.stat().st_size <= 44:
        if output_path.exists():
            output_path.unlink()
        raise MediaProcessingError(f"音轨没有可解码的声音：{task.source_path.name}")

    task.working_audio_path = output_path
    return output_path
