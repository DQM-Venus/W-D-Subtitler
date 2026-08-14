"""基于 Silero VAD 的轻量字幕时间轴精修。"""

from dataclasses import dataclass

from faster_whisper.audio import decode_audio
from faster_whisper.vad import VadOptions, get_speech_timestamps


@dataclass(frozen=True)
class TimelineRefinementStats:
    """单个文件的时间轴精修统计。"""

    refined: int
    skipped: int
    fallback: bool = False


def detect_speech_intervals(audio_path):
    """读取 16 kHz 工作音频并返回秒级语音区间。"""
    audio = decode_audio(str(audio_path), sampling_rate=16000)
    timestamps = get_speech_timestamps(
        audio,
        VadOptions(
            threshold=0.4,
            min_speech_duration_ms=100,
            min_silence_duration_ms=150,
            speech_pad_ms=80,
        ),
        sampling_rate=16000,
    )
    return [
        {
            "start": item["start"] / 16000.0,
            "end": item["end"] / 16000.0,
        }
        for item in timestamps
    ]


def _overlap(start_a, end_a, start_b, end_b):
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def split_segments_at_speech_gaps(
    segments,
    speech_intervals,
    max_chars=35,
    min_gap=0.4,
    min_part_chars=6,
):
    """利用片段内部的可靠静音点拆分缺少标点的长句。"""
    if not segments or not speech_intervals:
        return 0

    result = []
    split_count = 0
    for segment in segments:
        text = str(segment.get("text", "")).strip()
        start = float(segment["start"])
        end = float(segment["end"])
        if len(text) <= max_chars or any(mark in text for mark in "。？！?!…"):
            result.append(segment)
            continue

        speech = [
            interval for interval in speech_intervals
            if _overlap(interval["start"], interval["end"], start, end) > 0.05
        ]
        boundaries = [
            (speech[index]["end"] + speech[index + 1]["start"]) / 2.0
            for index in range(len(speech) - 1)
            if speech[index + 1]["start"] - speech[index]["end"] >= min_gap
            and start + 0.3 < speech[index]["end"] < end - 0.3
        ]
        if not boundaries:
            result.append(segment)
            continue

        time_points = [start, *boundaries, end]
        durations = [
            time_points[index + 1] - time_points[index]
            for index in range(len(time_points) - 1)
        ]
        total_duration = max(sum(durations), 0.001)
        char_points = [0]
        consumed = 0
        for duration in durations[:-1]:
            consumed += duration
            proposed = round(len(text) * consumed / total_duration)
            lower = char_points[-1] + min_part_chars
            upper = len(text) - min_part_chars * (len(durations) - len(char_points))
            char_points.append(max(lower, min(proposed, upper)))
        char_points.append(len(text))
        if any(
            char_points[index + 1] - char_points[index] < min_part_chars
            for index in range(len(char_points) - 1)
        ):
            result.append(segment)
            continue

        for index in range(len(durations)):
            part = dict(segment)
            part["start"] = round(time_points[index], 3)
            part["end"] = round(time_points[index + 1], 3)
            part["text"] = text[char_points[index]:char_points[index + 1]]
            part["words"] = None
            result.append(part)
        split_count += len(durations) - 1

    segments[:] = result
    return split_count


def refine_timeline(
    segments,
    speech_intervals,
    max_adjustment=0.5,
    min_duration=0.3,
    min_gap=0.02,
):
    """在相邻字幕安全区间内，把片段边界收紧到语音边界。"""
    if not segments or not speech_intervals:
        return TimelineRefinementStats(refined=0, skipped=len(segments))

    original_times = [
        (float(segment["start"]), float(segment["end"]))
        for segment in segments
    ]
    candidates = []
    for index, (original_start, original_end) in enumerate(original_times):
        safe_start = 0.0
        safe_end = float("inf")
        if index > 0:
            safe_start = (original_times[index - 1][1] + original_start) / 2.0
        if index + 1 < len(original_times):
            safe_end = (original_end + original_times[index + 1][0]) / 2.0

        search_start = max(safe_start, original_start - max_adjustment)
        search_end = min(safe_end, original_end + max_adjustment)
        matches = [
            interval for interval in speech_intervals
            if _overlap(
                interval["start"], interval["end"], search_start, search_end
            ) > 0
        ]
        if not matches:
            candidates.append(None)
            continue

        new_start = max(search_start, matches[0]["start"])
        new_end = min(search_end, matches[-1]["end"])
        new_start = max(original_start - max_adjustment, new_start)
        new_end = min(original_end + max_adjustment, new_end)
        if new_end - new_start < min_duration:
            candidates.append(None)
            continue
        candidates.append((new_start, new_end))

    refined = 0
    skipped = 0
    previous_end = None
    for index, segment in enumerate(segments):
        candidate = candidates[index]
        if candidate is None:
            skipped += 1
            previous_end = float(segment["end"])
            continue

        new_start, new_end = candidate
        if previous_end is not None:
            new_start = max(new_start, previous_end + min_gap)
        if index + 1 < len(segments):
            next_start = float(segments[index + 1]["start"])
            new_end = min(new_end, next_start - min_gap)
        if new_end - new_start < min_duration:
            skipped += 1
            previous_end = float(segment["end"])
            continue

        changed = (
            abs(new_start - float(segment["start"])) >= 0.001
            or abs(new_end - float(segment["end"])) >= 0.001
        )
        segment["start"] = round(new_start, 3)
        segment["end"] = round(new_end, 3)
        previous_end = new_end
        if changed:
            refined += 1
        else:
            skipped += 1

    return TimelineRefinementStats(refined=refined, skipped=skipped)


def refine_timeline_from_audio(segments, audio_path, speech_intervals=None):
    """精修失败时完整保留原时间轴。"""
    snapshot = [
        (segment.get("start"), segment.get("end"))
        for segment in segments
    ]
    try:
        intervals = speech_intervals or detect_speech_intervals(audio_path)
        return refine_timeline(segments, intervals)
    except Exception:
        for segment, (start, end) in zip(segments, snapshot):
            segment["start"] = start
            segment["end"] = end
        return TimelineRefinementStats(
            refined=0,
            skipped=len(segments),
            fallback=True,
        )
