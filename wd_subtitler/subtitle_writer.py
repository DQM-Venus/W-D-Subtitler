"""可靠、无覆盖的字幕文件写入。"""

import os
import sys
import tempfile
from pathlib import Path

from .processing_models import SubtitleSaveResult
from .subtitle_formats import format_timestamp


class SubtitleSaveError(RuntimeError):
    """字幕无法安全保存。"""


def candidate_paths(base_path, fmt, partial=False):
    """按编号顺序持续生成候选输出路径。"""
    base = Path(base_path)
    marker = ".partial" if partial else ""
    yield base.parent / f"{base.name}{marker}.{fmt}"
    index = 1
    while True:
        yield base.parent / f"{base.name}{marker} ({index}).{fmt}"
        index += 1


def choose_available_path(base_path, fmt, partial=False):
    """返回当前首个可用名称，仅供界面预览。"""
    return next(path for path in candidate_paths(base_path, fmt, partial) if not path.exists())


def _publish_without_overwrite(temp_path: Path, final_path: Path) -> None:
    """原子发布字幕，目标存在时绝不覆盖。"""
    if sys.platform == "win32":
        try:
            import ctypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            move_file = kernel32.MoveFileW
            move_file.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
            move_file.restype = ctypes.c_int
            if move_file(str(temp_path), str(final_path)):
                return
            error_code = ctypes.get_last_error()
            if error_code in {80, 183}:
                raise FileExistsError(str(final_path))
            raise OSError(error_code, f"Windows 原子提交失败：错误码 {error_code}")
        except ImportError as exc:
            raise OSError("当前 Windows 环境不支持安全原子提交") from exc

    try:
        os.link(temp_path, final_path)
    except FileExistsError:
        raise
    except OSError as exc:
        raise OSError(f"当前文件系统不支持安全原子提交：{exc}") from exc
    try:
        temp_path.unlink()
    except OSError:
        # 正式文件已经原子发布成功；残留临时文件不应把成功结果误报为失败。
        pass


def _segment_content(segment, with_translation, partial):
    translation = str(segment.get("trans") or "").strip()
    if with_translation and translation and translation != "(翻译失败)":
        return translation
    original = str(segment.get("text") or "").strip()
    if partial and with_translation:
        return f"[未翻译] {original}"
    return original


def _render_subtitles(segments, fmt, with_translation, partial):
    lines = []
    for index, segment in enumerate(segments, 1):
        content = _segment_content(segment, with_translation, partial)
        if fmt == "srt":
            start = format_timestamp(segment["start"], "srt")
            end = format_timestamp(segment["end"], "srt")
            lines.append(f"{index}\n{start} --> {end}\n{content}\n")
        elif fmt == "lrc":
            start = format_timestamp(segment["start"], "lrc")
            lines.append(f"{start}{content}")
        else:
            raise SubtitleSaveError(f"不支持的字幕格式：{fmt}")
    return "\n".join(lines) + ("\n" if lines else "")


def save_subtitles(base_path, segments, fmt, with_translation=True, partial=False):
    """先写临时文件并同步到磁盘，再原子提交为最终字幕。"""
    base = Path(base_path)
    base.parent.mkdir(parents=True, exist_ok=True)
    content = _render_subtitles(segments, fmt, with_translation, partial)
    temp_path = None
    try:
        file_descriptor, raw_temp_path = tempfile.mkstemp(
            prefix=f".{base.name}.",
            suffix=".tmp",
            dir=str(base.parent),
        )
        temp_path = Path(raw_temp_path)
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as output:
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        final_path = None
        for candidate in candidate_paths(base, fmt, partial=partial):
            try:
                _publish_without_overwrite(temp_path, candidate)
                final_path = candidate
                break
            except FileExistsError:
                continue
        if final_path is None:
            raise OSError("无法分配可用的字幕文件名")
        return SubtitleSaveResult(
            path=final_path,
            partial=partial,
            written_lines=len(segments),
        )
    except (OSError, ValueError) as exc:
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise SubtitleSaveError(f"字幕保存失败：{base.name}.{fmt}；{exc}") from exc
