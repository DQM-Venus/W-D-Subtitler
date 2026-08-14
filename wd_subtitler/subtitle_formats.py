"""字幕时间戳格式化工具。"""


def format_timestamp(seconds, fmt="srt"):
    """将秒数格式化为 SRT、LRC 或 VTT 时间戳。"""
    try:
        value = max(0.0, float(seconds))
    except (ValueError, TypeError):
        value = 0.0

    total_seconds = int(value)
    milliseconds = int((value - total_seconds) * 1000)
    seconds_part = total_seconds % 60
    if fmt == "lrc":
        minutes = total_seconds // 60
        return f"[{minutes:02}:{seconds_part:02}.{milliseconds // 10:02}]"

    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    if fmt == "srt":
        return f"{hours:02}:{minutes:02}:{seconds_part:02},{milliseconds:03}"
    if fmt == "vtt":
        return f"{hours:02}:{minutes:02}:{seconds_part:02}.{milliseconds:03}"
    return str(value)
