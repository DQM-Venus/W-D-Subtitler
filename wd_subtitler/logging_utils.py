"""统一的应用日志格式化工具。"""

import time
import unicodedata


LEVEL_LABELS = {
    "PHASE": "阶段",
    "INFO": "信息",
    "ASR": "识别",
    "TRANS": "翻译",
    "SUCCESS": "成功",
    "WARNING": "警告",
    "ERROR": "错误",
    "CONTEXT": "上下文",
    "SUMMARY": "摘要",
    "DEBUG": "调试",
}


def infer_log_level(message, default="INFO"):
    """根据服务层消息前缀推断日志级别。"""
    text = str(message).lstrip()
    if text.startswith("✅"):
        return "SUCCESS"
    if text.startswith("⚠️") or text.startswith("⚠"):
        return "WARNING"
    if text.startswith("❌"):
        return "ERROR"
    if text.startswith("⏹️") or text.startswith("⏹"):
        return "WARNING"
    return default


def normalize_log_message(message):
    """清理旧日志文案中的装饰符与多余缩进。"""
    lines = str(message).replace("\r\n", "\n").replace("\r", "\n").split("\n")
    normalized = []
    for line in lines:
        cleaned = line.strip()
        while cleaned and cleaned[0] in "✅❌⚠️ℹ️⏹️🎉💾🔎🧠⏱️📁📝🌐🚫•":
            cleaned = cleaned[1:].lstrip("️ ")
        normalized.append(cleaned)

    while normalized and not normalized[0]:
        normalized.pop(0)
    while normalized and not normalized[-1]:
        normalized.pop()
    return normalized or [""]


def display_width(text):
    """计算终端中包含中文字符的近似显示宽度。"""
    return sum(
        2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
        for character in str(text)
    )


def format_log_message(message, level="INFO", timestamp=None):
    """格式化为固定宽度的时间、级别与多行正文。"""
    level = str(level or "INFO").upper()
    timestamp = timestamp or time.strftime("%H:%M:%S")
    label = LEVEL_LABELS.get(level, level)
    prefix = f"[{timestamp}] [{label}] "
    continuation = " " * display_width(prefix)
    lines = normalize_log_message(message)
    return "\n".join(
        f"{prefix if index == 0 else continuation}{line}"
        for index, line in enumerate(lines)
    )


def format_phase_title(title, width=54):
    """生成规整的阶段分隔标题。"""
    title = str(title).strip()
    content = f" {title} "
    remaining = max(4, width - display_width(content))
    left = remaining // 2
    right = remaining - left
    return f"{'=' * left}{content}{'=' * right}"
