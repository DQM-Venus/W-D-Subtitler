"""ASR 热词与模型独立性策略。"""


def normalize_hotwords(value, max_length=500):
    """清理用户输入的热词，不从文件名或其他字段自动推断。"""
    if not value:
        return ""
    lines = [line.strip() for line in str(value).replace("\r", "\n").split("\n")]
    normalized = " ".join(line for line in lines if line)
    return normalized[:max_length].strip()


def get_primary_prompt_options(hotwords):
    """Kotoba 首轮仅使用用户明确填写的热词。"""
    normalized = normalize_hotwords(hotwords)
    return {
        "initial_prompt": None,
        "hotwords": normalized or None,
    }


def get_review_prompt_options():
    """Large-v3 不继承首轮热词，以保持候选独立性。"""
    return {
        "initial_prompt": None,
        "hotwords": None,
    }
