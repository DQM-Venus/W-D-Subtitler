"""识别质量模式的界面说明。"""


def get_quality_mode_description(mode, large_v3_enabled=True):
    """返回当前质量模式对应的简明说明。"""
    if mode == "高质量":
        review_text = (
            "Large-v3 将独立识别完整音频，分歧片段再局部识别一次"
            if large_v3_enabled
            else "Large-v3 复核已关闭"
        )
        return (
            "高质量：使用更充分的解码搜索与温度回退；"
            f"{review_text}。准确性优先，处理时间更长。"
        )

    review_text = (
        "Large-v3 仅复核可疑片段，分歧片段再局部识别一次"
        if large_v3_enabled
        else "不使用 Large-v3 复核"
    )
    return (
        "快速：使用轻量解码；"
        f"{review_text}。速度优先，适合批量处理或初步预览。"
    )
