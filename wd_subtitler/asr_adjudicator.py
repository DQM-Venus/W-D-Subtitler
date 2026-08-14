"""Kotoba 与 Large-v3 分歧候选的 AI 裁决。"""

import json
import re

from .processing_models import ASRDecision


VALID_ACTIONS = {"primary", "review", "corrected"}


def _contains_japanese(text):
    return bool(re.search(r"[ぁ-ゖァ-ヺ一-龯々]", text or ""))


def _is_valid_correction(text, primary_text, review_text):
    corrected = str(text or "").strip()
    if not corrected or "\n" in corrected or "\r" in corrected:
        return False
    if not _contains_japanese(corrected):
        return False
    longest = max(len(primary_text.strip()), len(review_text.strip()), 1)
    return longest * 0.5 <= len(corrected) <= longest * 2.0


def _build_item(segments, index):
    segment = segments[index]
    primary = segment.get("primary_candidate") or {}
    review = segment.get("review_candidate") or {}
    third = segment.get("third_candidate") or {}
    duration = max(0.0, float(segment.get("end", 0)) - float(segment.get("start", 0)))
    return {
        "id": f"S{index + 1:05d}",
        "segment_index": index,
        "previous": [
            segments[pos].get("text", "")
            for pos in range(max(0, index - 2), index)
        ],
        "primary": {
            "text": primary.get("text", segment.get("text", "")),
            "quality_score": primary.get("quality_score", segment.get("quality_score", 0)),
            "avg_logprob": segment.get("avg_logprob"),
            "no_speech_prob": segment.get("no_speech_prob"),
            "compression_ratio": segment.get("compression_ratio"),
            "decoding_temperature": segment.get("decoding_temperature"),
            "risk_reasons": segment.get("review_reasons", []),
        },
        "review": {
            "text": review.get("text", ""),
            "quality_score": review.get("quality_score", 0),
            "avg_logprob": review.get("avg_logprob"),
            "no_speech_prob": review.get("no_speech_prob"),
            "compression_ratio": review.get("compression_ratio"),
            "decoding_temperature": review.get("decoding_temperature"),
        },
        "third": {
            "text": third.get("text", ""),
            "quality_score": third.get("quality_score", 0),
        },
        "duration_seconds": round(duration, 3),
        "next": [
            segments[pos].get("text", "")
            for pos in range(index + 1, min(len(segments), index + 3))
        ],
    }


def build_adjudication_batches(segments, max_items=20, max_chars=6000):
    """按条数和字符数限制构造裁决批次。"""
    items = [
        _build_item(segments, index)
        for index, segment in enumerate(segments)
        if segment.get("needs_ai_review")
    ]
    batches = []
    current = []
    current_chars = 0
    for item in items:
        item_chars = len(json.dumps(item, ensure_ascii=False))
        if current and (len(current) >= max_items or current_chars + item_chars > max_chars):
            batches.append(current)
            current = []
            current_chars = 0
        current.append(item)
        current_chars += item_chars
    if current:
        batches.append(current)
    return batches


def _parse_response(text, expected_ids):
    """解析 JSON Lines，也兼容返回 JSON 数组。"""
    raw = str(text or "").strip()
    if raw.startswith("```"):
        first_newline = raw.find("\n")
        raw = raw[first_newline + 1:] if first_newline >= 0 else ""
        if raw.rstrip().endswith("```"):
            raw = raw.rstrip()[:-3].rstrip()

    objects = []
    try:
        decoded = json.loads(raw)
        objects = decoded if isinstance(decoded, list) else [decoded]
    except (json.JSONDecodeError, TypeError):
        for line in raw.splitlines():
            try:
                objects.append(json.loads(line.strip()))
            except (json.JSONDecodeError, TypeError):
                continue

    decisions = {}
    for item in objects:
        if not isinstance(item, dict):
            continue
        segment_id = str(item.get("id", "")).strip()
        action = str(item.get("action", "")).strip().lower()
        if segment_id in expected_ids and action in VALID_ACTIONS:
            decisions[segment_id] = item
    return decisions


def _messages(items):
    system_prompt = (
        "你是日语语音识别校对员。请根据上下文裁决 Kotoba 与 Whisper Large-v3 的候选。\n"
        "质量分数只表示各模型内部风险，不能直接横向比较。第三候选来自扩大边界后的局部重识别。\n"
        "你无法听到音频；遇到人名、音近词或缺乏上下文证据时，应保守选择已有候选，不得凭剧情补写。\n"
        "优先从 primary 和 review 中选择；third 只用于验证共识。只有主要候选均明显错误时才允许 corrected。\n"
        "逐行输出 JSON，不要使用 Markdown。字段必须为："
        '{"id":"S00001","action":"primary|review|corrected",'
        '"corrected_text":"仅 corrected 时填写日文单行文本","reason":"简短中文理由"}。'
    )
    payload = "\n".join(json.dumps(item, ensure_ascii=False) for item in items)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload},
    ]


def _decision_from_raw(item, raw):
    primary_text = item["primary"]["text"].strip()
    review_text = item["review"]["text"].strip()
    action = raw["action"]
    reason = str(raw.get("reason", "")).strip()[:120]
    if action == "primary":
        return ASRDecision(item["id"], action, primary_text, "kotoba-v2", reason)
    if action == "review" and review_text:
        return ASRDecision(item["id"], action, review_text, "large-v3", reason)
    corrected = str(raw.get("corrected_text", "")).strip()
    if action == "corrected" and _is_valid_correction(corrected, primary_text, review_text):
        return ASRDecision(item["id"], action, corrected, "ai-corrected", reason)
    return None


def adjudicate_segments(segments, api_call):
    """裁决全部待处理片段；缺失项仅重试一次。"""
    stats = {"requested": 0, "resolved": 0, "corrected": 0, "failed": 0}
    for batch in build_adjudication_batches(segments):
        stats["requested"] += len(batch)
        expected = {item["id"] for item in batch}
        raw_result = api_call(_messages(batch))
        parsed = _parse_response(raw_result, expected)
        missing = [item for item in batch if item["id"] not in parsed]
        if missing:
            retry_result = api_call(_messages(missing))
            parsed.update(_parse_response(retry_result, {item["id"] for item in missing}))

        for item in batch:
            segment = segments[item["segment_index"]]
            raw = parsed.get(item["id"])
            decision = _decision_from_raw(item, raw) if raw else None
            if not decision:
                stats["failed"] += 1
                segment["review_status"] = "AI 裁决失败，保留 Kotoba 并等待人工复核"
                segment["needs_ai_review"] = True
                continue
            segment["text"] = decision.final_text
            segment["asr_source"] = decision.source
            segment["needs_ai_review"] = False
            segment["needs_review"] = False
            segment["review_status"] = f"AI 裁决：{decision.reason or decision.action}"
            segment["ai_decision"] = {
                "id": decision.segment_id,
                "action": decision.action,
                "source": decision.source,
                "reason": decision.reason,
            }
            stats["resolved"] += 1
            if decision.action == "corrected":
                stats["corrected"] += 1
    return stats
