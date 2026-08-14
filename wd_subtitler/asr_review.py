"""Large-v3 可疑片段复核与候选选择。"""

import re


LARGE_V3_MODEL = "Systran/faster-whisper-large-v3"


def normalize_asr_text(text):
    """移除空白与常见标点，便于判断两个 ASR 候选是否一致。"""
    return re.sub(r"[\s。、，！？!?…・\-—~～]+", "", text or "").lower()


def build_review_clips(segments, padding=0.25, selection_key="review_requested"):
    """为待复核片段生成互不跨越相邻字幕的音频区间。"""
    clips = []
    for index, segment in enumerate(segments):
        if not segment.get(selection_key, segment.get("needs_review")):
            continue

        start = max(0.0, float(segment["start"]) - padding)
        end = float(segment["end"]) + padding
        if index > 0:
            start = max(start, float(segments[index - 1]["end"]))
        if index + 1 < len(segments):
            end = min(end, float(segments[index + 1]["start"]))
        if end <= start:
            start = float(segment["start"])
            end = max(start + 0.1, float(segment["end"]))

        clips.append({
            "segment_index": index,
            "start": round(start, 3),
            "end": round(end, 3),
        })
    return clips


def flatten_clip_timestamps(clips):
    """转换为 Faster-Whisper 接受的时间区间列表。"""
    timestamps = []
    for clip in clips:
        timestamps.extend([clip["start"], clip["end"]])
    return timestamps


def _overlap(start_a, end_a, start_b, end_b):
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def map_review_candidates(clips, review_segments):
    """按时间重叠把 Large-v3 输出映射回各个 Kotoba 片段。"""
    grouped = {clip["segment_index"]: [] for clip in clips}
    for candidate in review_segments:
        candidate_start = float(candidate["start"])
        candidate_end = float(candidate["end"])
        best_clip = None
        best_overlap = 0.0
        for clip in clips:
            overlap = _overlap(candidate_start, candidate_end, clip["start"], clip["end"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_clip = clip
        if best_clip is not None and best_overlap > 0:
            grouped[best_clip["segment_index"]].append(candidate)

    mapped = {}
    for segment_index, candidates in grouped.items():
        if not candidates:
            continue
        candidates.sort(key=lambda item: item["start"])
        text = "".join(item.get("text", "").strip() for item in candidates).strip()
        if not text:
            continue
        scores = [item.get("quality_score") for item in candidates if item.get("quality_score") is not None]
        mapped[segment_index] = {
            "text": text,
            "quality_score": min(scores) if scores else 0,
            "avg_logprob": min(
                (item["avg_logprob"] for item in candidates if item.get("avg_logprob") is not None),
                default=None,
            ),
            "no_speech_prob": max(
                (item["no_speech_prob"] for item in candidates if item.get("no_speech_prob") is not None),
                default=None,
            ),
            "compression_ratio": max(
                (item["compression_ratio"] for item in candidates if item.get("compression_ratio") is not None),
                default=None,
            ),
            "decoding_temperature": max(
                (item["decoding_temperature"] for item in candidates if item.get("decoding_temperature") is not None),
                default=None,
            ),
        }
    return mapped


def map_full_review_candidates(segments, review_segments):
    """把独立整段识别结果按时间顺序多对多投影到首轮字幕。"""
    projected = {index: [] for index in range(len(segments))}
    for candidate in review_segments:
        candidate_start = float(candidate["start"])
        candidate_end = float(candidate["end"])
        text = str(candidate.get("text", "")).strip()
        if not text or candidate_end <= candidate_start:
            continue
        targets = [
            index for index, segment in enumerate(segments)
            if _overlap(
                candidate_start,
                candidate_end,
                float(segment["start"]),
                float(segment["end"]),
            ) > 0
        ]
        if not targets:
            continue
        if len(targets) == 1:
            projected[targets[0]].append((text, candidate))
            continue

        cuts = [0]
        for target_index in targets[:-1]:
            boundary = min(float(segments[target_index]["end"]), candidate_end)
            ratio = (boundary - candidate_start) / (candidate_end - candidate_start)
            cuts.append(max(cuts[-1], min(len(text), round(len(text) * ratio))))
        cuts.append(len(text))
        for position, target_index in enumerate(targets):
            part = text[cuts[position]:cuts[position + 1]].strip()
            if part:
                projected[target_index].append((part, candidate))

    mapped = {}
    for index, parts in projected.items():
        if not parts:
            continue
        candidates = [candidate for _, candidate in parts]
        mapped[index] = {
            "text": "".join(text for text, _ in parts),
            "quality_score": min(
                (item.get("quality_score", 0) for item in candidates),
                default=0,
            ),
            "avg_logprob": min(
                (item["avg_logprob"] for item in candidates if item.get("avg_logprob") is not None),
                default=None,
            ),
            "no_speech_prob": max(
                (item["no_speech_prob"] for item in candidates if item.get("no_speech_prob") is not None),
                default=None,
            ),
            "compression_ratio": max(
                (item["compression_ratio"] for item in candidates if item.get("compression_ratio") is not None),
                default=None,
            ),
            "decoding_temperature": max(
                (item["decoding_temperature"] for item in candidates if item.get("decoding_temperature") is not None),
                default=None,
            ),
        }
    return mapped


def apply_review_candidates(segments, mapped_candidates):
    """比较两个模型；分歧时不再直接横向比较未经校准的分数。"""
    stats = {"reviewed": 0, "replaced": 0, "agreed": 0, "needs_ai": 0, "missing": 0}
    for index, segment in enumerate(segments):
        if not segment.get("review_requested", segment.get("needs_review")):
            continue

        stats["reviewed"] += 1
        primary_text = segment.get("text", "")
        primary_score = int(segment.get("quality_score", 0))
        candidate = mapped_candidates.get(index)
        segment["primary_candidate"] = {
            "text": primary_text,
            "quality_score": primary_score,
            "source": "kotoba-v2",
        }

        if not candidate:
            stats["missing"] += 1
            segment["review_status"] = "Large-v3 未返回有效候选"
            segment["needs_ai_review"] = False
            continue

        segment["review_candidate"] = {
            **candidate,
            "source": "large-v3",
        }
        candidate_text = candidate["text"]
        candidate_score = int(candidate.get("quality_score", 0))

        if normalize_asr_text(primary_text) == normalize_asr_text(candidate_text):
            stats["agreed"] += 1
            segment["review_status"] = "两模型结果一致"
            segment["needs_review"] = False
            segment["needs_ai_review"] = False
            segment["asr_source"] = "kotoba-v2+large-v3"
            segment["quality_score"] = max(primary_score, candidate_score)
            segment["low_confidence"] = False
            continue

        stats["needs_ai"] += 1
        segment["review_status"] = "候选存在分歧，等待第三次局部识别"
        segment["needs_ai_review"] = True
        segment["asr_source"] = "kotoba-v2"

    return stats


def apply_third_candidates(segments, mapped_candidates):
    """用第三次局部解码寻找二比一共识，仍分歧的候选交给 AI。"""
    stats = {"reviewed": 0, "resolved": 0, "selected_review": 0, "unresolved": 0}
    for index, segment in enumerate(segments):
        if not segment.get("needs_ai_review"):
            continue
        stats["reviewed"] += 1
        candidate = mapped_candidates.get(index)
        if not candidate or not candidate.get("text"):
            stats["unresolved"] += 1
            segment["review_status"] = "第三次识别无候选，等待 AI 裁决"
            continue
        segment["third_candidate"] = {**candidate, "source": "large-v3-local"}
        primary = segment.get("primary_candidate", {}).get("text", segment.get("text", ""))
        review = segment.get("review_candidate", {}).get("text", "")
        third = candidate["text"]
        normalized_third = normalize_asr_text(third)
        if normalized_third and normalized_third == normalize_asr_text(primary):
            segment["text"] = primary
            segment["asr_source"] = "kotoba-v2+local-consensus"
        elif normalized_third and normalized_third == normalize_asr_text(review):
            segment["text"] = review
            segment["asr_source"] = "large-v3+local-consensus"
            stats["selected_review"] += 1
        else:
            stats["unresolved"] += 1
            segment["review_status"] = "三个候选仍存在分歧，等待 AI 裁决"
            continue
        segment["needs_ai_review"] = False
        segment["needs_review"] = False
        segment["low_confidence"] = False
        segment["review_status"] = "第三次识别形成二比一共识"
        stats["resolved"] += 1
    return stats
