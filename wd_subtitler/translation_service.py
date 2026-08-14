import requests
import time
import re
import json
from dataclasses import dataclass
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .asr_adjudicator import adjudicate_segments
from .processing_control import ProcessingCancelled


@dataclass(frozen=True)
class APIErrorInfo:
    """翻译 API 的结构化错误信息。"""

    code: str
    message: str
    status_code: int | None = None
    retryable: bool = False
    detail: str = ""


class TranslationAPIError(RuntimeError):
    """翻译 API 调用失败。"""

    def __init__(self, info: APIErrorInfo):
        self.info = info
        super().__init__(info.message)

    def __str__(self):
        status = f"（HTTP {self.info.status_code}）" if self.info.status_code else ""
        detail = f"：{self.info.detail}" if self.info.detail else ""
        return f"{self.info.message}{status}{detail}"


class DeepSeekService:
    """DeepSeek V4 Flash 翻译与文本分析服务"""

    DEFAULT_MODEL = "deepseek-v4-flash"
    DEFAULT_API_URL = "https://api.deepseek.com/chat/completions"

    def __init__(self, model=None):
        self.session = requests.Session()
        # 重试由本服务显式控制，才能在每次重试前响应用户取消。
        retries = Retry(total=0)
        self.session.mount('https://', HTTPAdapter(max_retries=retries))
        self.model = model or self.DEFAULT_MODEL

    def _get_api_url(self, base_url):
        """构造完整的 API URL"""
        if not base_url or not base_url.strip():
            return self.DEFAULT_API_URL

        url = base_url.strip()
        if url.endswith("/"):
            url = url[:-1]
        if "chat/completions" not in url:
            return f"{url}/chat/completions"
        return url

    @staticmethod
    def _check_cancel(cancel_check):
        if cancel_check and cancel_check():
            raise ProcessingCancelled("处理已取消，当前 API 请求已结束")

    @classmethod
    def _retry_wait(cls, seconds, cancel_check):
        """可响应取消的短间隔重试等待。"""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            cls._check_cancel(cancel_check)
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))

    def _call_api(
        self,
        messages,
        api_key,
        base_url=None,
        temperature=0.3,
        timeout=120,
        max_tokens=None,
        cancel_check=None,
    ):
        """调用 API，成功时返回文本，失败时抛出结构化异常。"""
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            # DeepSeek V4 默认开启思考模式。字幕任务需要稳定的结构化正文，
            # 关闭思考可避免推理内容耗尽 max_tokens 后正文为空。
            "thinking": {"type": "disabled"},
        }
        # P1: 限制响应长度，防止模型跑飞
        if max_tokens:
            payload["max_tokens"] = max_tokens

        target_url = self._get_api_url(base_url)

        max_attempts = 3
        last_error = None
        for attempt in range(max_attempts):
            self._check_cancel(cancel_check)
            try:
                resp = self.session.post(
                    target_url,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                self._check_cancel(cancel_check)
                if resp.status_code == 200:
                    try:
                        data = resp.json()
                        choice = data["choices"][0]
                        content = choice["message"].get("content")
                        finish_reason = choice.get("finish_reason")
                    except (ValueError, KeyError, IndexError, TypeError) as e:
                        raise TranslationAPIError(APIErrorInfo(
                            code="invalid_response",
                            message="翻译 API 返回格式异常",
                            status_code=200,
                            detail=str(e),
                        )) from e

                    if not isinstance(content, str) or not content.strip():
                        detail = (
                            "输出达到长度限制，模型未生成最终正文"
                            if finish_reason == "length"
                            else f"finish_reason={finish_reason or '未知'}"
                        )
                        last_error = TranslationAPIError(APIErrorInfo(
                            code="empty_response",
                            message="翻译 API 返回了空内容",
                            status_code=200,
                            retryable=True,
                            detail=detail,
                        ))
                        if attempt < max_attempts - 1:
                            self._retry_wait(1 + attempt, cancel_check)
                            continue
                        raise last_error
                    return content

                detail = resp.text[:200]
                if resp.status_code == 400:
                    raise TranslationAPIError(APIErrorInfo(
                        code="bad_request",
                        message="翻译请求参数错误",
                        status_code=400,
                        detail=detail,
                    ))
                if resp.status_code == 401:
                    raise TranslationAPIError(APIErrorInfo(
                        code="unauthorized",
                        message="API 密钥错误，请检查 API Key",
                        status_code=401,
                    ))

                retryable = resp.status_code == 429 or 500 <= resp.status_code < 600
                last_error = TranslationAPIError(APIErrorInfo(
                    code="rate_limited" if resp.status_code == 429 else "http_error",
                    message="API 请求过于频繁" if resp.status_code == 429 else "翻译 API 请求失败",
                    status_code=resp.status_code,
                    retryable=retryable,
                    detail=detail,
                ))
                if retryable and attempt < max_attempts - 1:
                    wait_time = 5 * (attempt + 1) if resp.status_code == 429 else 2 + attempt * 2
                    self._retry_wait(wait_time, cancel_check)
                    continue
                raise last_error
            except TranslationAPIError:
                raise
            except requests.exceptions.Timeout as e:
                last_error = TranslationAPIError(APIErrorInfo(
                    code="timeout",
                    message="翻译 API 请求超时",
                    retryable=True,
                ))
                if attempt < max_attempts - 1:
                    self._retry_wait(2, cancel_check)
                    continue
                raise last_error from e
            except requests.exceptions.RequestException as e:
                last_error = TranslationAPIError(APIErrorInfo(
                    code="network_error",
                    message="翻译 API 网络请求失败",
                    retryable=True,
                    detail=str(e),
                ))
                if attempt < max_attempts - 1:
                    self._retry_wait(2, cancel_check)
                    continue
                raise last_error from e

        raise last_error or TranslationAPIError(APIErrorInfo(
            code="unknown_error",
            message="翻译 API 请求失败",
        ))

    def analyze_full_text(self, combined_text, api_key, base_url=None, cancel_check=None):
        """生成系列翻译总纲，确保跨文件翻译一致性"""
        system_prompt = (
            "你是一个专业的影视字幕翻译顾问。阅读以下包含多个视频/音频的日语字幕合集，"
            "总结出一份「系列翻译总纲」，以确保整个系列的翻译统一性。\n"
            "请输出以下信息：\n"
            "1. 【系列背景】：总结这些文件共同讲述的故事或主题（时间、地点、核心事件）。\n"
            "2. 【角色关系网】：列出贯穿系列的主要角色名（保留日文原名+中文译名），并说明他们的身份及关系。\n"
            "3. 【关键术语表】：列出特殊的专有名词、机构名、技能名及其统一译名。\n"
            "4. 【整体语气】：对话的整体风格（如正式、街头、古风、学术、日常等）。\n"
            "只记录原文能够支持的事实，不得补写剧情。无法确认的名称或关系放入待确认项。\n"
            "只输出 JSON 对象，字段为 background、characters、terms、tone、uncertain；"
            "characters 和 terms 中每项都要包含 source、translation、evidence。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请分析以下字幕合集内容：\n\n{combined_text}"}
        ]

        return self._call_api(
            messages, api_key, base_url, temperature=0.2, timeout=180,
            cancel_check=cancel_check,
        )

    def extract_terms(self, text, api_key, base_url=None, cancel_check=None):
        """
        P2: 从单文件文本中提取人名和术语表（轻量版，比全局大纲快）。
        返回结构化术语 JSON，可直接注入翻译提示词。
        """
        # 截断过长文本
        if len(text) > 20000:
            text = text[:10000] + "\n...\n" + text[-5000:]

        system_prompt = (
            "你是一个日语术语提取助手。阅读以下日语字幕文本，提取其中的：\n"
            "1. 角色人名（保留日文写法，给出建议中文译名）\n"
            "2. 专有名词/术语（地名、机构名、技能名等，给出建议中文译名）\n"
            "只输出 JSON 对象，格式为 "
            '{"terms":[{"source":"日文原文","translation":"中文译名",'
            '"evidence":"出现该词的短句"}],"uncertain":[]}。\n'
            "无法确定译名时放入 uncertain，不得猜测。没有内容时 terms 输出空数组。最多 30 条。"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"请分析以下文本：\n\n{text}"}
        ]

        result = self._call_api(
            messages, api_key, base_url, temperature=0.1, timeout=60,
            cancel_check=cancel_check,
        )
        return result.strip()

    def adjudicate_asr(self, segments, api_key, base_url=None, cancel_check=None):
        """对 Kotoba 与 Large-v3 的分歧候选执行结构化 AI 裁决。"""
        def api_call(messages):
            return self._call_api(
                messages,
                api_key,
                base_url,
                temperature=0.1,
                timeout=90,
                max_tokens=3000,
                cancel_check=cancel_check,
            )

        return adjudicate_segments(segments, api_call)

    @staticmethod
    def _locked_terms(context_summary):
        """从结构化总纲中提取可确定的术语映射。"""
        if not context_summary:
            return {}
        raw = context_summary.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw, flags=re.I)
        try:
            data = json.loads(raw)
        except (ValueError, TypeError):
            return {}
        terms = []
        if isinstance(data, dict):
            terms.extend(data.get("terms", []))
            terms.extend(data.get("characters", []))
        return {
            str(item.get("source", "")).strip(): str(item.get("translation", "")).strip()
            for item in terms
            if isinstance(item, dict)
            and str(item.get("source", "")).strip()
            and str(item.get("translation", "")).strip()
        }

    @staticmethod
    def _translation_target(length_seconds):
        """按显示时长给出柔性的中文字幕长度参考。"""
        return max(8, min(32, round(max(0.5, length_seconds) * 5.5)))

    def _build_translation_messages(
        self,
        segments,
        line_ids,
        context_summary="",
        prev_context=None,
        next_preview=None,
    ):
        lines = []
        for line_id, segment in zip(line_ids, segments):
            text = segment["text"].strip()
            duration = max(
                0.0,
                float(segment.get("end", 0)) - float(segment.get("start", 0)),
            )
            reliability = "低" if segment.get("low_confidence") else "正常"
            lines.append(
                f"[L{line_id}] [时长={duration:.1f}s] [ASR可靠度={reliability}] {text}"
            )

        context_parts = []
        if context_summary:
            context_parts.append(f"=== 翻译参考 ===\n{context_summary}\n================")
        if prev_context:
            previous = []
            for item in prev_context[-5:]:
                previous.extend((
                    f"原文：{item['original']}",
                    f"译文：{item['translated']}",
                ))
            context_parts.append("=== 上文参考 ===\n" + "\n".join(previous) + "\n================")
        if next_preview:
            context_parts.append(
                "=== 下文预览 ===\n"
                + "\n".join(f"原文：{text}" for text in next_preview[:3])
                + "\n================"
            )

        system_prompt = (
            "你是资深日语影视字幕翻译。请把带稳定行号的日文翻译为自然、准确、简洁的中文。\n"
            + ("\n".join(context_parts) + "\n" if context_parts else "")
            + "要求：\n"
            "1. 每个输入行号必须且只能输出一次，格式严格为 [L序号] 译文；不得合并、遗漏或重排行。\n"
            "2. 忽略时长和 ASR 可靠度元数据，不要把它们抄进译文。\n"
            "3. 仅当上下文明确且中文表达确有必要时补全主语；无法确定时保持省略，不得猜测人物身份或剧情。\n"
            "4. ASR 可靠度低时，在译文开头保留 [听不清]；只翻译现有原文，不得根据剧情补写。\n"
            "5. 字幕长度是柔性目标：结合每行显示时长压缩表达，但不得为了变短删除关键信息。\n"
            "6. 严格遵守翻译参考中的已确定名称和术语；待确认项不得擅自定译。\n"
            "7. 仅输出翻译行，不要解释、Markdown 或其他文字。"
        )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": "\n".join(lines)},
        ]

    def _translation_issue(self, segment, translation, locked_terms):
        """返回确定性译文问题；空字符串表示通过。"""
        text = str(translation or "").strip()
        source = str(segment.get("text", "")).strip()
        if not text:
            return "缺失"
        if segment.get("low_confidence") and not text.startswith("[听不清]"):
            return "低置信度标记丢失"
        clean_text = text.removeprefix("[听不清]").strip()
        japanese_count = len(re.findall(r"[ぁ-ゖァ-ヺ]", clean_text))
        if japanese_count >= max(4, len(clean_text) // 3):
            return "残留日文过多"
        for original, expected in locked_terms.items():
            if original in source and expected not in clean_text:
                return f"术语未统一：{original}"
        duration = max(0.5, float(segment.get("end", 0)) - float(segment.get("start", 0)))
        hard_limit = max(30, self._translation_target(duration) * 2)
        if len(clean_text) > hard_limit:
            return "译文远超显示时长"
        return ""

    def translate_batch(
        self,
        segments_batch,
        api_key,
        base_url=None,
        context_summary="",
        prev_context=None,
        next_preview=None,
        cancel_check=None,
    ):
        """
        批量翻译字幕片段。

        P0 改进：
        - temperature 降至 0.3，保证翻译一致性
        - 支持滑窗上下文（prev_context + next_preview）

        P2 改进：
        - 翻译结果校验与自动重试

        Args:
            segments_batch: 字幕片段列表
            context_summary: 全局大纲或术语表
            prev_context: 前文上下文 [{"original": str, "translated": str}, ...]
            next_preview: 后文预览 [str, ...]
        """
        if not segments_batch:
            return []

        messages = self._build_translation_messages(
            segments_batch,
            range(len(segments_batch)),
            context_summary,
            prev_context,
            next_preview,
        )

        # P0: temperature 1.0 → 0.3, P1: max_tokens 限制
        batch_size = len(segments_batch)
        max_tokens = min(4000, batch_size * 80 + 200)
        result = self._call_api(
            messages, api_key, base_url, temperature=0.3, timeout=90,
            max_tokens=max_tokens, cancel_check=cancel_check,
        )

        translations = self.parse_batch_response(result, len(segments_batch))

        # 校验翻译结果，对缺失或不可靠的行重试。
        return self._validate_and_retry(
            translations, segments_batch, api_key, base_url,
            context_summary, prev_context, next_preview, cancel_check
        )

    def _validate_and_retry(
        self,
        translations,
        segments_batch,
        api_key,
        base_url,
        context_summary,
        prev_context,
        next_preview,
        cancel_check,
    ):
        """
        校验译文，对缺失或违反确定性规则的行进行小批量重试。
        """
        locked_terms = self._locked_terms(context_summary)
        missing_indices = [
            index
            for index, (segment, translation) in enumerate(zip(segments_batch, translations))
            if self._translation_issue(segment, translation, locked_terms)
        ]

        if not missing_indices:
            return translations

        # 缺失超过一半，直接用小批量重试全部
        if len(missing_indices) > len(segments_batch) // 2:
            return self._retry_full_batch(
                segments_batch, api_key, base_url,
                context_summary, prev_context, next_preview, cancel_check
            )

        # 只重试缺失的行
        retry_batch = [segments_batch[i] for i in missing_indices]

        messages = self._build_translation_messages(
            retry_batch,
            missing_indices,
            context_summary,
            prev_context,
            next_preview,
        )

        result = self._call_api(
            messages, api_key, base_url, temperature=0.3, timeout=60,
            cancel_check=cancel_check,
        )

        retry_translations = self.parse_batch_response(result, len(segments_batch))
        for idx in missing_indices:
            if retry_translations[idx]:
                translations[idx] = retry_translations[idx]

        # 仍未通过校验的行标记为失败，避免把明显异常译文写入正式字幕。
        for i, segment in enumerate(segments_batch):
            if self._translation_issue(segment, translations[i], locked_terms):
                translations[i] = "(翻译失败)"

        return translations

    def _retry_full_batch(
        self,
        segments_batch,
        api_key,
        base_url,
        context_summary,
        prev_context,
        next_preview,
        cancel_check,
    ):
        """全部重试（分多个小批次，每批 15 条，并保留完整翻译协议）。"""
        translations = [""] * len(segments_batch)
        CHUNK_SIZE = 15

        for chunk_start in range(0, len(segments_batch), CHUNK_SIZE):
            chunk_end = min(chunk_start + CHUNK_SIZE, len(segments_batch))
            small_batch = segments_batch[chunk_start:chunk_end]

            stable_ids = range(chunk_start, chunk_end)
            messages = self._build_translation_messages(
                small_batch,
                stable_ids,
                context_summary,
                prev_context,
                next_preview,
            )

            max_tokens = min(2000, len(small_batch) * 80 + 200)
            result = self._call_api(
                messages, api_key, base_url, temperature=0.3, timeout=60,
                max_tokens=max_tokens, cancel_check=cancel_check,
            )

            retry_trans = self.parse_batch_response(result, len(segments_batch))
            for stable_id in range(chunk_start, chunk_end):
                if retry_trans[stable_id]:
                    translations[stable_id] = retry_trans[stable_id]

        # 标记仍缺失的行
        locked_terms = self._locked_terms(context_summary)
        for i, segment in enumerate(segments_batch):
            if self._translation_issue(segment, translations[i], locked_terms):
                translations[i] = "(翻译失败)"

        return translations

    def parse_batch_response(self, text, count):
        """解析批量翻译返回结果，按 [Lx] 标记提取译文"""
        # P0: 去除 Markdown 代码块包裹
        text = text.strip()
        if text.startswith("```"):
            # 去掉开头的 ```markdown 或 ```
            first_newline = text.find('\n')
            if first_newline > 0:
                text = text[first_newline + 1:]
            # 去掉结尾的 ```
            if text.rstrip().endswith("```"):
                text = text.rstrip()[:-3].rstrip()

        translations = [""] * count
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if not line:
                continue
            match = re.match(r'\[L(\d+)\]\s*(.*)', line)
            if match:
                try:
                    idx = int(match.group(1))
                    content = match.group(2).strip()
                    if 0 <= idx < count:
                        translations[idx] = content
                except (ValueError, IndexError):
                    continue
        return translations
