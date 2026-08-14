"""翻译服务回归测试。"""

import unittest
from unittest.mock import patch

from wd_subtitler.translation_service import (
    APIErrorInfo,
    DeepSeekService,
    TranslationAPIError,
)
from wd_subtitler.processing_control import ProcessingCancelled


class FakeResponse:
    """最小 HTTP 响应替身。"""

    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        return self._payload


class TranslationServiceTests(unittest.TestCase):
    def test_正常行标记响应会被解析(self):
        service = DeepSeekService()
        service._call_api = lambda *args, **kwargs: "[L0] 你好\n[L1] 世界"
        segments = [{"text": "こんにちは"}, {"text": "世界"}]

        result = service.translate_batch(segments, "测试密钥")

        self.assertEqual(["你好", "世界"], result)

    def test_认证失败返回结构化异常(self):
        service = DeepSeekService()
        service.session.post = lambda *args, **kwargs: FakeResponse(401)

        with self.assertRaises(TranslationAPIError) as raised:
            service._call_api([], "错误密钥")

        info = raised.exception.info
        self.assertEqual("unauthorized", info.code)
        self.assertEqual(401, info.status_code)
        self.assertFalse(info.retryable)

    def test_异常字符串包含结构化详情(self):
        error = TranslationAPIError(APIErrorInfo(
            code="bad_request",
            message="翻译请求参数错误",
            status_code=400,
            detail="模型不存在",
        ))

        self.assertIn("HTTP 400", str(error))
        self.assertIn("模型不存在", str(error))

    def test_空响应会重试并关闭思考模式(self):
        service = DeepSeekService()
        responses = iter([
            FakeResponse(200, {
                "choices": [{
                    "message": {"content": "", "reasoning_content": "仍在思考"},
                    "finish_reason": "length",
                }],
            }),
            FakeResponse(200, {
                "choices": [{
                    "message": {"content": "[L0] 正常译文"},
                    "finish_reason": "stop",
                }],
            }),
        ])
        payloads = []

        def fake_post(*args, **kwargs):
            payloads.append(kwargs["json"])
            return next(responses)

        service.session.post = fake_post
        with patch("wd_subtitler.translation_service.time.sleep"):
            result = service._call_api([], "测试密钥", max_tokens=200)

        self.assertEqual("[L0] 正常译文", result)
        self.assertEqual(2, len(payloads))
        self.assertEqual({"type": "disabled"}, payloads[0]["thinking"])

    def test_连续空响应返回可重试结构化错误(self):
        service = DeepSeekService()
        service.session.post = lambda *args, **kwargs: FakeResponse(200, {
            "choices": [{
                "message": {"content": None},
                "finish_reason": "length",
            }],
        })

        with patch("wd_subtitler.translation_service.time.sleep"):
            with self.assertRaises(TranslationAPIError) as raised:
                service._call_api([], "测试密钥")

        self.assertEqual("empty_response", raised.exception.info.code)
        self.assertTrue(raised.exception.info.retryable)
        self.assertIn("长度限制", raised.exception.info.detail)

    def test_API返回后检测到取消不会继续处理或重试(self):
        service = DeepSeekService()
        calls = []
        service.session.post = lambda *args, **kwargs: (
            calls.append(1)
            or FakeResponse(200, {
                "choices": [{"message": {"content": "[L0] 译文"}, "finish_reason": "stop"}]
            })
        )
        checks = iter([False, True])

        with self.assertRaises(ProcessingCancelled):
            service._call_api([], "测试密钥", cancel_check=lambda: next(checks))

        self.assertEqual(1, len(calls))

    def test_翻译提示词禁止猜测并包含时长(self):
        service = DeepSeekService()
        messages = service._build_translation_messages(
            [{"start": 0.0, "end": 2.5, "text": "分かりません", "low_confidence": True}],
            [7],
        )

        self.assertIn("无法确定时保持省略", messages[0]["content"])
        self.assertIn("不得根据剧情补写", messages[0]["content"])
        self.assertIn("[L7] [时长=2.5s] [ASR可靠度=低]", messages[1]["content"])

    def test_低置信度标记丢失会触发原编号重试(self):
        service = DeepSeekService()
        responses = iter([
            "[L0] 不知道\n[L1] 没问题",
            "[L0] [听不清] 不知道",
        ])
        requests = []

        def fake_call(messages, *args, **kwargs):
            requests.append(messages)
            return next(responses)

        service._call_api = fake_call
        result = service.translate_batch(
            [
                {"start": 0.0, "end": 1.0, "text": "分かりません", "low_confidence": True},
                {"start": 1.0, "end": 2.0, "text": "大丈夫です"},
            ],
            "测试密钥",
        )

        self.assertEqual(["[听不清] 不知道", "没问题"], result)
        self.assertIn("[L0]", requests[1][1]["content"])

    def test_结构化术语不一致会触发修复(self):
        service = DeepSeekService()
        context = '{"terms":[{"source":"夏希","translation":"夏希","evidence":"夏希先輩"}]}'

        issue = service._translation_issue(
            {"start": 0.0, "end": 2.0, "text": "夏希先輩"},
            "夏树学姐",
            service._locked_terms(context),
        )

        self.assertIn("术语未统一", issue)

    def test_总纲使用低温并要求证据(self):
        service = DeepSeekService()
        captured = {}

        def fake_call(messages, *args, **kwargs):
            captured["messages"] = messages
            captured.update(kwargs)
            return "{}"

        service._call_api = fake_call
        service.analyze_full_text("字幕", "测试密钥")

        self.assertEqual(0.2, captured["temperature"])
        self.assertIn("evidence", captured["messages"][0]["content"])


if __name__ == "__main__":
    unittest.main()
