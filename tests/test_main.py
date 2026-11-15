"""Unit tests for main.py: wire translation and routing rules.

Standard library only. Run from the repo root:
    python3 -m unittest discover -s tests -v
No network, no env: every test works on canned bytes and dicts.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main


def _openai_stream():
    return (
        b'data: {"id":"chatcmpl-sim","object":"chat.completion.chunk","created":1767225600,'
        b'"model":"x","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}]}\n\n'
        b'data: {"id":"chatcmpl-sim","object":"chat.completion.chunk","created":1767225600,'
        b'"model":"x","choices":[{"index":0,"delta":{"content":"ANSWER ECHO"},"finish_reason":null}]}\n\n'
        b'data: {"id":"chatcmpl-sim","object":"chat.completion.chunk","created":1767225600,'
        b'"model":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        b'data: {"id":"chatcmpl-sim","object":"chat.completion.chunk","created":1767225600,'
        b'"model":"x","choices":[],"usage":{"prompt_tokens":36,"completion_tokens":46,"total_tokens":82}}\n\n'
        b'data: [DONE]\n\n'
    )


class TestAnthropicSseFromOpenAI(unittest.TestCase):
    def test_usage_chunk_after_finish_chunk_is_reported(self):
        out = main._anthropic_sse_from_openai(_openai_stream(), "auto")
        self.assertIsNotNone(out)
        events = []
        for frame in out.decode().split("\n\n"):
            if frame.startswith("event:"):
                name, data = frame.split("\ndata: ", 1)
                events.append((name[len("event: "):], json.loads(data)))
        by_name = {name: data for name, data in events}
        self.assertEqual(by_name["message_start"]["message"]["usage"]["input_tokens"], 36)
        self.assertEqual(by_name["message_delta"]["usage"]["output_tokens"], 46)
        self.assertEqual(by_name["message_delta"]["delta"]["stop_reason"], "end_turn")
        text = "".join(
            data["delta"]["text"]
            for name, data in events
            if name == "content_block_delta" and data["delta"].get("type") == "text_delta"
        )
        self.assertEqual(text, "ANSWER ECHO")
        self.assertEqual([n for n, _ in events][-1], "message_stop")

    def test_anthropic_to_openai_stream_asks_for_usage(self):
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 10,
            "stream": True,
        }
        payload = main._openai_body_from_anthropic(body, "some-model", True)
        self.assertEqual(payload["stream_options"], {"include_usage": True})
        self.assertEqual(payload["model"], "some-model")


class TestGeminiTranslation(unittest.TestCase):
    def test_openai_json_request_sets_json_mime_type(self):
        body = {
            "model": "auto",
            "messages": [{"role": "user", "content": "hi"}],
            "response_format": {"type": "json_object"},
            "max_tokens": 10,
        }
        payload = main._gemini_body_from_openai(body, False)
        self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")
        self.assertEqual(payload["generationConfig"]["maxOutputTokens"], 10)

    def test_gemini_stream_to_openai_carries_usage(self):
        raw = (
            b'data: {"candidates":[{"content":{"parts":[{"text":"ANSWER "}],"role":"model"},"index":0}]}\n\n'
            b'data: {"candidates":[{"content":{"parts":[{"text":"ECHO"}],"role":"model"},"finishReason":"STOP","index":0}],'
            b'"usageMetadata":{"promptTokenCount":20,"candidatesTokenCount":5,"totalTokenCount":25},"modelVersion":"m"}\n\n'
        )
        out = main._openai_sse_from_gemini(raw, "gemini-3.8-flash")
        self.assertIsNotNone(out)
        text = out.decode()
        joined = "".join(
            json.loads(frame[len("data: "):])["choices"][0]["delta"].get("content", "")
            for frame in text.split("\n\n")
            if frame.startswith("data: {")
        )
        self.assertEqual(joined, "ANSWER ECHO")
        self.assertIn('"finish_reason":"stop"', text)
        self.assertIn('"prompt_tokens":20', text)
        self.assertIn('"completion_tokens":5', text)
        self.assertTrue(text.rstrip().endswith("data: [DONE]"))


class TestWireBodies(unittest.TestCase):
    def test_openai_image_url_to_anthropic_base64(self):
        body = {
            "model": "auto",
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": "look"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
                ],
            }],
        }
        payload = main._anthropic_body_from_openai(body, "m", False)
        image = payload["messages"][0]["content"][1]
        self.assertEqual(image["type"], "image")
        self.assertEqual(image["source"], {"type": "base64", "media_type": "image/png", "data": "AAAA"})

    def test_openai_json_from_anthropic_keeps_usage(self):
        body = {
            "id": "msg_sim",
            "content": [{"type": "text", "text": "ANSWER ECHO"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 57, "output_tokens": 39},
        }
        out = main._openai_json_from_anthropic(body, "auto")
        self.assertEqual(out["usage"], {"prompt_tokens": 57, "completion_tokens": 39, "total_tokens": 96})
        self.assertEqual(out["choices"][0]["finish_reason"], "stop")


class TestRouting(unittest.TestCase):
    def _aggregator(self):
        brief = {
            "providers": [],
            "ratings": {
                "ratings": [
                    {"model": "top-novis", "provider": "p1", "rating_milli": 1475,
                     "context_length": 204800, "tools": True, "vision": False, "json": True, "free": True},
                    {"model": "vis", "provider": "p2", "rating_milli": 1458,
                     "context_length": 1048576, "tools": True, "vision": True, "json": True, "free": True},
                ],
                "best": [],
            },
            "workload": {
                "manifest": [
                    {"id": "r01", "api": "openai", "stream": False, "weight": 2, "requires": {"tools": True, "vision": True}},
                    {"id": "r02", "api": "openai", "stream": False, "weight": 2, "requires": {"json": True}},
                ]
            },
        }
        probes = [
            {"provider": {"id": "p1", "wire": "openai", "auth": "bearer", "base": "/p/p1", "key_env": "P1", "key": "k1"},
             "ok": True,
             "models": [{"id": "top-novis", "free": True, "context_length": 204800, "tools": True, "vision": False, "json": True}]},
            {"provider": {"id": "p2", "wire": "gemini", "auth": "x-goog-api-key", "base": "/p/p2", "key_env": "P2", "key": "k2"},
             "ok": True,
             "models": [{"id": "vis", "free": True, "context_length": 1048576, "tools": True, "vision": True, "json": True}]},
        ]
        return main.Aggregator(brief, probes)

    def test_manifest_row_matches_arriving_request(self):
        agg = self._aggregator()
        # The first arriving request is r01 (tools+vision): its row, not r02's.
        self.assertEqual(agg.requirements_for("openai", {}, 0), {"tools": True, "vision": True})
        self.assertEqual(agg.requirements_for("openai", {}, 1), {"json": True})

    def test_vision_request_excludes_non_vision_top_model(self):
        agg = self._aggregator()
        candidates = agg.candidates_for("openai", {"tools": True, "vision": True}, False)
        self.assertEqual([c["model_id"] for c in candidates], ["vis"])
        candidates = agg.candidates_for("openai", {"json": True}, False)
        self.assertEqual([c["model_id"] for c in candidates], ["top-novis", "vis"])

    def test_json_request_excludes_anthropic_wire(self):
        agg = self._aggregator()
        agg.probes[1]["provider"]["wire"] = "anthropic"
        candidates = agg.candidates_for("openai", {"json": True}, True)
        self.assertEqual([c["model_id"] for c in candidates], ["top-novis"])


class TestRetrySameCandidateOnce(unittest.TestCase):
    """Reproduction for the quality-99 miss (trial 0 r04, trial 1 r11,
    trial 2 r02/r04/r12): a provider's refusal that is a one-off fail_call
    must be retried on the SAME candidate before falling back, because the
    env refuses by call number — the next call number usually succeeds.

    Trial 0 evidence (analysis/quality_99.py): openrouter call#2 is in
    fail_calls=[2, 10] and call#3 succeeded, but the entry fell back to
    cloudflare glm-4.7-flash (1470) and lost 5 rating points x weight 2.
    """

    def _retry_aggregator(self):
        brief = {
            "providers": [],
            "ratings": {
                "ratings": [
                    {"model": "top", "provider": "p1", "rating_milli": 1475,
                     "context_length": 204800, "tools": True, "vision": False, "json": True, "free": True},
                    {"model": "second", "provider": "p2", "rating_milli": 1470,
                     "context_length": 204800, "tools": True, "vision": False, "json": True, "free": True},
                ],
                "best": [],
            },
            "workload": {
                "manifest": [
                    {"id": "r01", "api": "openai", "stream": False, "weight": 2, "requires": {}},
                ]
            },
        }
        probes = [
            {"provider": {"id": "p1", "wire": "openai", "auth": "bearer", "base": "/p/p1", "key_env": "P1", "key": "k1"},
             "ok": True,
             "models": [{"id": "top", "free": True, "context_length": 204800, "tools": True, "vision": False, "json": True}]},
            {"provider": {"id": "p2", "wire": "openai", "auth": "bearer", "base": "/p/p2", "key_env": "P2", "key": "k2"},
             "ok": True,
             "models": [{"id": "second", "free": True, "context_length": 204800, "tools": True, "vision": False, "json": True}]},
        ]
        return main.Aggregator(brief, probes)

    @staticmethod
    def _ok_body(model):
        return json.dumps({
            "id": "chatcmpl-x", "object": "chat.completion", "created": 1, "model": model,
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }).encode("utf-8")

    def _serve_once(self, agg, script):
        calls = {"p1": 0, "p2": 0}
        real_call = main.call_upstream

        def fake(url, headers, payload):
            pid = "p1" if "/p/p1/" in url else "p2"
            calls[pid] += 1
            outcome = script[pid][calls[pid] - 1] if calls[pid] - 1 < len(script[pid]) else "ok"
            if outcome == "fail":
                return 429, b'{"error":{"message":"rate limited"}}', "HTTP 429"
            return 200, self._ok_body("top" if pid == "p1" else "second"), None

        main.call_upstream = fake
        try:
            status, payload, _ctype = agg.serve("openai", {
                "model": "auto",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            })
        finally:
            main.call_upstream = real_call
        return status, payload, calls

    def test_one_off_refusal_is_retried_on_same_candidate(self):
        agg = self._retry_aggregator()
        status, payload, calls = self._serve_once(agg, {"p1": ["fail", "ok"], "p2": ["ok"]})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["model"], "top")
        self.assertEqual(calls, {"p1": 2, "p2": 0})

    def test_permanent_refusal_falls_back_after_exactly_one_retry(self):
        agg = self._retry_aggregator()
        status, payload, calls = self._serve_once(agg, {"p1": ["fail", "fail"], "p2": ["ok"]})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["model"], "second")
        self.assertEqual(calls, {"p1": 2, "p2": 1})

    def test_success_first_try_makes_no_extra_call(self):
        agg = self._retry_aggregator()
        status, payload, calls = self._serve_once(agg, {"p1": ["ok"], "p2": ["ok"]})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(payload)["model"], "top")
        self.assertEqual(calls, {"p1": 1, "p2": 0})


if __name__ == "__main__":
    unittest.main()
