from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from deep_research_ollama.config import Settings
from deep_research_ollama.ollama import OllamaClient, OllamaError


class _FakeHTTPResponse:
    def __init__(self, payload: dict) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeHTTPResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class OllamaClientTests(unittest.TestCase):
    def test_chat_json_sends_schema_object_in_format(self) -> None:
        captured: dict[str, object] = {}
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }

        def fake_urlopen(req, timeout=0):
            captured["payload"] = json.loads(req.data.decode("utf-8"))
            return _FakeHTTPResponse({"message": {"content": '{"answer":"ok"}'}})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client = OllamaClient(Settings())
            payload = client.chat_json("system", "user", schema=schema)

        self.assertEqual(payload, {"answer": "ok"})
        request_payload = captured["payload"]
        self.assertIsInstance(request_payload, dict)
        self.assertEqual(request_payload["format"], schema)

    def test_chat_json_rejects_schema_invalid_response(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }

        def fake_urlopen(req, timeout=0):
            return _FakeHTTPResponse({"message": {"content": '{"answer":123}'}})

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            client = OllamaClient(Settings())
            with self.assertRaises(OllamaError):
                client.chat_json("system", "user", schema=schema)


if __name__ == "__main__":
    unittest.main()
