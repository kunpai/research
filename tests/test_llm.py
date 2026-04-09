from __future__ import annotations

import unittest
from unittest.mock import patch

from deep_research_ollama.config import Settings
from deep_research_ollama.llm import LLMClient, LLMError


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class LLMClientTests(unittest.TestCase):
    def test_chat_json_reports_missing_litellm_cleanly(self) -> None:
        with patch("deep_research_ollama.llm.completion", None):
            client = LLMClient(Settings(llm_provider="openai", llm_model="gpt-4o-mini"))
            with self.assertRaises(LLMError) as exc:
                client.chat_json(
                    "system",
                    "user",
                    schema={
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                )

        self.assertIn("LiteLLM is not installed", str(exc.exception))

    def test_chat_json_sends_json_schema_response_format(self) -> None:
        captured: dict[str, object] = {}
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _FakeResponse('{"answer":"ok"}')

        with (
            patch("deep_research_ollama.llm.get_supported_openai_params", return_value=["response_format"]),
            patch("deep_research_ollama.llm.completion", side_effect=fake_completion),
        ):
            client = LLMClient(Settings(llm_provider="openai", llm_model="gpt-4o-mini"))
            payload = client.chat_json("system", "user", schema=schema)

        self.assertEqual(payload, {"answer": "ok"})
        self.assertEqual(captured["model"], "openai/gpt-4o-mini")
        self.assertEqual(
            captured["response_format"],
            {
                "type": "json_schema",
                "json_schema": {
                    "name": "deep_research_schema",
                    "schema": schema,
                    "strict": True,
                },
            },
        )

    def test_chat_json_uses_provider_prefix_and_api_key(self) -> None:
        captured: dict[str, object] = {}

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return _FakeResponse('{"answer":"ok"}')

        with (
            patch("deep_research_ollama.llm.get_supported_openai_params", return_value=["response_format"]),
            patch("deep_research_ollama.llm.completion", side_effect=fake_completion),
        ):
            client = LLMClient(
                Settings(
                    llm_provider="gemini",
                    llm_model="gemini-2.5-flash",
                    llm_api_key="secret-key",
                )
            )
            client.chat_json(
                "system",
                "user",
                schema={
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
            )

        self.assertEqual(captured["model"], "gemini/gemini-2.5-flash")
        self.assertEqual(captured["api_key"], "secret-key")

    def test_chat_json_rejects_schema_invalid_response(self) -> None:
        schema = {
            "type": "object",
            "additionalProperties": False,
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }

        with (
            patch("deep_research_ollama.llm.get_supported_openai_params", return_value=["response_format"]),
            patch("deep_research_ollama.llm.completion", return_value=_FakeResponse('{"answer":123}')),
        ):
            client = LLMClient(Settings(llm_provider="openai", llm_model="gpt-4o-mini"))
            with self.assertRaises(LLMError):
                client.chat_json("system", "user", schema=schema)


if __name__ == "__main__":
    unittest.main()
