from __future__ import annotations

import unittest

from deep_research_ollama.providers import (
    default_api_base,
    provider_api_env_overrides,
    provider_metadata,
    provider_names,
    suggested_models,
)


class ProviderTests(unittest.TestCase):
    def test_provider_names_include_nvidia_nim(self) -> None:
        self.assertIn("nvidia_nim", provider_names())

    def test_provider_metadata_has_nvidia_labels(self) -> None:
        metadata = provider_metadata("nvidia_nim")

        self.assertEqual(metadata["label"], "NVIDIA NIM")
        self.assertEqual(metadata["api_key_label"], "NVIDIA API Key")
        self.assertIn("NVIDIA_NIM_API_KEY", metadata["api_key_envs"])

    def test_provider_overrides_include_generic_and_provider_specific_envs(self) -> None:
        overrides = provider_api_env_overrides(
            "nvidia_nim",
            "secret-key",
            "https://integrate.api.nvidia.com/v1",
        )

        self.assertEqual(overrides["LLM_API_KEY"], "secret-key")
        self.assertEqual(overrides["NVIDIA_NIM_API_KEY"], "secret-key")
        self.assertEqual(overrides["LLM_API_BASE"], "https://integrate.api.nvidia.com/v1")
        self.assertEqual(
            overrides["NVIDIA_NIM_API_BASE"],
            "https://integrate.api.nvidia.com/v1",
        )

    def test_nvidia_nim_suggestions_include_chat_models(self) -> None:
        models = suggested_models("nvidia_nim")

        self.assertIn("deepseek-ai/deepseek-r1", models)
        self.assertIn("meta/llama-3.3-70b-instruct", models)

    def test_openai_suggestions_include_chat_models_not_images(self) -> None:
        models = suggested_models("openai")

        self.assertIn("gpt-5", models)
        self.assertIn("gpt-5-mini", models)
        self.assertNotIn("dall-e-3", models)

    def test_default_api_base_only_filled_for_ollama(self) -> None:
        self.assertEqual(default_api_base("ollama"), "http://127.0.0.1:11434")
        self.assertEqual(default_api_base("openai"), "")


if __name__ == "__main__":
    unittest.main()
