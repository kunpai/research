from __future__ import annotations

from functools import lru_cache
from typing import Any

try:
    import litellm
except ModuleNotFoundError:
    litellm = None


POPULAR_PROVIDER_ORDER = [
    "ollama",
    "openai",
    "anthropic",
    "gemini",
    "vertex_ai",
    "nvidia_nim",
    "openrouter",
    "xai",
    "groq",
    "together_ai",
    "deepinfra",
    "mistral",
    "cohere",
    "huggingface",
    "azure",
    "bedrock",
    "perplexity",
    "github",
]

PROVIDER_METADATA: dict[str, dict[str, Any]] = {
    "ollama": {
        "label": "Ollama",
        "default_api_base": "http://127.0.0.1:11434",
        "requires_api_key": False,
        "api_key_envs": [],
        "api_base_envs": ["OLLAMA_BASE_URL"],
        "api_base_label": "Ollama Base URL",
        "api_base_placeholder": "http://127.0.0.1:11434",
        "model_hint": "Installed local models are suggested automatically. You can still type any model tag manually.",
    },
    "ollama_chat": {
        "label": "Ollama (Chat)",
        "default_api_base": "http://127.0.0.1:11434",
        "requires_api_key": False,
        "api_key_envs": [],
        "api_base_envs": ["OLLAMA_BASE_URL"],
        "api_base_label": "Ollama Base URL",
        "api_base_placeholder": "http://127.0.0.1:11434",
    },
    "openai": {
        "label": "OpenAI",
        "api_key_envs": ["OPENAI_API_KEY"],
        "api_base_envs": ["OPENAI_API_BASE"],
        "api_key_label": "OpenAI API Key",
        "api_base_label": "OpenAI Base URL",
        "api_base_placeholder": "Optional. Leave blank for OpenAI defaults.",
    },
    "anthropic": {
        "label": "Anthropic",
        "api_key_envs": ["ANTHROPIC_API_KEY"],
        "api_key_label": "Anthropic API Key",
        "api_base_label": "Anthropic Base URL",
        "api_base_placeholder": "Optional. Leave blank for Anthropic defaults.",
    },
    "gemini": {
        "label": "Google Gemini",
        "api_key_envs": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "api_key_label": "Google API Key",
        "api_base_label": "Gemini Base URL",
        "api_base_placeholder": "Optional. Leave blank for Gemini defaults.",
    },
    "vertex_ai": {
        "label": "Google Vertex AI",
        "api_key_envs": ["GOOGLE_API_KEY"],
        "api_key_label": "Google API Key (optional)",
        "api_base_label": "Vertex Base URL",
        "api_base_placeholder": "Optional. Vertex AI often uses project/location auth instead of an API key.",
    },
    "nvidia_nim": {
        "label": "NVIDIA NIM",
        "api_key_envs": ["NVIDIA_NIM_API_KEY"],
        "api_base_envs": ["NVIDIA_NIM_API_BASE"],
        "api_key_label": "NVIDIA API Key",
        "api_base_label": "NVIDIA API Base",
        "api_base_placeholder": "Optional. For hosted NVIDIA use https://integrate.api.nvidia.com/v1 or your self-hosted NIM URL.",
        "model_hint": "NVIDIA model names depend on the NIM endpoint. If your deployment is missing from suggestions, type it manually.",
    },
    "openrouter": {
        "label": "OpenRouter",
        "api_key_envs": ["OPENROUTER_API_KEY"],
        "api_key_label": "OpenRouter API Key",
        "api_base_label": "OpenRouter Base URL",
        "api_base_placeholder": "Optional. Leave blank for OpenRouter defaults.",
    },
    "xai": {
        "label": "xAI",
        "api_key_envs": ["XAI_API_KEY"],
        "api_key_label": "xAI API Key",
        "api_base_label": "xAI Base URL",
        "api_base_placeholder": "Optional. Leave blank for xAI defaults.",
    },
    "groq": {
        "label": "Groq",
        "api_key_envs": ["GROQ_API_KEY"],
        "api_key_label": "Groq API Key",
        "api_base_label": "Groq Base URL",
        "api_base_placeholder": "Optional. Leave blank for Groq defaults.",
    },
    "together_ai": {
        "label": "Together AI",
        "api_key_envs": ["TOGETHERAI_API_KEY", "TOGETHER_API_KEY"],
        "api_key_label": "Together API Key",
        "api_base_label": "Together Base URL",
        "api_base_placeholder": "Optional. Leave blank for Together defaults.",
    },
    "deepinfra": {
        "label": "DeepInfra",
        "api_key_envs": ["DEEPINFRA_API_KEY"],
        "api_key_label": "DeepInfra API Key",
        "api_base_label": "DeepInfra Base URL",
        "api_base_placeholder": "Optional. Leave blank for DeepInfra defaults.",
    },
    "mistral": {
        "label": "Mistral",
        "api_key_envs": ["MISTRAL_API_KEY"],
        "api_key_label": "Mistral API Key",
        "api_base_label": "Mistral Base URL",
        "api_base_placeholder": "Optional. Leave blank for Mistral defaults.",
    },
    "cohere": {
        "label": "Cohere",
        "api_key_envs": ["COHERE_API_KEY"],
        "api_key_label": "Cohere API Key",
        "api_base_label": "Cohere Base URL",
        "api_base_placeholder": "Optional. Leave blank for Cohere defaults.",
    },
    "huggingface": {
        "label": "Hugging Face",
        "api_key_envs": ["HUGGINGFACE_API_KEY", "HF_TOKEN"],
        "api_key_label": "Hugging Face API Key",
        "api_base_label": "Inference Endpoint URL",
        "api_base_placeholder": "Set this if you are using a dedicated Hugging Face inference endpoint.",
    },
    "azure": {
        "label": "Azure OpenAI",
        "api_key_envs": ["AZURE_API_KEY"],
        "api_base_envs": ["AZURE_API_BASE"],
        "api_key_label": "Azure API Key",
        "api_base_label": "Azure Base URL",
        "api_base_placeholder": "https://your-resource.openai.azure.com",
    },
    "bedrock": {
        "label": "AWS Bedrock",
        "api_key_envs": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"],
        "api_key_label": "AWS Credentials (optional)",
        "api_base_label": "Bedrock Base URL",
        "api_base_placeholder": "Optional. Most Bedrock setups use AWS credentials rather than a custom base URL.",
    },
    "perplexity": {
        "label": "Perplexity",
        "api_key_envs": ["PERPLEXITYAI_API_KEY", "PERPLEXITY_API_KEY"],
        "api_key_label": "Perplexity API Key",
        "api_base_label": "Perplexity Base URL",
        "api_base_placeholder": "Optional. Leave blank for Perplexity defaults.",
    },
    "github": {
        "label": "GitHub Models",
        "api_key_envs": ["GITHUB_TOKEN"],
        "api_key_label": "GitHub Token",
        "api_base_label": "GitHub Models Base URL",
        "api_base_placeholder": "Optional. Leave blank for GitHub defaults.",
    },
}

POPULAR_MODEL_SUGGESTIONS: dict[str, list[str]] = {
    "openai": ["gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5.4", "gpt-5.4-mini", "o3", "o4-mini"],
    "anthropic": ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1"],
    "gemini": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-3-pro-preview", "gemini-3-flash-preview"],
    "vertex_ai": ["gemini-2.5-flash", "gemini-3-pro-preview", "gemini-3-flash-preview", "claude-sonnet-4-5@20250929"],
    "nvidia_nim": [
        "nvidia/llama-3_3-nemotron-super-49b-v1_5",
        "nvidia/nemotron-3-super-120b-a12b",
        "meta/llama-3.3-70b-instruct",
        "deepseek-ai/deepseek-r1",
        "deepseek-ai/deepseek-v3_1",
    ],
    "openrouter": ["auto", "openai/gpt-5", "anthropic/claude-sonnet-4.5", "google/gemini-2.5-pro", "deepseek/deepseek-r1"],
    "xai": ["grok-4-latest", "grok-4-fast-reasoning", "grok-3-latest"],
    "groq": ["llama-3.3-70b-versatile", "meta-llama/llama-4-maverick-17b-128e-instruct", "qwen/qwen3-32b", "openai/gpt-oss-120b"],
    "together_ai": ["Qwen/Qwen3-235B-A22B-Instruct-2507-tput", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo", "deepseek-ai/DeepSeek-V3.1"],
    "deepinfra": ["deepseek-ai/DeepSeek-V3.1", "Qwen/Qwen3-32B", "meta-llama/Llama-4-Scout-17B-16E-Instruct", "openai/gpt-oss-120b"],
    "mistral": ["mistral-large-latest", "mistral-medium", "codestral-latest", "devstral-small-latest"],
    "cohere": ["command-a-03-2025", "command-r-plus", "command-r7b-12-2024"],
    "huggingface": ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf"],
}

MODEL_ATTRIBUTE_NAMES: dict[str, list[str]] = {
    "openai": ["open_ai_chat_completion_models"],
    "anthropic": ["anthropic_models"],
    "gemini": ["gemini_models"],
    "vertex_ai": [
        "vertex_language_models",
        "vertex_anthropic_models",
        "vertex_llama3_models",
        "vertex_mistral_models",
        "vertex_openai_models",
        "vertex_deepseek_models",
        "vertex_zai_models",
    ],
    "nvidia_nim": ["nvidia_nim_models"],
    "openrouter": ["openrouter_models"],
    "xai": [],
    "groq": ["groq_models"],
    "together_ai": ["together_ai_models"],
    "deepinfra": ["deepinfra_models"],
    "mistral": ["mistral_chat_models"],
    "cohere": ["cohere_models", "cohere_chat_models"],
    "huggingface": ["huggingface_models"],
    "azure": ["azure_models"],
    "bedrock": ["bedrock_models", "bedrock_converse_models"],
    "perplexity": ["perplexity_models"],
    "github": ["github_copilot_models"],
}

DISALLOWED_MODEL_SUBSTRINGS = (
    "embedding",
    "image",
    "audio",
    "transcribe",
    "transcription",
    "tts",
    "moderation",
    "rerank",
    "ocr",
    "video",
    "veo",
    "dall-e",
    "sora",
    "whisper",
    "speech",
    "polly",
)


def provider_names() -> list[str]:
    known = set(_litellm_provider_names()) | set(PROVIDER_METADATA)
    ordered = [provider for provider in POPULAR_PROVIDER_ORDER if provider in known]
    remainder = sorted(known - set(ordered))
    return ordered + remainder


def provider_metadata(provider: str) -> dict[str, Any]:
    normalized = _normalize_provider(provider)
    base = dict(PROVIDER_METADATA.get(normalized, {}))
    label = str(base.get("label") or _humanize_provider(normalized))
    local_provider = normalized in {"ollama", "ollama_chat"}
    metadata = {
        "id": normalized,
        "label": label,
        "default_api_base": str(base.get("default_api_base", "")),
        "requires_api_key": bool(base.get("requires_api_key", not local_provider)),
        "api_key_envs": list(base.get("api_key_envs", ["LLM_API_KEY"])),
        "api_base_envs": list(base.get("api_base_envs", [])),
        "api_key_label": str(base.get("api_key_label", "API Key")),
        "api_base_label": str(base.get("api_base_label", "API Base")),
        "api_base_placeholder": str(
            base.get(
                "api_base_placeholder",
                "Optional. Leave blank unless this provider requires a custom base URL.",
            )
        ),
        "model_hint": str(
            base.get(
                "model_hint",
                "Suggestions come from LiteLLM's local catalog. You can type any supported model manually.",
            )
        ),
    }
    return metadata


def default_api_base(provider: str) -> str:
    return str(provider_metadata(provider).get("default_api_base", ""))


def provider_api_env_overrides(provider: str, api_key: str, api_base: str) -> dict[str, str]:
    metadata = provider_metadata(provider)
    overrides: dict[str, str] = {}
    if api_key.strip():
        overrides["LLM_API_KEY"] = api_key.strip()
        for env_name in metadata.get("api_key_envs", []):
            overrides[str(env_name)] = api_key.strip()
    if api_base.strip():
        overrides["LLM_API_BASE"] = api_base.strip()
        for env_name in metadata.get("api_base_envs", []):
            overrides[str(env_name)] = api_base.strip()
    return overrides


def suggested_models(provider: str, limit: int = 80) -> list[str]:
    normalized = _normalize_provider(provider)
    models: list[str] = []
    seen: set[str] = set()
    for model in POPULAR_MODEL_SUGGESTIONS.get(normalized, []):
        cleaned = _normalize_model_name(model, normalized)
        if cleaned and cleaned not in seen and _is_suggestible_model(cleaned):
            seen.add(cleaned)
            models.append(cleaned)
    for model in _dynamic_models_for_provider(normalized):
        cleaned = _normalize_model_name(model, normalized)
        if cleaned and cleaned not in seen and _is_suggestible_model(cleaned):
            seen.add(cleaned)
            models.append(cleaned)
        if len(models) >= limit:
            break
    return models[:limit]


@lru_cache(maxsize=1)
def _litellm_provider_names() -> tuple[str, ...]:
    names: set[str] = set()
    if litellm is None:
        return tuple()
    provider_list = getattr(litellm, "provider_list", [])
    for item in provider_list:
        value = getattr(item, "value", item)
        if not value:
            continue
        names.add(str(value).strip().lower())
    return tuple(sorted(names))


@lru_cache(maxsize=128)
def _dynamic_models_for_provider(provider: str) -> tuple[str, ...]:
    if litellm is None:
        return tuple()

    models: set[str] = set()
    models_by_provider = getattr(litellm, "models_by_provider", {})
    provider_models = models_by_provider.get(provider)
    models.update(_coerce_string_collection(provider_models))

    for attr_name in MODEL_ATTRIBUTE_NAMES.get(provider, []):
        models.update(_coerce_string_collection(getattr(litellm, attr_name, None)))

    model_cost = getattr(litellm, "model_cost", {})
    if isinstance(model_cost, dict):
        prefix = f"{provider}/"
        for model_name in model_cost:
            if isinstance(model_name, str) and model_name.startswith(prefix):
                models.add(model_name)

    cleaned = sorted(
        (model for model in models if isinstance(model, str) and model.strip()),
        key=_dynamic_model_sort_key,
    )
    return tuple(cleaned)


def _coerce_string_collection(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, dict):
        return {str(key).strip() for key in value if str(key).strip()}
    if isinstance(value, (set, list, tuple)):
        return {str(item).strip() for item in value if str(item).strip()}
    if isinstance(value, str) and value.strip():
        return {value.strip()}
    return set()


def _dynamic_model_sort_key(model: str) -> tuple[int, int, str]:
    lowered = model.lower()
    score = 0
    if "latest" in lowered:
        score -= 4
    if any(token in lowered for token in ("gpt-5", "claude", "gemini", "grok", "qwen", "llama", "deepseek", "mistral", "command")):
        score -= 2
    if "preview" in lowered or "beta" in lowered or "experimental" in lowered:
        score += 1
    return (score, len(model), lowered)


def _normalize_provider(provider: str) -> str:
    return provider.strip().lower()


def _normalize_model_name(model: str, provider: str) -> str:
    cleaned = model.strip()
    prefix = f"{provider}/"
    if cleaned.startswith(prefix):
        return cleaned[len(prefix) :]
    return cleaned


def _is_suggestible_model(model: str) -> bool:
    lowered = model.lower()
    if not lowered:
        return False
    if lowered.startswith(("256-x-", "512-x-", "1024-x-", "1536-x-", "1792-x-")):
        return False
    if any(token in lowered for token in DISALLOWED_MODEL_SUBSTRINGS):
        return False
    if lowered.endswith("-search-api"):
        return False
    return True


def _humanize_provider(provider: str) -> str:
    replacements = {
        "ai": "AI",
        "nim": "NIM",
        "api": "API",
        "llm": "LLM",
        "xai": "xAI",
        "vllm": "vLLM",
    }
    parts = provider.replace("-", " ").replace("_", " ").split()
    if not parts:
        return "Custom"
    words: list[str] = []
    for part in parts:
        lowered = part.lower()
        words.append(replacements.get(lowered, part.capitalize()))
    return " ".join(words)
