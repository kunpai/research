from __future__ import annotations

import json
from typing import Any

try:
    from litellm import completion, get_supported_openai_params
    _LITELLM_IMPORT_ERROR: Exception | None = None
except ModuleNotFoundError as exc:
    completion = None
    get_supported_openai_params = None
    _LITELLM_IMPORT_ERROR = exc

from deep_research_ollama.config import Settings


class LLMError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def chat_text(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.2,
        json_mode: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> str:
        self._require_litellm()
        resolved_model = self.settings.resolved_model()
        if not resolved_model:
            raise LLMError("No model configured. Set LLM_MODEL or choose a model in the GUI.")

        kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "timeout": self.settings.request_timeout_seconds,
        }
        if self.settings.llm_api_base.strip():
            kwargs["base_url"] = self.settings.llm_api_base.strip()
        if self.settings.llm_api_key:
            kwargs["api_key"] = self.settings.llm_api_key

        response_format = self._response_format(
            resolved_model,
            json_mode=json_mode,
            schema=schema,
        )
        if response_format is not None:
            kwargs["response_format"] = response_format

        try:
            response = completion(**kwargs)
        except Exception as exc:
            if response_format is not None and self._should_retry_without_response_format(exc):
                kwargs.pop("response_format", None)
                try:
                    response = completion(**kwargs)
                except Exception as retry_exc:
                    raise LLMError(f"LiteLLM request failed: {retry_exc}") from retry_exc
            else:
                raise LLMError(f"LiteLLM request failed: {exc}") from exc

        content = self._extract_text(response)
        if not content:
            raise LLMError("Model returned an empty response.")
        return content.strip()

    def chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        schema: dict[str, Any] | None = None,
    ) -> dict:
        raw = self.chat_text(
            system_prompt,
            user_prompt,
            temperature=temperature,
            json_mode=schema is None,
            schema=schema,
        )
        parsed = self._load_json_loose(raw)
        if schema is not None:
            self._validate_schema(parsed, schema)
        return parsed

    def _response_format(
        self,
        resolved_model: str,
        *,
        json_mode: bool,
        schema: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not json_mode and schema is None:
            return None
        if not self._supports_response_format(resolved_model):
            return None
        if schema is None:
            return {"type": "json_object"}
        return {
            "type": "json_schema",
            "json_schema": {
                "name": "deep_research_schema",
                "schema": schema,
                "strict": True,
            },
        }

    @staticmethod
    def _should_retry_without_response_format(exc: Exception) -> bool:
        text = str(exc).lower()
        return "response_format" in text or "json_schema" in text or "json object" in text

    @staticmethod
    def _extract_text(response: Any) -> str:
        choices = getattr(response, "choices", None)
        if choices is None and isinstance(response, dict):
            choices = response.get("choices")
        if not choices:
            return ""
        first = choices[0]
        message = getattr(first, "message", None)
        if message is None and isinstance(first, dict):
            message = first.get("message")
        if message is None:
            return ""
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if isinstance(item, dict):
                    text = item.get("text")
                    if isinstance(text, str):
                        parts.append(text)
                        continue
                text = getattr(item, "text", None)
                if isinstance(text, str):
                    parts.append(text)
            return "\n".join(part for part in parts if part)
        return ""

    @staticmethod
    def _load_json_loose(raw: str) -> dict:
        for candidate in (
            raw,
            LLMClient._extract_braced(raw),
            LLMClient._extract_list(raw),
        ):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise LLMError("Model response was not valid JSON.")

    @staticmethod
    def _extract_braced(raw: str) -> str:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return ""
        return raw[start : end + 1]

    @staticmethod
    def _extract_list(raw: str) -> str:
        start = raw.find("[")
        end = raw.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return ""
        return '{"items": ' + raw[start : end + 1] + "}"

    @staticmethod
    def _validate_schema(value: Any, schema: dict[str, Any], path: str = "$") -> None:
        schema_type = schema.get("type")
        if isinstance(schema_type, list):
            if not any(LLMClient._matches_type(value, item) for item in schema_type):
                raise LLMError(
                    f"Model response at {path} did not match any allowed schema types: {schema_type}."
                )
        elif schema_type and not LLMClient._matches_type(value, schema_type):
            raise LLMError(
                f"Model response at {path} expected type {schema_type}, got {type(value).__name__}."
            )

        enum_values = schema.get("enum")
        if enum_values is not None and value not in enum_values:
            raise LLMError(f"Model response at {path} contained unsupported value {value!r}.")

        if schema_type == "object":
            if not isinstance(value, dict):
                raise LLMError(f"Model response at {path} expected an object.")
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            for key in required:
                if key not in value:
                    raise LLMError(f"Model response at {path} is missing required key {key!r}.")
            if schema.get("additionalProperties") is False:
                extras = sorted(key for key in value if key not in properties)
                if extras:
                    raise LLMError(
                        f"Model response at {path} had unexpected keys: {', '.join(extras)}."
                    )
            for key, subschema in properties.items():
                if key in value:
                    LLMClient._validate_schema(value[key], subschema, f"{path}.{key}")
            return

        if schema_type == "array":
            if not isinstance(value, list):
                raise LLMError(f"Model response at {path} expected an array.")
            max_items = schema.get("maxItems")
            if isinstance(max_items, int) and len(value) > max_items:
                raise LLMError(f"Model response at {path} exceeded maxItems={max_items}.")
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    LLMClient._validate_schema(item, item_schema, f"{path}[{index}]")

    @staticmethod
    def _matches_type(value: Any, schema_type: str) -> bool:
        if schema_type == "object":
            return isinstance(value, dict)
        if schema_type == "array":
            return isinstance(value, list)
        if schema_type == "string":
            return isinstance(value, str)
        if schema_type == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if schema_type == "number":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if schema_type == "boolean":
            return isinstance(value, bool)
        if schema_type == "null":
            return value is None
        return True

    @staticmethod
    def _supports_response_format(model: str) -> bool:
        if get_supported_openai_params is None:
            return False
        try:
            params = get_supported_openai_params(model=model)
        except Exception:
            return False
        return "response_format" in (params or [])

    @staticmethod
    def _require_litellm() -> None:
        if completion is not None:
            return
        raise LLMError(
            "LiteLLM is not installed in the current environment. "
            "Reinstall the project dependencies with `pip install -e .` "
            "or install `litellm` directly."
        ) from _LITELLM_IMPORT_ERROR
