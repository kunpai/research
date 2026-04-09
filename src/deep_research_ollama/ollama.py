from __future__ import annotations

import json
import socket
from typing import Any
from urllib import error, request

from deep_research_ollama.config import Settings


class OllamaError(RuntimeError):
    pass


class OllamaClient:
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
        payload = {
            "model": self.settings.ollama_model,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": temperature},
        }
        if schema is not None:
            payload["format"] = schema
        elif json_mode:
            payload["format"] = "json"

        endpoint = f"{self.settings.ollama_base_url.rstrip('/')}/api/chat"
        req = request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(
                req, timeout=self.settings.request_timeout_seconds
            ) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (TimeoutError, socket.timeout) as exc:
            raise OllamaError(
                f"Ollama request timed out after {self.settings.request_timeout_seconds} seconds."
            ) from exc
        except error.URLError as exc:
            raise OllamaError(
                f"Failed to contact Ollama at {self.settings.ollama_base_url}: {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise OllamaError("Ollama returned invalid JSON.") from exc

        content = (
            data.get("message", {}).get("content")
            if isinstance(data, dict)
            else None
        )
        if not content:
            raise OllamaError("Ollama returned an empty response.")
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

    @staticmethod
    def _load_json_loose(raw: str) -> dict:
        for candidate in (raw, OllamaClient._extract_braced(raw), OllamaClient._extract_list(raw)):
            if not candidate:
                continue
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed
        raise OllamaError("Model response was not valid JSON.")

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
            if not any(OllamaClient._matches_type(value, item) for item in schema_type):
                raise OllamaError(
                    f"Model response at {path} did not match any allowed schema types: {schema_type}."
                )
        elif schema_type and not OllamaClient._matches_type(value, schema_type):
            raise OllamaError(
                f"Model response at {path} expected type {schema_type}, got {type(value).__name__}."
            )

        enum_values = schema.get("enum")
        if enum_values is not None and value not in enum_values:
            raise OllamaError(f"Model response at {path} contained unsupported value {value!r}.")

        if schema_type == "object":
            if not isinstance(value, dict):
                raise OllamaError(f"Model response at {path} expected an object.")
            properties = schema.get("properties", {})
            required = schema.get("required", [])
            for key in required:
                if key not in value:
                    raise OllamaError(f"Model response at {path} is missing required key {key!r}.")
            if schema.get("additionalProperties") is False:
                extras = sorted(key for key in value if key not in properties)
                if extras:
                    raise OllamaError(
                        f"Model response at {path} had unexpected keys: {', '.join(extras)}."
                    )
            for key, subschema in properties.items():
                if key in value:
                    OllamaClient._validate_schema(value[key], subschema, f"{path}.{key}")
            return

        if schema_type == "array":
            if not isinstance(value, list):
                raise OllamaError(f"Model response at {path} expected an array.")
            max_items = schema.get("maxItems")
            if isinstance(max_items, int) and len(value) > max_items:
                raise OllamaError(
                    f"Model response at {path} exceeded maxItems={max_items}."
                )
            item_schema = schema.get("items")
            if isinstance(item_schema, dict):
                for index, item in enumerate(value):
                    OllamaClient._validate_schema(item, item_schema, f"{path}[{index}]")

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
