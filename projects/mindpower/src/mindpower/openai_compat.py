from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


DEFAULT_NRP_PROVIDER_CONFIG = {
    "id": "custom-provider-963ccbe7-7e74-4dbe-beda-8054a6590245",
    "name": "NRP",
    "type": "openai",
    "settings": {
        "apiHost": "https://ellm.nrp-nautilus.io",
        "models": [
            {"modelId": "qwen3", "capabilities": ["reasoning", "vision", "tool_use"], "contextWindow": 1010000},
            {"modelId": "qwen3-small", "capabilities": ["reasoning", "vision", "tool_use"], "contextWindow": 1010000},
            {"modelId": "gpt-oss", "capabilities": ["reasoning", "tool_use"], "contextWindow": 131072},
            {"modelId": "gemma", "capabilities": ["reasoning", "vision", "tool_use"], "contextWindow": 262144},
            {"modelId": "kimi", "capabilities": ["reasoning", "vision", "tool_use"], "contextWindow": 262144},
            {"modelId": "glm-4.7", "capabilities": ["reasoning", "tool_use"], "contextWindow": 202752},
            {"modelId": "minimax-m2", "capabilities": ["reasoning", "tool_use"], "contextWindow": 196608},
            {"modelId": "olmo", "capabilities": ["tool_use"], "contextWindow": 65536},
            {"modelId": "qwen3-27b", "capabilities": ["reasoning", "vision", "tool_use"], "contextWindow": 262144},
        ],
    },
}

DEFAULT_NRP_MODEL = "Qwen/Qwen3.5-397B-A17B-FP8"
DEFAULT_NRP_REQUEST_MODEL = "qwen3"


def _normalize_base_url(api_host: str) -> str:
    host = api_host.rstrip("/")
    if host.endswith("/v1"):
        return host
    return host + "/v1"


def _resolve_request_model(provider: "OpenAIProviderConfig", model: str) -> str:
    available_model_ids = {entry.get("modelId") for entry in provider.models if isinstance(entry, dict)}
    if model in available_model_ids:
        return model
    if provider.name.lower() == "nrp" and model == DEFAULT_NRP_MODEL and DEFAULT_NRP_REQUEST_MODEL in available_model_ids:
        return DEFAULT_NRP_REQUEST_MODEL
    return model


def _extract_json_block(text: str) -> Optional[dict[str, Any]]:
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


@dataclass
class OpenAIProviderConfig:
    provider_id: str
    name: str
    provider_type: str
    api_host: str
    api_key: Optional[str] = None
    models: list[dict[str, Any]] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return _normalize_base_url(self.api_host)

    @classmethod
    def from_dict(cls, raw: dict[str, Any], api_key: str | None = None) -> "OpenAIProviderConfig":
        settings = raw.get("settings", {})
        return cls(
            provider_id=raw.get("id", "custom-openai-provider"),
            name=raw.get("name", "Custom OpenAI Provider"),
            provider_type=raw.get("type", "openai"),
            api_host=settings.get("apiHost", ""),
            api_key=api_key or settings.get("apiKey"),
            models=settings.get("models", []),
        )

    @classmethod
    def from_json_file(cls, path: str | Path, api_key: str | None = None) -> "OpenAIProviderConfig":
        with Path(path).open() as f:
            raw = json.load(f)
        return cls.from_dict(raw, api_key=api_key)


class OpenAICompatibleWrapper:
    def __init__(
        self,
        provider: OpenAIProviderConfig,
        *,
        model: str = DEFAULT_NRP_MODEL,
        api_key: str | None = None,
        enable_thinking: bool = False,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "The `openai` package is not installed. Install it with "
                "`pip install openai` or enable the project's annotation dependencies."
            )
        resolved_api_key = api_key or provider.api_key or os.environ.get("MINDPOWER_NRP_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            raise ValueError(
                "Missing API key. Pass --api_key, set MINDPOWER_NRP_API_KEY, or provide a provider config containing apiKey."
            )
        self.provider = provider
        self.model = model
        self.request_model = _resolve_request_model(provider, model)
        self.enable_thinking = enable_thinking
        self.client = OpenAI(api_key=resolved_api_key, base_url=provider.base_url)

    def create_chat_completion(
        self,
        *,
        messages: list[dict[str, Any]],
        response_format: dict[str, Any] | None = None,
        max_completion_tokens: int = 4000,
        temperature: float = 0.2,
        extra_body: dict[str, Any] | None = None,
    ):
        request_kwargs = {
            "model": self.request_model,
            "messages": messages,
            "max_completion_tokens": max_completion_tokens,
            "temperature": temperature,
        }
        if response_format is not None:
            request_kwargs["response_format"] = response_format
        merged_extra_body = {}
        if not self.enable_thinking:
            merged_extra_body["chat_template_kwargs"] = {"enable_thinking": False}
        if extra_body:
            merged_extra_body.update(extra_body)
        if merged_extra_body:
            request_kwargs["extra_body"] = merged_extra_body
        return self.client.chat.completions.create(**request_kwargs)

    def create_plain_json(
        self,
        *,
        messages: list[dict[str, Any]],
        max_completion_tokens: int = 4000,
        temperature: float = 0.2,
        extra_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        response = self.create_chat_completion(
            messages=messages,
            response_format=None,
            max_completion_tokens=max_completion_tokens,
            temperature=temperature,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content or ""
        parsed = _extract_json_block(content)
        if parsed is None:
            raise ValueError("Plain JSON response did not contain a valid JSON object.")
        return parsed

    def create_structured_json(
        self,
        *,
        messages: list[dict[str, Any]],
        json_schema: dict[str, Any],
        max_completion_tokens: int = 4000,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        try:
            response = self.create_chat_completion(
                messages=messages,
                response_format=json_schema,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            parsed = _extract_json_block(content)
            if parsed is None:
                raise ValueError("Structured response did not contain valid JSON.")
            return parsed
        except Exception as first_error:
            fallback_messages = list(messages)
            fallback_messages.append(
                {
                    "role": "system",
                    "content": (
                        "Return valid JSON only. Do not include markdown fences, commentary, or any text "
                        "outside a single JSON object matching the requested schema."
                    ),
                }
            )
            response = self.create_chat_completion(
                messages=fallback_messages,
                response_format=None,
                max_completion_tokens=max_completion_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            parsed = _extract_json_block(content)
            if parsed is None:
                raise RuntimeError(f"Failed to parse JSON after structured-output fallback: {first_error}") from first_error
            return parsed
