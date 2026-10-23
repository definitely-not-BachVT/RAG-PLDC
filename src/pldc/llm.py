"""Chat LLM with provider fallback across the configured Gemini models.

Fallback (each configured Gemini model in turn) happens on "provider is unavailable" errors (auth,
rate limit, server errors, timeouts, connection errors). Errors caused by our own
request (bad request) are raised so they are not hidden by switching providers.
When streaming, we only fall back before the first token arrives.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass
from typing import Iterator, Protocol

import httpx
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from .config import Settings, get_settings

log = logging.getLogger(__name__)

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": str}


class Provider(Protocol):
    name: str

    def stream(self, messages: list[Message], temperature: float) -> Iterator[str]: ...

    def should_fall_back(self, exc: Exception) -> bool: ...


class GeminiProvider:
    _FALLBACK_STATUS = {401, 403, 404, 408, 429}

    def __init__(self, api_key: str, model: str, timeout: float, thinking_budget: int | None = None):
        # Flash models "think" by default: ~55 s before the first token on a RAG prompt, which
        # blows the timeout. A budget of 0 turns it off. Lite models reject the setting (400).
        self.thinking = (genai_types.ThinkingConfig(thinking_budget=thinking_budget)
                         if thinking_budget is not None and "lite" not in model else None)
        self.client = genai.Client(
            api_key=api_key, http_options=genai_types.HttpOptions(timeout=int(timeout * 1000)))
        self.model = model
        self.name = f"Gemini ({model})"

    def stream(self, messages: list[Message], temperature: float) -> Iterator[str]:
        system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
        contents = [
            genai_types.Content(role="model" if m["role"] == "assistant" else "user",
                                parts=[genai_types.Part(text=m["content"])])
            for m in messages if m["role"] != "system"
        ]
        config = genai_types.GenerateContentConfig(
            system_instruction=system or None, temperature=temperature,
            thinking_config=self.thinking)
        for chunk in self.client.models.generate_content_stream(
                model=self.model, contents=contents, config=config):
            if chunk.text:
                yield chunk.text

    def should_fall_back(self, exc: Exception) -> bool:
        if isinstance(exc, genai_errors.APIError):
            return exc.code in self._FALLBACK_STATUS or (exc.code or 0) >= 500
        # The google-genai SDK surfaces network failures as raw httpx errors (e.g. ReadTimeout).
        return isinstance(exc, (TimeoutError, ConnectionError, httpx.TimeoutException, httpx.TransportError))


class AllProvidersFailed(RuntimeError):
    pass


@dataclass
class StreamResult:
    provider: str
    chunks: Iterator[str]


class FallbackLLM:
    def __init__(self, providers: list[Provider]):
        if not providers:
            raise AllProvidersFailed("Chưa cấu hình API key nào (GOOGLE_API_KEY).")
        self.providers = providers

    @property
    def provider_names(self) -> list[str]:
        return [p.name for p in self.providers]

    def stream(self, messages: list[Message], temperature: float = 0.2) -> StreamResult:
        """Start streaming from the first provider that produces a token."""
        errors = []
        for provider in self.providers:
            it = provider.stream(messages, temperature)
            try:
                first = next(it, "")
            except Exception as exc:  # noqa: BLE001 - classified below
                if not provider.should_fall_back(exc):
                    raise
                log.warning("%s failed, falling back: %s", provider.name, exc)
                errors.append(f"{provider.name}: {type(exc).__name__}: {exc}")
                continue
            return StreamResult(provider.name, itertools.chain([first], it))
        raise AllProvidersFailed("Không gọi được LLM nào.\n" + "\n".join(errors))

    def complete(self, messages: list[Message], temperature: float = 0.2) -> tuple[str, str]:
        """Non-streaming convenience: returns (text, provider name)."""
        res = self.stream(messages, temperature)
        return "".join(res.chunks), res.provider


def build_llm(settings: Settings | None = None) -> FallbackLLM:
    s = settings or get_settings()
    providers: list[Provider] = []
    if s.google_api_key:
        # Several Gemini models: one is often overloaded (503) while another still answers.
        providers += [GeminiProvider(s.google_api_key, m, s.llm_timeout, s.gemini_thinking_budget)
                      for m in s.gemini_models]
    return FallbackLLM(providers)
