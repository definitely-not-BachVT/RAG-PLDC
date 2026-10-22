"""Providers are tried in order (Gemini models); fallback must happen only on availability errors."""
import httpx
import pytest

from pldc.config import get_settings
from pldc.llm import AllProvidersFailed, FallbackLLM, GeminiProvider, build_llm


class FakeProvider:
    def __init__(self, name, chunks=None, error=None, fall_back=True):
        self.name, self.chunks, self.error, self.fall_back = name, chunks or [], error, fall_back
        self.calls = 0

    def stream(self, messages, temperature):
        self.calls += 1
        if self.error:
            raise self.error
        yield from self.chunks

    def should_fall_back(self, exc):
        return self.fall_back


MSGS = [{"role": "user", "content": "hi"}]


def test_primary_answers_when_healthy():
    a, b = FakeProvider("A", ["x", "y"]), FakeProvider("B", ["z"])
    text, provider = FallbackLLM([a, b]).complete(MSGS)
    assert (text, provider, b.calls) == ("xy", "A", 0)


def test_falls_back_on_availability_error():
    a, b = FakeProvider("A", error=ConnectionError("down")), FakeProvider("B", ["ok"])
    assert FallbackLLM([a, b]).complete(MSGS) == ("ok", "B")


def test_does_not_hide_request_errors():
    a = FakeProvider("A", error=ValueError("bad request"), fall_back=False)
    with pytest.raises(ValueError):
        FallbackLLM([a, FakeProvider("B", ["ok"])]).complete(MSGS)


def test_all_failed():
    with pytest.raises(AllProvidersFailed):
        FallbackLLM([FakeProvider("A", error=ConnectionError())]).complete(MSGS)


def test_no_key_means_no_llm(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "")
    with pytest.raises(AllProvidersFailed):
        build_llm()


@pytest.mark.live
def test_live_gemini_answers():
    """Real call through the Gemini chain."""
    if not get_settings().google_api_key:
        pytest.skip("GOOGLE_API_KEY not set")
    text, provider = build_llm().complete([{"role": "user", "content": "Trả lời đúng một từ: xin chào"}])
    assert provider.startswith("Gemini") and text.strip()


def test_gemini_timeout_falls_back():
    p = GeminiProvider("k", "m", timeout=1)
    assert p.should_fall_back(httpx.ReadTimeout("timed out"))
    assert p.should_fall_back(httpx.ConnectError("refused"))
