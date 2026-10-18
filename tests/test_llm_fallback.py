"""The provider priority is DeepSeek -> Gemini; fallback must happen only on availability errors."""
import httpx
import openai
import pytest

from pldc.config import get_settings
from pldc.llm import AllProvidersFailed, DeepSeekProvider, FallbackLLM, build_llm


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


def test_deepseek_error_classification():
    p = DeepSeekProvider("k", "m", "http://127.0.0.1:9", timeout=1)
    req = httpx.Request("POST", "http://x")

    def status(code):
        return openai.APIStatusError("e", response=httpx.Response(code, request=req),
                                     body=None)

    assert p.should_fall_back(openai.APIConnectionError(request=req))
    assert p.should_fall_back(status(402))  # out of balance
    assert p.should_fall_back(status(429))
    assert p.should_fall_back(status(503))
    assert not p.should_fall_back(status(400))


@pytest.mark.live
def test_live_unreachable_deepseek_falls_back_to_gemini(monkeypatch):
    """Real call: DeepSeek pointed at a closed port must fall back to Gemini quickly."""
    if not get_settings().google_api_key:
        pytest.skip("GOOGLE_API_KEY not set")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-invalid")
    monkeypatch.setenv("DEEPSEEK_BASE_URL", "http://127.0.0.1:9")
    llm = build_llm()
    assert llm.provider_names[0] == "DeepSeek"
    text, provider = llm.complete([{"role": "user", "content": "Trả lời đúng một từ: xin chào"}])
    assert provider.startswith("Gemini") and text.strip()
