import httpx
import openai
import pytest

import llm


def _request():
    return httpx.Request("POST", "http://localhost/chat/completions")


def _status_error(cls, code):
    return cls("simulated", response=httpx.Response(code, request=_request()), body=None)


class FakeCompletions:
    """Raises the queued errors in order, then returns a normal reply."""

    def __init__(self, errors):
        self.errors = list(errors)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        message = type("M", (), {"content": "hello"})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()


def make_client(errors, max_retries=3):
    client = llm.LLMClient(api_key="k", base_url="http://localhost", model="m", max_retries=max_retries)
    fake = FakeCompletions(errors)
    client._client = type("Fake", (), {"chat": type("Chat", (), {"completions": fake})()})()
    return client, fake


@pytest.fixture
def sleeps(monkeypatch):
    recorded = []
    monkeypatch.setattr(llm.time, "sleep", recorded.append)
    return recorded


def test_missing_settings_are_named_in_the_error(monkeypatch):
    for name in ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError) as err:
        llm.LLMClient()
    assert "LLM_API_KEY" in str(err.value) and "LLM_MODEL" in str(err.value)


def test_a_normal_reply_needs_no_retry(sleeps):
    client, fake = make_client([])
    assert client.complete("system", "user") == "hello"
    assert fake.calls == 1 and sleeps == []


def test_server_errors_are_retried_with_growing_waits(sleeps):
    errors = [_status_error(openai.InternalServerError, 503)] * 2
    client, fake = make_client(errors)
    assert client.complete("s", "u") == "hello"
    assert fake.calls == 3
    assert sleeps == [10, 20]


def test_rate_limits_are_retried(sleeps):
    client, fake = make_client([_status_error(openai.RateLimitError, 429)])
    assert client.complete("s", "u") == "hello"
    assert sleeps == [10]


def test_timeouts_are_retried(sleeps):
    client, fake = make_client([openai.APITimeoutError(request=_request())])
    assert client.complete("s", "u") == "hello"
    assert sleeps == [10]


def test_it_gives_up_after_the_maximum_number_of_attempts(sleeps):
    errors = [_status_error(openai.InternalServerError, 503)] * 5
    client, fake = make_client(errors, max_retries=3)
    with pytest.raises(openai.InternalServerError):
        client.complete("s", "u")
    assert fake.calls == 3
    assert sleeps == [10, 20]


def test_other_errors_are_not_retried(sleeps):
    client, fake = make_client([ValueError("a real bug")])
    with pytest.raises(ValueError):
        client.complete("s", "u")
    assert fake.calls == 1 and sleeps == []


def test_waits_are_capped_at_a_minute(sleeps):
    errors = [_status_error(openai.InternalServerError, 503)] * 6
    client, _ = make_client(errors, max_retries=7)
    client.complete("s", "u")
    assert max(sleeps) == 60
