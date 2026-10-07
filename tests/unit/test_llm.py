"""Tests for agentforge.llm."""

from agentforge.llm import MockLLMClient


def test_mock_returns_default_response() -> None:
    client = MockLLMClient()
    response = client.complete("hello")
    assert "[mock:1]" in response
    assert "hello" in response


def test_mock_counts_calls() -> None:
    client = MockLLMClient()
    client.complete("a")
    client.complete("b")
    client.complete("c")
    assert client.call_count == 3


def test_mock_records_history() -> None:
    client = MockLLMClient()
    client.complete("first")
    client.complete("second")
    assert client.history == ["first", "second"]


def test_mock_canned_response_used_when_matched() -> None:
    client = MockLLMClient(canned_responses={"bull": "positive case here"})
    assert client.complete("You are the BULL agent") == "positive case here"


def test_mock_falls_back_when_no_canned_match() -> None:
    client = MockLLMClient(canned_responses={"xyz": "unused"})
    response = client.complete("nothing matches")
    assert "unused" not in response
    assert "[mock:" in response


def test_mock_truncates_to_max_tokens() -> None:
    client = MockLLMClient(canned_responses={"a": "x" * 100})
    assert len(client.complete("a", max_tokens=10)) == 10


def test_mock_model_name() -> None:
    assert MockLLMClient().model_name == "mock-llm-v1"
