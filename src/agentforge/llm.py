"""LLM client abstraction with a deterministic MockLLM default.

Why MockLLM by default:

- No API key, no signup, no credit card.
- Tests are deterministic and fast.
- The architecture is decoupled from any specific provider.

Real providers (Groq, OpenAI, Ollama) implement the same ``LLMClient``
interface. Switch by passing a different client to the Orchestrator.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import structlog

logger = structlog.get_logger(__name__)


class LLMError(RuntimeError):
    """Raised when an LLM call fails irrecoverably."""


class LLMClient(ABC):
    """Abstract LLM interface. All methods are synchronous by design.

    Async can be added later; the sync interface keeps the orchestrator
    simple and easy to reason about.
    """

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Human-readable model identifier."""
        raise NotImplementedError

    @abstractmethod
    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        """Return a completion for the given prompt."""
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Deterministic LLM stub for tests and demos.

    Produces responses that depend only on the prompt content — no
    randomness, no network, no API key. This makes it possible to
    write end-to-end tests of the debate logic without flakiness.

    Canned responses are matched case-insensitively: a key "bull"
    matches any prompt containing "bull", "Bull", or "BULL".
    """

    def __init__(self, canned_responses: dict[str, str] | None = None) -> None:
        self._canned = dict(canned_responses or {})
        self._call_count = 0
        self._history: list[str] = []

    @property
    def model_name(self) -> str:
        return "mock-llm-v1"

    @property
    def call_count(self) -> int:
        """Number of complete() calls made so far."""
        return self._call_count

    @property
    def history(self) -> list[str]:
        """Prompts received in order."""
        return list(self._history)

    def complete(self, prompt: str, max_tokens: int = 512) -> str:
        self._call_count += 1
        self._history.append(prompt)
        prompt_lower = prompt.lower()
        for key, response in self._canned.items():
            if key.lower() in prompt_lower:
                return response[:max_tokens]
        return f"[mock:{self._call_count}] {prompt[:120]}"


DEFAULT_TIMEOUT_SECONDS: Final[float] = 30.0
