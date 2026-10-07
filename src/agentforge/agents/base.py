"""Abstract base class for debate agents."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import UTC, datetime

from agentforge.llm import LLMClient
from agentforge.schemas import AgentMessage, AgentRole, DebateTranscript, Stance


class Agent(ABC):
    """An agent participates in a debate by producing structured messages.

    Each agent owns its role, its stance, and its perspective prompt. The
    orchestrator hands the agent the current transcript and the agent
    returns exactly one AgentMessage per turn.
    """

    def __init__(self, llm: LLMClient, role: AgentRole) -> None:
        self._llm = llm
        self._role = role

    @property
    def role(self) -> AgentRole:
        return self._role

    @property
    @abstractmethod
    def stance(self) -> Stance:
        """The directional stance this agent argues for."""
        raise NotImplementedError

    @property
    @abstractmethod
    def perspective(self) -> str:
        """One-line description of the agent's analytical lens."""
        raise NotImplementedError

    @abstractmethod
    def build_prompt(self, topic: str, transcript: DebateTranscript) -> str:
        """Construct the prompt for this agent for the current round."""
        raise NotImplementedError

    def contribute(self, topic: str, transcript: DebateTranscript) -> AgentMessage:
        """Run the LLM and produce a structured AgentMessage."""
        prompt = self.build_prompt(topic, transcript)
        content = self._llm.complete(prompt).strip()
        return AgentMessage(
            role=self._role,
            stance=self.stance,
            content=content or "(empty response)",
            evidence=[],
            confidence=0.5,
            timestamp=datetime.now(UTC),
        )
