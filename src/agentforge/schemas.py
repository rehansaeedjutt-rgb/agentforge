"""Core models for multi-agent debate and reports."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AgentRole(StrEnum):
    """Role an agent plays in the debate."""

    BULL = "bull"
    BEAR = "bear"
    RISK = "risk"
    JUDGE = "judge"


class Stance(StrEnum):
    """Directional stance on a topic."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class AgentMessage(BaseModel):
    """A single contribution from an agent."""

    model_config = ConfigDict(frozen=True)

    role: AgentRole
    stance: Stance
    content: str = Field(..., min_length=1)
    evidence: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    timestamp: datetime


class DebateTurn(BaseModel):
    """One round of debate: each agent contributes in order."""

    model_config = ConfigDict(frozen=True)

    round_number: int = Field(..., ge=1)
    messages: list[AgentMessage]


class DebateTranscript(BaseModel):
    """Full transcript of a multi-round debate."""

    model_config = ConfigDict(frozen=True)

    topic: str
    turns: list[DebateTurn]

    @property
    def total_messages(self) -> int:
        """Total number of individual agent messages."""
        return sum(len(turn.messages) for turn in self.turns)


class ResearchReport(BaseModel):
    """Final structured report produced by the Judge."""

    model_config = ConfigDict(frozen=True)

    topic: str
    final_stance: Stance
    summary: str = Field(..., min_length=1)
    key_points: list[str] = Field(default_factory=list)
    confidence: float = Field(..., ge=0.0, le=1.0)
    rationale: str = Field(..., min_length=1)
    dissenting_views: list[str] = Field(default_factory=list)
    generated_at: datetime
