"""Tests for agentforge.schemas."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from agentforge.schemas import (
    AgentMessage,
    AgentRole,
    DebateTranscript,
    DebateTurn,
    ResearchReport,
    Stance,
)

_TS = datetime(2026, 1, 1, tzinfo=UTC)


def test_agent_message_valid() -> None:
    msg = AgentMessage(
        role=AgentRole.BULL,
        stance=Stance.POSITIVE,
        content="Bitcoin is a strong asset.",
        confidence=0.8,
        timestamp=_TS,
    )
    assert msg.role is AgentRole.BULL
    assert msg.confidence == 0.8


def test_agent_message_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        AgentMessage(
            role=AgentRole.BULL,
            stance=Stance.POSITIVE,
            content="",
            confidence=0.5,
            timestamp=_TS,
        )


def test_agent_message_rejects_confidence_out_of_range() -> None:
    with pytest.raises(ValidationError):
        AgentMessage(
            role=AgentRole.BULL,
            stance=Stance.POSITIVE,
            content="ok",
            confidence=1.5,
            timestamp=_TS,
        )


def test_agent_message_immutable() -> None:
    msg = AgentMessage(
        role=AgentRole.BULL,
        stance=Stance.POSITIVE,
        content="ok",
        confidence=0.5,
        timestamp=_TS,
    )
    with pytest.raises(ValidationError):
        msg.confidence = 0.9  # type: ignore[misc]


def test_debate_transcript_total_messages() -> None:
    m1 = AgentMessage(
        role=AgentRole.BULL, stance=Stance.POSITIVE, content="a", confidence=0.5, timestamp=_TS
    )
    m2 = AgentMessage(
        role=AgentRole.BEAR, stance=Stance.NEGATIVE, content="b", confidence=0.5, timestamp=_TS
    )
    t = DebateTranscript(
        topic="test",
        turns=[
            DebateTurn(round_number=1, messages=[m1, m2]),
            DebateTurn(round_number=2, messages=[m1]),
        ],
    )
    assert t.total_messages == 3


def test_research_report_valid() -> None:
    report = ResearchReport(
        topic="test",
        final_stance=Stance.POSITIVE,
        summary="Conclusion text.",
        confidence=0.7,
        rationale="Based on evidence.",
        generated_at=_TS,
    )
    assert report.final_stance is Stance.POSITIVE
    assert report.key_points == []
    assert report.dissenting_views == []
