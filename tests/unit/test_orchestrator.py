"""Tests for agentforge.orchestrator."""

import pytest

from agentforge.agents import BearAgent, BullAgent, RiskAgent
from agentforge.llm import MockLLMClient
from agentforge.orchestrator import DebateOrchestrator
from agentforge.schemas import AgentRole, Stance


def _make_orchestrator() -> tuple[DebateOrchestrator, MockLLMClient]:
    llm = MockLLMClient()
    debaters = [BullAgent(llm), BearAgent(llm), RiskAgent(llm)]
    return DebateOrchestrator(llm, debaters), llm


def test_orchestrator_requires_at_least_one_debater() -> None:
    llm = MockLLMClient()
    with pytest.raises(ValueError, match="at least one"):
        DebateOrchestrator(llm, [])


def test_orchestrator_requires_at_least_two_debaters() -> None:
    llm = MockLLMClient()
    with pytest.raises(ValueError, match="at least two"):
        DebateOrchestrator(llm, [BullAgent(llm)])


def test_orchestrator_rejects_empty_topic() -> None:
    orch, _ = _make_orchestrator()
    with pytest.raises(ValueError, match="topic"):
        orch.run("", rounds=1)


def test_orchestrator_rejects_zero_rounds() -> None:
    orch, _ = _make_orchestrator()
    with pytest.raises(ValueError, match="rounds"):
        orch.run("valid topic", rounds=0)


def test_orchestrator_runs_two_rounds_three_debaters() -> None:
    orch, _ = _make_orchestrator()
    result = orch.run("Bitcoin in a retail portfolio?", rounds=2)
    assert result.rounds_completed == 2
    assert result.messages_exchanged == 6  # 2 rounds x 3 debaters
    assert len(result.transcript.turns) == 2
    assert result.transcript.total_messages == 6


def test_orchestrator_produces_report() -> None:
    orch, _ = _make_orchestrator()
    result = orch.run("Is ETH a good buy?", rounds=1)
    assert result.report.topic == "Is ETH a good buy?"
    assert result.report.final_stance in {Stance.POSITIVE, Stance.NEGATIVE, Stance.NEUTRAL}
    assert 0.0 <= result.report.confidence <= 1.0
    assert len(result.report.summary) > 0


def test_orchestrator_judge_stance_positive_when_bull_dominates() -> None:
    orch, _ = _make_orchestrator()
    # 2 rounds => 2 bull (positive) + 2 bear (negative) + 2 risk (neutral)
    # Tie between positive and negative -> judge returns NEUTRAL.
    result = orch.run("Balanced topic", rounds=2)
    assert result.report.final_stance is Stance.NEUTRAL


def test_orchestrator_transcript_has_all_roles() -> None:
    orch, _ = _make_orchestrator()
    result = orch.run("topic", rounds=1)
    roles_seen = {msg.role for turn in result.transcript.turns for msg in turn.messages}
    assert AgentRole.BULL in roles_seen
    assert AgentRole.BEAR in roles_seen
    assert AgentRole.RISK in roles_seen


def test_orchestrator_debaters_property_returns_copy() -> None:
    orch, _ = _make_orchestrator()
    a = orch.debaters
    b = orch.debaters
    assert a == b
    assert a is not b
