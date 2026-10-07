"""Tests for agentforge.agents."""

from agentforge.agents import BearAgent, BullAgent, JudgeAgent, RiskAgent
from agentforge.llm import MockLLMClient
from agentforge.schemas import AgentRole, DebateTranscript, Stance


def _empty_transcript() -> DebateTranscript:
    return DebateTranscript(topic="test topic", turns=[])


def test_bull_role_and_stance() -> None:
    llm = MockLLMClient()
    agent = BullAgent(llm)
    assert agent.role is AgentRole.BULL
    assert agent.stance is Stance.POSITIVE


def test_bear_role_and_stance() -> None:
    llm = MockLLMClient()
    agent = BearAgent(llm)
    assert agent.role is AgentRole.BEAR
    assert agent.stance is Stance.NEGATIVE


def test_risk_role_and_stance() -> None:
    llm = MockLLMClient()
    agent = RiskAgent(llm)
    assert agent.role is AgentRole.RISK
    assert agent.stance is Stance.NEUTRAL


def test_judge_role() -> None:
    llm = MockLLMClient()
    agent = JudgeAgent(llm)
    assert agent.role is AgentRole.JUDGE


def test_bull_contribute_produces_valid_message() -> None:
    llm = MockLLMClient()
    agent = BullAgent(llm)
    msg = agent.contribute("Is ETH a good buy?", _empty_transcript())
    assert msg.role is AgentRole.BULL
    assert msg.stance is Stance.POSITIVE
    assert len(msg.content) > 0


def test_bear_contribute_produces_valid_message() -> None:
    llm = MockLLMClient()
    agent = BearAgent(llm)
    msg = agent.contribute("Is ETH a good buy?", _empty_transcript())
    assert msg.role is AgentRole.BEAR
    assert msg.stance is Stance.NEGATIVE


def test_agent_prompts_include_topic_and_perspective() -> None:
    llm = MockLLMClient()
    agent = BullAgent(llm)
    prompt = agent.build_prompt("bitcoin adoption", _empty_transcript())
    assert "bitcoin adoption" in prompt
    assert "BULL" in prompt


def test_agent_prompts_include_prior_messages() -> None:
    llm = MockLLMClient()
    bull = BullAgent(llm)
    bear = BearAgent(llm)

    # Bull first round
    first_msg = bull.contribute("topic", _empty_transcript())
    from agentforge.schemas import DebateTurn

    transcript = DebateTranscript(
        topic="topic",
        turns=[DebateTurn(round_number=1, messages=[first_msg])],
    )
    # Bear sees bull's message
    prompt = bear.build_prompt("topic", transcript)
    assert "bull" in prompt.lower()
