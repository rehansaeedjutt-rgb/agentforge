"""Orchestrator that runs a multi-round debate and produces a report."""

from __future__ import annotations

from dataclasses import dataclass, field

import structlog

from agentforge.agents.base import Agent
from agentforge.agents.judge import JudgeAgent
from agentforge.llm import LLMClient
from agentforge.schemas import DebateTranscript, DebateTurn, ResearchReport

logger = structlog.get_logger(__name__)


@dataclass
class DebateResult:
    """Final result of a debate run."""

    transcript: DebateTranscript
    report: ResearchReport
    rounds_completed: int = 0
    messages_exchanged: int = field(default=0)


class DebateOrchestrator:
    """Runs a structured debate between multiple agents.

    Flow:
      1. For each round:
         - each debater contributes one message based on prior transcript
         - messages are grouped into a DebateTurn
      2. Judge reads the full transcript and produces a ResearchReport.
      3. Returns transcript + report.
    """

    def __init__(self, llm: LLMClient, debaters: list[Agent]) -> None:
        if not debaters:
            msg = "at least one debater is required"
            raise ValueError(msg)
        if len(debaters) < 2:
            msg = "at least two debaters are required for a meaningful debate"
            raise ValueError(msg)
        self._llm = llm
        self._debaters = list(debaters)
        self._judge = JudgeAgent(llm)

    @property
    def debaters(self) -> list[Agent]:
        return list(self._debaters)

    def run(self, topic: str, rounds: int = 2) -> DebateResult:
        """Run the debate. Returns the transcript and final report."""
        if not topic.strip():
            msg = "topic must be non-empty"
            raise ValueError(msg)
        if rounds < 1:
            msg = "rounds must be >= 1"
            raise ValueError(msg)

        turns: list[DebateTurn] = []
        total_messages = 0

        for r in range(1, rounds + 1):
            transcript_so_far = DebateTranscript(topic=topic, turns=list(turns))
            messages = []
            for agent in self._debaters:
                message = agent.contribute(topic, transcript_so_far)
                messages.append(message)
                total_messages += 1
                logger.info(
                    "debate.message",
                    round=r,
                    role=agent.role.value,
                    stance=message.stance.value,
                    confidence=message.confidence,
                )
            turns.append(DebateTurn(round_number=r, messages=messages))

        transcript = DebateTranscript(topic=topic, turns=turns)
        report = self._judge.judge(topic, transcript)

        logger.info(
            "debate.complete",
            topic=topic,
            rounds=rounds,
            messages=total_messages,
            final_stance=report.final_stance.value,
            confidence=report.confidence,
        )

        return DebateResult(
            transcript=transcript,
            report=report,
            rounds_completed=rounds,
            messages_exchanged=total_messages,
        )
