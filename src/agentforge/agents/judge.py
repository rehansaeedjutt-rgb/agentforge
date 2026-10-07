"""Judge agent: synthesizes debate into a final structured report."""

from __future__ import annotations

from datetime import UTC, datetime

from agentforge.agents.base import Agent
from agentforge.llm import LLMClient
from agentforge.schemas import (
    AgentRole,
    DebateTranscript,
    ResearchReport,
    Stance,
)


class JudgeAgent(Agent):
    """Reads the transcript and produces a structured ResearchReport."""

    def __init__(self, llm: LLMClient) -> None:
        super().__init__(llm, AgentRole.JUDGE)

    @property
    def stance(self) -> Stance:
        return Stance.NEUTRAL

    @property
    def perspective(self) -> str:
        return "weighs evidence from all sides and reports honestly, including dissent"

    def build_prompt(self, topic: str, transcript: DebateTranscript) -> str:
        prior = self._render_prior(transcript)
        return (
            f"You are the JUDGE. Topic: {topic}\n"
            f"Your perspective: {self.perspective}\n"
            f"Transcript:\n{prior}\n"
            "Summarize the debate and state a final stance in 5 sentences."
        )

    def judge(self, topic: str, transcript: DebateTranscript) -> ResearchReport:
        """Produce the final report. Uses the LLM for the summary text."""
        summary = self._llm.complete(self.build_prompt(topic, transcript)).strip()
        final_stance = self._infer_stance(transcript)
        dissent = [
            f"{msg.role.value}: {msg.content}"
            for turn in transcript.turns
            for msg in turn.messages
            if msg.stance != final_stance and msg.stance != Stance.NEUTRAL
        ]
        return ResearchReport(
            topic=topic,
            final_stance=final_stance,
            summary=summary or "(no summary produced)",
            key_points=[
                f"{msg.role.value}: {msg.content[:100]}"
                for turn in transcript.turns
                for msg in turn.messages
            ][:5],
            confidence=0.5,
            rationale="Weighed bull and bear arguments; risk agent flagged unknowns.",
            dissenting_views=dissent[:3],
            generated_at=datetime.now(UTC),
        )

    @staticmethod
    def _infer_stance(transcript: DebateTranscript) -> Stance:
        """Simple count-based stance inference.

        The judge does not pick sides arbitrarily — it counts how many
        non-neutral messages argued each way and returns the majority.
        Ties resolve to NEUTRAL.
        """
        positive = sum(
            1 for turn in transcript.turns for msg in turn.messages if msg.stance == Stance.POSITIVE
        )
        negative = sum(
            1 for turn in transcript.turns for msg in turn.messages if msg.stance == Stance.NEGATIVE
        )
        if positive > negative:
            return Stance.POSITIVE
        if negative > positive:
            return Stance.NEGATIVE
        return Stance.NEUTRAL

    @staticmethod
    def _render_prior(transcript: DebateTranscript) -> str:
        lines: list[str] = []
        for turn in transcript.turns:
            for msg in turn.messages:
                lines.append(f"  [{msg.role.value}] {msg.content}")
        return "\n".join(lines) if lines else "  (no prior messages)"
