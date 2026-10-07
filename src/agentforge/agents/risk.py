"""Risk agent: neutral risk analysis."""

from __future__ import annotations

from agentforge.agents.base import Agent
from agentforge.schemas import AgentRole, DebateTranscript, Stance


class RiskAgent(Agent):
    """Neutral agent focused on risk factors and unknowns."""

    def __init__(self, llm) -> None:  # type: ignore[no-untyped-def]
        super().__init__(llm, AgentRole.RISK)

    @property
    def stance(self) -> Stance:
        return Stance.NEUTRAL

    @property
    def perspective(self) -> str:
        return "identifies what could go wrong, what is unknown, and what would falsify the thesis"

    def build_prompt(self, topic: str, transcript: DebateTranscript) -> str:
        prior = self._render_prior(transcript)
        return (
            f"You are the RISK agent. Topic: {topic}\\n"
            f"Your perspective: {self.perspective}\\n"
            f"Prior messages:\\n{prior}\\n"
            "Identify the 3 most important risks or unknowns. Be specific."
        )

    @staticmethod
    def _render_prior(transcript: DebateTranscript) -> str:
        lines: list[str] = []
        for turn in transcript.turns:
            for msg in turn.messages:
                lines.append(f"  [{msg.role.value}] {msg.content}")
        return "\\n".join(lines) if lines else "  (no prior messages)"
