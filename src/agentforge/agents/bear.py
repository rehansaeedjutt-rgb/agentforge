"""Bear agent: argues for downside."""

from __future__ import annotations

from agentforge.agents.base import Agent
from agentforge.schemas import AgentRole, DebateTranscript, Stance


class BearAgent(Agent):
    """Argues the negative case for a topic."""

    def __init__(self, llm) -> None:  # type: ignore[no-untyped-def]
        super().__init__(llm, AgentRole.BEAR)

    @property
    def stance(self) -> Stance:
        return Stance.NEGATIVE

    @property
    def perspective(self) -> str:
        return "argues the strongest case for downside, risk, and structural weakness"

    def build_prompt(self, topic: str, transcript: DebateTranscript) -> str:
        prior = self._render_prior(transcript)
        return (
            f"You are the BEAR agent. Topic: {topic}\n"
            f"Your perspective: {self.perspective}\n"
            f"Prior messages:\n{prior}\n"
            "State the strongest negative case in 3 sentences. Be specific."
        )

    @staticmethod
    def _render_prior(transcript: DebateTranscript) -> str:
        lines: list[str] = []
        for turn in transcript.turns:
            for msg in turn.messages:
                lines.append(f"  [{msg.role.value}] {msg.content}")
        return "\n".join(lines) if lines else "  (no prior messages)"
