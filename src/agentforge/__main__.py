"""Command-line demo for agentforge."""

from __future__ import annotations

import argparse
import sys

from agentforge import __version__
from agentforge.agents import BearAgent, BullAgent, RiskAgent
from agentforge.llm import LLMClient, MockLLMClient
from agentforge.orchestrator import DebateOrchestrator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentforge",
        description="Multi-agent AI research system.",
    )
    parser.add_argument("--version", action="version", version=f"agentforge {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    debate = sub.add_parser("debate", help="Run a multi-agent debate on a topic.")
    debate.add_argument("topic", type=str, help="The question to debate.")
    debate.add_argument(
        "--rounds", type=int, default=2, help="Number of debate rounds (default: 2)."
    )
    return parser


def _make_llm() -> LLMClient:
    """Return the default LLM client (MockLLM — no API key required)."""
    return MockLLMClient()


def _print_transcript(orchestrator_result) -> None:  # type: ignore[no-untyped-def]
    print("=== Transcript ===")
    for turn in orchestrator_result.transcript.turns:
        print()
        print(f"Round {turn.round_number}:")
        for msg in turn.messages:
            preview = msg.content[:100].replace(chr(10), " ")
            print(f"  [{msg.role.value:<5}] ({msg.stance.value:<8}) {preview}...")


def _run_debate(topic: str, rounds: int) -> int:
    llm = _make_llm()
    print(f"LLM:    {llm.model_name}")
    print(f"Topic:  {topic}")
    print(f"Rounds: {rounds}")
    print()

    orchestrator = DebateOrchestrator(
        llm,
        [BullAgent(llm), BearAgent(llm), RiskAgent(llm)],
    )
    result = orchestrator.run(topic, rounds=rounds)

    _print_transcript(result)

    print()
    print("=== Final Report ===")
    print(f"Topic:       {result.report.topic}")
    print(f"Final:       {result.report.final_stance.value}")
    print(f"Confidence:  {result.report.confidence}")
    print(f"Summary:     {result.report.summary[:200]}")
    if result.report.dissenting_views:
        print()
        print("Dissent:")
        for view in result.report.dissenting_views:
            print(f"  - {view[:100]}")
    print()
    print("Research tool. AI-generated output. Not financial advice.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "debate":
        return _run_debate(args.topic, args.rounds)
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
