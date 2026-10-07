# agentforge

> Multi-agent AI research system with structured debate and honest confidence scoring.

[![CI](https://github.com/rehansaeedjutt-rgb/agentforge/actions/workflows/ci.yml/badge.svg)](https://github.com/rehansaeedjutt-rgb/agentforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Coverage](https://img.shields.io/badge/coverage-97%25-brightgreen.svg)](#testing)

## Overview

`agentforge` is a multi-agent system for structured AI research: multiple
specialized agents independently analyze the same topic, debate each other
in rounds, and a Judge agent produces a final structured report with honest
confidence scoring and dissent tracking.

**Status:** Phase 1 complete — orchestrator, four agents, CLI, tests, CI.

**Free-first:** Defaults to `MockLLMClient` — no API key, no signup, no credit
card. Optional Groq adapter for real LLM inference (extra install).

## Why this project exists

Most "AI agent" demos are a single prompt wrapped in marketing. Real
institutional research desks work differently: multiple analysts argue, a
risk officer flags unknowns, and a senior analyst synthesizes — including the
disagreements.

`agentforge` models that process honestly:

- **Multiple perspectives** — Bull, Bear, and Risk agents debate in rounds.
- **Structured output** — every message is a validated Pydantic model.
- **Dissent is preserved** — the report lists viewpoints that disagree with the majority.
- **No fake confidence** — the judge scores confidence transparently.
- **Testable by default** — `MockLLMClient` makes the whole system deterministic.

## Live demo

```
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
python -m agentforge debate "Should Bitcoin be part of a retail portfolio?" --rounds 2
```

Example output:

```
LLM:    mock-llm-v1
Topic:  Should Bitcoin be part of a retail portfolio?
Rounds: 2

=== Transcript ===

Round 1:
  [bull ] (positive) [mock:1] You are the BULL agent. Topic: Should Bitcoin...
  [bear ] (negative) [mock:2] You are the BEAR agent. Topic: Should Bitcoin...
  [risk ] (neutral ) [mock:3] You are the RISK agent. Topic: Should Bitcoin...

Round 2:
  [bull ] (positive) [mock:4] You are the BULL agent. Topic: Should Bitcoin...
  [bear ] (negative) [mock:5] You are the BEAR agent. Topic: Should Bitcoin...
  [risk ] (neutral ) [mock:6] You are the RISK agent. Topic: Should Bitcoin...

=== Final Report ===
Topic:       Should Bitcoin be part of a retail portfolio?
Final:       neutral
Confidence:  0.5
Summary:     [mock:7] You are the JUDGE. Topic: Should Bitcoin...

Dissent:
  - bull: [mock:1] You are the BULL agent. Topic: Should Bitcoin...
  - bear: [mock:2] You are the BEAR agent. Topic: Should Bitcoin...

Research tool. AI-generated output. Not financial advice.
```

The default `MockLLMClient` produces deterministic output for tests and demos.
Real inference requires swapping in a provider (see Architecture below).

## Architecture

```
      Topic
        |
        v
  +---------------------+
  |   Orchestrator      |
  |---------------------|
  |  for each round:    |
  |    bull.contribute  |
  |    bear.contribute  |
  |    risk.contribute  |
  |  judge.judge()      |
  +---------------------+
        |
        v
  [ DebateTranscript ] ---> [ ResearchReport ]
        |                         |
        v                         v
   [ AgentMessage ]           final_stance
   [ DebateTurn ]              summary
   prior context              dissent[]
```

### Design choices

| Choice | Why |
|--------|-----|
| `LLMClient` ABC with `MockLLMClient` default | Decouples architecture from any provider; enables free CI + deterministic tests |
| Sync LLM interface | Simpler orchestrator; async can layer on later without rewriting agents |
| Frozen Pydantic models | Immutable domain objects — the transcript cannot be silently corrupted |
| Judge infers stance by counting | No arbitrary favoritism; ties resolve to NEUTRAL |
| Dissent preserved in report | Honest research; hides nothing |
| Confidence as `[0.0, 1.0]` | Forces explicit numeric honesty, not "high/medium/low" labels |

## Tech stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.12+ |
| Schemas | `pydantic` v2 |
| Config | `pydantic-settings` |
| Logging | `structlog` |
| Testing | `pytest` |
| Quality | `ruff`, `mypy --strict` |
| CI | GitHub Actions on Python 3.12 / 3.13 / 3.14 |
| Optional LLM | `groq` (extra install) |

## Project structure

```
agentforge/
├── src/agentforge/
│   ├── schemas.py           # AgentMessage, DebateTurn, DebateTranscript, ResearchReport
│   ├── llm.py               # LLMClient ABC + MockLLMClient
│   ├── orchestrator.py      # DebateOrchestrator + DebateResult
│   ├── agents/
│   │   ├── base.py          # Agent ABC
│   │   ├── bull.py          # BullAgent (positive case)
│   │   ├── bear.py          # BearAgent (negative case)
│   │   ├── risk.py          # RiskAgent (neutral, risk-focused)
│   │   └── judge.py         # JudgeAgent (final report)
│   └── __main__.py          # CLI
├── tests/unit/              # 30 unit tests
├── pyproject.toml
└── README.md
```

## Testing

30 unit tests, all offline (MockLLM, no network):

| Module | Coverage |
|--------|----------|
| `schemas.py` | 100% |
| `orchestrator.py` | 100% |
| `agents/bull.py` | 100% |
| `agents/bear.py` | 100% |
| `agents/risk.py` | 100% |
| `agents/judge.py` | 89% |
| `agents/base.py` | 89% |
| `llm.py` | 95% |
| **Total** | **97%** |

## Roadmap

- [x] Phase 1 — Orchestrator, four agents, CLI, tests
- [ ] Phase 2 — Real Groq provider (drop-in LLMClient implementation)
- [ ] Phase 3 — Local Ollama provider (fully offline inference)
- [ ] Phase 4 — Tool-calling agents (fetch live data during debate)
- [ ] Phase 5 — Persistence: store transcripts + reports in SQLite
- [ ] Phase 6 — Confidence scoring driven by evidence quality, not fixed 0.5

## Limitations

- Default `MockLLMClient` produces **echo-style** output, not real analysis.
  This is intentional — real inference requires a provider.
- Confidence is hardcoded at 0.5 in Phase 1; Phase 6 will compute it.
- No tool-calling yet — agents reason from prompts alone.
- Judge stance inference is a simple count; real-world judges would weigh evidence.

## Disclaimer

Research and educational tool. Outputs are AI-generated and not financial advice.

## License

MIT - see [LICENSE](LICENSE) file.

## Author

**Muhammad Rehan Saeed** - [GitHub](https://github.com/rehansaeedjutt-rgb)
