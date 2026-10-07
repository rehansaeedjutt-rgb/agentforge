# agentforge

> Multi-agent AI research system with structured debate and honest confidence scoring.

[![CI](https://github.com/rehansaeedjutt-rgb/agentforge/actions/workflows/ci.yml/badge.svg)](https://github.com/rehansaeedjutt-rgb/agentforge/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

## Overview

`agentforge` is a multi-agent system for structured AI research: multiple
specialized agents (Bull, Bear, Risk Officer) independently analyze the same
topic, then debate each other in turns, and a Judge agent produces a final
structured report with confidence scoring.

**Status:** Phase 1 in progress.

**Free-first:** Defaults to a `MockLLMClient` for tests and demos — no API key,
no signup, no credit card. Optional Groq provider for real LLM inference.

## Disclaimer

Research and educational tool. Outputs are AI-generated and not financial advice.

## License

MIT - see [LICENSE](LICENSE) file.

## Author

**Muhammad Rehan Saeed** - [GitHub](https://github.com/rehansaeedjutt-rgb)
