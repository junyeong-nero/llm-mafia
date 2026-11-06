# AGENTS.md
Practical guide for human and AI coding agents in `llm-mafia`.
This document is based on repository evidence from `pyproject.toml`, `README.md`, `src/`, and `tests/`.

## 1) Project Snapshot
- Language/runtime: Python `>=3.13` (`pyproject.toml`)
- Dependency manager/runner: `uv`
- Main entrypoint: `main.py`
- Web dashboard: `src/web/server.py` (FastAPI + uvicorn, HTML/JS frontend)
- Test framework: `pytest` (dev dependency)
- Match runner dispatcher: `src/runner/match_runner.py` (default runner: `graph`)

## 2) Primary Files To Read First
- `pyproject.toml` (runtime + dependencies)
- `README.md` (setup/run commands)
- `src/config.py` (config schema and validation)
- `src/runner/match_runner.py` and `src/runner/graph_runner.py` (current runner flow)
- `tests/` (expected behavior and test style)

## 3) Setup Commands
Run from repo root:
```bash
uv sync
cp .env.sample .env
```
`OPENROUTER_API_KEY` is required in `.env` for real provider calls.

## 4) Run Commands
Single match:
```bash
uv run main.py
```
With options:
```bash
uv run main.py --seed 7 --max-rounds 10 --config config.yaml --runner legacy
```
Default graph runner explicitly:
```bash
uv run main.py --runner graph
```
Web dashboard:
```bash
uv run main.py --web --host 127.0.0.1 --port 8000
```
Entrypoint health checks:
```bash
uv run python main.py --help
```

## 5) Test Commands (Canonical)
Full suite (quiet):
```bash
uv run pytest -q
```
Collection integrity:
```bash
uv run pytest --collect-only -q
```
Single test file:
```bash
uv run pytest tests/test_day_vote.py -q
```
Single test case (node id):
```bash
uv run pytest tests/test_day_vote.py::test_parse_day_vote_accepts_vote_prefix_and_name -q
```
Keyword subset:
```bash
uv run pytest -k "parse_day_vote" -q
```
Tip: if node id is unknown, run `uv run pytest --collect-only -q` first.

## 6) Lint / Typecheck / Build Reality
Current repo has no dedicated config for:
- lint (`ruff`, `flake8`, `pylint`)
- static type checking (`mypy`, `pyright`)
- CI workflows (`.github/workflows/`)
- task runners (`Makefile`, `justfile`, etc.)
Implication: `pytest` is the mandatory automated verification baseline.

## 7) Code Style Guidelines (Observed)
### Imports
- Start files with `from __future__ import annotations`.
- Import order: stdlib, third-party, local `src.*` imports.
- Use explicit imports; avoid wildcard imports.

### Formatting
- 4-space indentation.
- Keep lines readable; split long calls/strings across lines.
- Prefer small helper functions for parsing and validation logic.

### Types
- Add explicit parameter and return type hints.
- Use modern generics (`list[str]`, `dict[str, object]`) and `str | None` unions.
- Use `@dataclass(frozen=True)` for immutable config/domain records.
- Keep mutable state explicit (`GameState` is intentionally non-frozen).
- Avoid broad `Any` unless unavoidable boundary code (tests/Streamlit glue).

### Naming
- `snake_case`: functions, variables, helpers.
- `PascalCase`: classes, dataclasses, exception types.
- `UPPER_SNAKE_CASE`: module constants (`RETRYABLE_STATUSES`).
- Enum members uppercase, enum values lowercase strings.

### Error handling
- Validate early and fail with precise messages.
- Use `ValueError` for invalid input/config data.
- Use `FileNotFoundError` for missing files.
- Use domain exceptions when needed (`OpenRouterError`).
- Preserve causal context with `raise ... from exc`.
- Do not silently swallow exceptions.

### Control flow
- Prefer guard clauses to reduce nesting.
- Keep pure transforms separate from side effects where practical.
- Isolate normalization/parsing in dedicated helpers.

## 8) Testing Conventions
- Tests are in `tests/` and named `test_*.py`.
- Test functions use behavior-focused `test_*` names.
- Assert concrete outputs/messages, not only truthiness.
- Add helper builders/fixtures for repeated setup.

## 9) Architecture Boundaries
- Simulation core: `src/engine/`
- Agent behavior/prompts: `src/agents/`
- Match orchestration dispatch: `src/runner/match_runner.py`
- Graph runner: `src/runner/graph_runner.py`
- Legacy runner: `src/runner/single_match.py`
- Provider integration: `src/providers/openrouter_client.py`
- Metrics/reporting: `src/metrics/`
- Web dashboard: `src/web/`
  - `view_model.py` — pure view state + payload→view-event logic (no framework imports)
  - `match_session.py` — background thread session + SSE hub
  - `server.py` — FastAPI app, REST/SSE endpoints, static file serving
  - `static/` — `index.html`, `style.css`, `icons.js`, `app.js` frontend
  - UI must follow root `DESIGN.md`; use SVG icons from `icons.js`, never emoji
Keep changes within these boundaries to avoid cross-layer leakage.

## 10) Repository Rules To Honor
This `AGENTS.md` remains the repository guidance file. Key rules:
- Small incremental changes with immediate verification
- File length target `<= 1000` lines
- Refactor while implementing (avoid long-lived duplication)
- Re-run verification after fixes

## 11) Cursor / Copilot Rule Files
Checked paths:
- `.cursor/rules/`
- `.cursorrules`
- `.github/copilot-instructions.md`
Current status: none of these files exist in this repository.

## 12) Recommended Agent Workflow
1. Read target module and related tests first.
2. Implement minimal focused changes.
3. Run relevant targeted test(s).
4. Run full suite: `uv run pytest -q`.
5. If entrypoint/UI changed, run both `--help` checks.
6. Update docs when behavior/config/commands change.
This workflow keeps changes small, verifiable, and aligned with repository patterns.
