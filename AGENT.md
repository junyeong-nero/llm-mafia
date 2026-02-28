# AGENT.md

This document defines shared working rules for people and AI agents in this repository.
These rules are the minimum standard for "small changes with immediate verification."

## 1) Core Principles (Required)

- Implement in small units and verify immediately after each change.
- Remove duplication as soon as it appears, and improve ambiguous names immediately.
- Do not separate feature work and refactoring; perform lightweight structural improvement in every task.
- Do not change code based on guesses. Confirm evidence from files/tests/logs first, then modify.

## 2) File Length Limit (Required)

- All source code files must stay at **1000 lines or fewer**.
- Treat files over 800 lines as split candidates and start module decomposition.
- If a planned change would exceed 1000 lines, split/refactor the file before adding new functionality.

Scope:

- `*.py`, `*.ts`, `*.tsx`, `*.js`, `*.jsx`

Exceptions (only when unavoidable):

- Auto-generated files
- External vendor code
- Snapshot/lock files

When using an exception, include the following three items in the PR/commit description:

- Target file under exception
- Reason the exception is required
- Follow-up cleanup plan (when/how it will be split)

## 3) Continuous Refactoring Rules (Required)

Check the following together with every feature or bug fix:

- Function length and responsibility: split a function if it has multiple responsibilities
- Data structures: unify duplicated struct/dictionary shapes
- Dependency direction: separate interfaces so high-level policy is not directly coupled to low-level implementation
- Error handling: consolidate into common handling patterns
- Logging: keep the minimum context needed for debugging

## 4) Decomposition Guide

When a file grows large, decompose it in this priority order:

1. Domain logic (rules/judgment)
2. Infrastructure logic (API calls/persistence)
3. Presentation logic (output/CLI/UI)
4. Utilities (parsing/formatting/validation)

## 5) Verification Rules (Required)

Run commands at the repository root.

### 5.1 Common

- After feature changes, bug fixes, or refactoring, run tests at least once.
- If a failure occurs, fix the root cause and rerun the same verification command.

```bash
uv run pytest -q
```

### 5.2 Quick Integrity Check

- Use the command below when you need to verify collection integrity before/after test execution.

```bash
uv run pytest --collect-only -q
```

### 5.3 Extra Checks for Entrypoint Changes

- If you modified `main.py` or CLI argument handling logic:

```bash
uv run python main.py --help
```

- If you modified `src/streamlit_app.py` or dashboard run paths:

```bash
uv run streamlit run src/streamlit_app.py --help
```

## 6) Documentation Sync Rules

- When behavior/config/command changes, review `README.en.md`, `README.ko.md`, and related `docs/` together.
- If docs and implementation diverge, reconcile code or docs within the same task.
- If configuration validation rules (`player_count`, sum of role/model counts, `day_max_speeches_per_player > 0`) change, sync the validation rules in documentation as well.

## 7) Work Checklist

Before closing a task, confirm all of the following:

- File length is 1000 lines or fewer
- No new duplicate code
- Responsibility boundaries are clear
- No mismatch between docs (`README.md`, `README.en.md`, `README.ko.md`, `docs/`) and implementation
- Verification commands for the change type were executed and results confirmed

## 8) Recommended Working Style

- Keep PRs/commits small and feature-scoped
- During review, check both "does it work" and "did structure improve"
- Repay technical debt within the current impact scope instead of deferring as TODO
- Do not leave rule exceptions as verbal agreements; record them in writing
