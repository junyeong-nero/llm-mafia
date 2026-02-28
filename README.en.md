# llm-mafia

Language: English | [Korean](README.ko.md)

This project is a simulator where LLM agents play a mafia game and record turn/phase-level events as logs.
It supports both CLI execution (`main.py`) and a Streamlit dashboard (`streamlit_app.py`).

## 1. Project Overview

- Game loop: `setup -> night -> day -> vote -> night ...`
- Core execution function: `run_single_match()` in `app/runner/single_match.py`
- Main outputs:
  - Event log (`events.jsonl`)
  - Summary file (`summary.json`)
  - Metrics report (stdout)

Win conditions are checked immediately after night ends and immediately after voting ends. If a condition is met, the game ends with `Phase.END`.

## 2. Speech Request Logic (Core)

The day (`day`) phase in this project is not a free-for-all speaking mode. It uses a **request-based queue (FIFO) speaking system**.
The core implementation is `_append_day_phase_talk()` in `app/runner/single_match.py` and `app/runner/speech_queue.py`.

### 2.1 Initial Request Collection at Day Start

When the day starts, the system iterates over all alive players once and does the following:

1. Each player generates a strategy sentence (`strategy`)
2. Each player decides whether to request speaking rights
   - Required LLM response format: `REQUEST: <reason>` or `PASS: <reason>`
   - Parsing function: `_parse_speech_request()`

At this time, the following event kinds are logged:

- `strategy`
- `speak_request`
- `speak_request_reason`

### 2.2 Request Parsing Rules

`_parse_speech_request()` determines whether a request is made using these rules:

- If the response contains `PASS`, return `False`
- If the response contains `REQUEST`, return `True`
- If neither appears but text is non-empty, treat it as an implicit request (`True`)
- If there is a colon (`:`), use the right side as the reason

In other words, even if the LLM response slightly deviates from the format, it can still be treated as a request as long as it is not completely empty.

### 2.3 Queue Insertion Constraints (Important)

Not every requester is added to the queue. `_enqueue_requester()` applies the following constraints:

- A player who already spoke `day_max_speeches_per_player` times cannot be enqueued
- A player already in the queue cannot be enqueued again (no duplicates)

The base queue is `SpeechQueue(items: list[int])`, and FIFO is implemented via `enqueue` + `dequeue(pop(0))`.

### 2.4 Actual Speech Handling + Follow-up Re-requests

After initial request collection, if the queue is not empty, the system repeats until the queue is exhausted:

1. Dequeue the front player
2. Generate one public speech (`speech`)
3. Right after that speech, re-ask **all alive players except the current speaker** for follow-up requests
   - Prompt intent: decide whether to rebut or reinforce the latest speech
   - Responses are also parsed with the same `REQUEST/PASS` rule
4. Enqueue newly `REQUEST`ing players at the FIFO tail

As a result, day-phase speaking order expands dynamically:

- Initial applicants are processed in FIFO order
- Follow-up applicants generated after each speech are appended to the tail
- Once the max speech limit is reached, later re-requests are ignored

### 2.5 Tests That Guarantee Speech Logic

The following tests verify key properties of the speech-request logic:

- `tests/test_single_match.py::test_day_phase_collects_requests_before_speeches`
  - Verifies request events happen before speech events
- `tests/test_single_match.py::test_day_phase_rechecks_requests_after_each_speech`
  - Verifies follow-up re-requests are actually performed after each speech
- `tests/test_single_match.py::test_day_phase_respects_configured_max_speeches_per_player`
  - Verifies per-player max speech limits are respected
- `tests/test_speech_queue.py`
  - Verifies FIFO ordering and empty-queue behavior

## 3. Usage

The commands below assume the repository root.

### 3.1 Requirements

- Python 3.13+
- `uv` is recommended (`uv.lock` is included)

### 3.2 Install

```bash
uv sync --frozen
```

### 3.3 Environment Variables

An OpenRouter API key is required.

```bash
cp .env.sample .env
```

Set at least the following value in `.env`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

Optional values:

- `OPENROUTER_HTTP_REFERER`
- `OPENROUTER_APP_TITLE`

### 3.4 Prepare the Config File

The default config file is `config.yaml`.

Important fields:

- `game.player_count`
- `game.day_max_speeches_per_player`
- `game.roles` (mafia/police/doctor/citizen)
- `llm.provider` (`openrouter` only)
- `llm.models` (each model's `name`, `model`, `count`)

Validation rules:

- Sum of role counts == `player_count`
- Sum of model counts == `player_count`
- `day_max_speeches_per_player` > 0

### 3.5 Run One Match via CLI

```bash
uv run python main.py --config config.yaml --seed 42 --max-rounds 10
```

Help:

```bash
uv run python main.py --help
```

After execution, metrics and log paths are printed to stdout.

### 3.6 Run the Streamlit Dashboard

```bash
uv run streamlit run streamlit_app.py
```

Help:

```bash
uv run streamlit run streamlit_app.py --help
```

Items visible in the dashboard:

- Turn/phase progress status
- Player alive status
- Real-time speech queue (`speech_queue`) and current speaker
- Chat replay

### 3.7 Run Tests

```bash
uv run pytest -q
```

Collect tests only:

```bash
uv run pytest --collect-only -q
```

## 4. Event Log Reference

Examples of major event kinds:

- `setup`
- `mafia_chat`
- `night_result`
- `strategy`
- `speak_request`
- `speak_request_reason`
- `speech_queue`
- `speech`
- `vote_result`
- `game_end`

By reading `events.jsonl`, you can trace at turn granularity who requested speaking, who entered the queue, and when each player actually spoke.

## 5. Code Map

- Entry point: `main.py`
- Dashboard: `streamlit_app.py`
- Match runner: `app/runner/single_match.py`
- Speech queue: `app/runner/speech_queue.py`
- Phase transitions: `app/engine/phase.py`
- Win/night rules: `app/engine/rules.py`
- Vote resolution: `app/engine/vote.py`
- Config loader/validation: `config.py`
