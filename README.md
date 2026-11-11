# LLM Mafia

A Mafia game simulator where LLMs lie, bluff, and try to outsmart each other — **all in Korean**. Players can mix the official **K-EXAONE 2.5 API** with **OpenRouter** models in the same match, and tournaments compare win rates by model and role without quietly swapping one model for another.

![Web Dashboard](assets/demo.gif)

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Game Flow](#game-flow)
3. [Agent Internals](#agent-internals)
4. [Getting Started](#getting-started)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Project Structure](#project-structure)
8. [Development](#development)

---

## Architecture Overview

```
main.py (CLI / web dashboard)
  → match_runner          picks a runner
      ├── graph_runner    LangGraph-based (default)
      └── single_match    legacy loop
            → engine/     game rules & state
            → agents/     LLM players
            → providers/  EXAONE / OpenRouter APIs
            → metrics/, io/   logs & stats
```

---

## Game Flow

A match repeats these phases until one side wins:

```
SETUP → NIGHT → DAY → VOTE → (repeat)
```

- **Night** — mafia chat privately and vote on a target. The police and doctor each pick their own target through the LLM (`INVESTIGATE: <name>` / `PROTECT: <name>`). Investigating shows only *mafia or not* — never the exact non-mafia role. Self-protection and consecutive protection are allowed.
- **Day** — players don't just take turns. A **SpeechQueue** decides the order, so players can request to speak and respond to earlier speeches like a real debate, until everyone who wanted to talk has talked.
- **Vote** — simultaneous ballots, plurality elimination, then a win check.

If an LLM outputs a garbled special action, the action is skipped and logged privately (`night_action_invalid`) — never replaced with a random pick.

---

## Agent Internals

Each player is an `LLMAgent` with its own memory of public dialogue and its own confirmed role knowledge. Anything unconfirmed is left to the LLM to reason about — there's no keyword-scoring formula pretending to be deduction.

The key design rule: **separate facts from claims.**

- The police player's investigation results are facts. Everything else — accusations, denials, "I'm the police!" — stays a claim, kept in the original wording so the model can spot contradictions itself.
- Private info (night choices, investigation results, personal strategies) is visible only to its owner. Mafia players can't peek at police or doctor secrets.
- Voting uses a frozen snapshot of history: you see everyone's past public votes and statements, but never this round's other ballots.
- Full event history is provided without truncation. It costs more tokens, but no evidence is dropped at decision time.

Prompts are built from three layers: a system prompt (role persona + rules + Korean language policy), the player-visible event history, and the current action prompt. Mafia are allowed to deceive; citizens are told to separate recorded facts from claims.

---

## Getting Started

**Requirements:** Python 3.13+ and [uv](https://github.com/astral-sh/uv).

```bash
git clone https://github.com/your-username/llm-mafia.git
cd llm-mafia

uv sync
cp .env.sample .env
```

Set `EXAONE_API_KEY` in `.env` for the default match, and `OPENROUTER_API_KEY` too if you want the comparison preset.

### EXAONE credentials (Pi / macOS Keychain)

If Pi already stored your key in the macOS Keychain under `pi-exaone-api-key`, you can reuse it without copying the secret into this repo:

```bash
export EXAONE_API_KEY_COMMAND='security find-generic-password -s pi-exaone-api-key -w'
uv run main.py
```

`EXAONE_API_KEY` (if set directly) takes precedence over the command. The command is opt-in, runs without a shell, and its output is captured but never logged.

Both providers use non-streaming chat completions, a 4,096-token output cap, a 120-second timeout, and up to four retry attempts. Sampling is provider-default, so a seed fixes roles and engine randomness — **not** LLM outputs. Missing credentials or a broken API response stops the match; agents never silently fall back to another model.

---

## Configuration

`config.yaml` defaults to eight K-EXAONE 2.5 players:

```yaml
game:
    player_count: 8
    day_max_speeches_per_player: 2   # max speeches per player during the day
    roles:
        mafia: 2
        police: 1
        doctor: 1
        citizen: 4

llm:
    provider: exaone
    models:
        - name: k-exaone-2.5
          model: k-exaone_v2.5
          count: 8
```

The `roles` sum and the `models[*].count` sum must both equal `player_count`, and model names must be unique. Each model can override the default provider with `provider: exaone` or `provider: openrouter` — see `config.compare.yaml` for a mixed example.

All responses are requested in Korean. If a natural-language reply has no Hangul, the same model is asked once to restate it in Korean; a second violation stops the match. (Name-only votes and special actions are exempt.)

---

## Usage

### Single match (CLI)

```bash
uv run main.py
```

| Option | Default | Description |
|--------|---------|-------------|
| `--runner {graph,legacy}` | `graph` | Runner selection (`graph`: LangGraph, `legacy`: loop) |
| `--seed <int>` | random | Seed for reproducible runs |
| `--max-rounds <int>` | `10` | Maximum number of rounds |
| `--config <path>` | `config.yaml` | Config file path |

```bash
uv run main.py --runner legacy --seed 7 --max-rounds 10
```

### Balanced model comparison

```bash
# Requires both EXAONE and OpenRouter credentials.
uv run main.py --config config.compare.yaml --tournament 3 --seed 42 --max-rounds 5
```

The comparison preset seats two players each from K-EXAONE 2.5 and two free OpenRouter models (`nvidia/nemotron-3-ultra-550b-a55b:free`, `nvidia/nemotron-3.5-super-vl:free`, picked from OpenRouter's weekly free top ranking on 2026-11-10). With `--tournament N`, model groups rotate through the same seats so names, speaking order, and roles are balanced; the seed advances per block.

Results go to `logs/`, with a tournament summary at `logs/tournaments/<run>/summary.json`. An API failure stops the tournament and excludes the unfinished match from scores.

**Reading the numbers:** a player's win counts as their team's victory, and win rate is wins ÷ appearances. Police and doctor count as the citizen team. Keep in mind that a few matches are a smoke test, not proof of superiority — appearances within one game aren't independent samples, and partial rotations after a failure aren't balanced.

The [2026-11-10 pilot report](docs/comparison-2026-11-10.md) is **historical**: it predates the private night-action and voting-context fixes, so its win rates don't reflect the current engine. A [post-fix rerun](docs/comparison-2026-11-10-postfix.md) completed only 1 of 3 matches (EXAONE's daily 100-call limit), so its partial numbers also shouldn't be ranked.

### Web dashboard

Watch the match live in your browser — speeches, votes, and results stream over SSE. The UI follows [`DESIGN.md`](DESIGN.md) and uses inline SVG icons.

```bash
uv run main.py --web --port 9000
```

| Option | Default | Description |
|--------|---------|-------------|
| `--web` | off | Launch the web dashboard instead of a CLI match |
| `--host <str>` | `127.0.0.1` | Dashboard bind host |
| `--port <int>` | `8000` | Dashboard port |

---

## Project Structure

```
llm-mafia/
├── main.py                        # CLI entrypoint
├── config.yaml                    # default K-EXAONE 2.5 match
├── config.compare.yaml            # mixed-provider comparison preset
├── DESIGN.md                      # dashboard design system
│
├── src/
│   ├── agents/                    # LLMAgent, personas, prompt templates
│   ├── engine/                    # game state, phases, rules, visibility, votes
│   ├── runner/                    # graph/legacy runners, night phase, speech queue, tournament
│   ├── web/                       # FastAPI server, SSE session, dashboard static files
│   ├── providers/                 # EXAONE / OpenRouter clients, factory, settings
│   ├── metrics/                   # stats collection, win rates, reports
│   └── io/                        # JSONL event logs, config loader
│
├── docs/rule.md                   # detailed game rules
└── tests/                         # pytest suite
```

---

## Development

```bash
uv run pytest -q
```

The `graph` and `legacy` runners share the same day/vote and night helpers, and tests verify they produce identical results. To see it yourself:

```bash
uv run main.py --runner graph  --seed 42
uv run main.py --runner legacy --seed 42
```

---

## License

MIT License — see [LICENSE](LICENSE)
