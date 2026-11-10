# LLM Mafia

A Mafia game simulator where LLMs deceive and reason against each other **in Korean**. Players can use the official **K-EXAONE 2.5 API** or **OpenRouter** in the same match. Balanced tournaments compare team win rates by model and role without silently substituting another model.

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
┌─────────────────────────────────────────────────────────┐
│                        main.py                          │
│     CLI entrypoint · runner selection · web dashboard   │
└────────────────────────┬────────────────────────────────┘
                         │
          ┌──────────────┴──────────────┐
          │        match_runner         │  runner dispatcher
          └──────────┬──────────────────┘
          ┌──────────┴──────────┐
          │                     │
   graph_runner          single_match          ← runner implementations
   (LangGraph-based)     (legacy loop)
          │
          │ from each node
          │
   ┌──────┴────────────────────────────────┐
   │              engine/                  │
   │  game_state  · phase · rules · vote   │  game rules & state
   └──────────────────────────────────────┘
   ┌──────────────────────────────────────┐
   │              agents/                 │
   │  LLMAgent · RoleBelief · Prompt      │  LLM players
   └──────────────────────────────────────┘
   ┌──────────────────────────────────────┐
   │            providers/               │
   │    ExaoneClient · OpenRouterClient   │  LLM API
   └──────────────────────────────────────┘
   ┌──────────────────────────────────────┐
   │            metrics/ · io/           │
   │     event logs · stats collection    │
   └──────────────────────────────────────┘
```

**Default execution flow**: `main.py` → `match_runner` → `graph_runner` (default) or `single_match` (legacy) → each node calls `engine` and `agents` → results are recorded via `metrics` and `io`.

---

## Game Flow

A match repeats the following phases until a win condition is met.

```
SETUP
  │  seat players, assign roles at random
  ▼
NIGHT  ──────────────────────────────────────────────────────┐
  │  mafia-only chat → consensus vote → choose elimination    │
  │  police: investigate / doctor: protect (internal)         │
  ▼                                                          │
DAY    ──────────────────────────────────────────────────────┤
  │  plan strategy → SpeechQueue-based discussion (see below) │
  ▼                                                          │
VOTE                                                         │
  │  every player votes → majority elimination → win check    │
  └──────────────────── repeat unless game over ─────────────┘
```

### SpeechQueue — Day Discussion Mechanism

During the day, players do not all speak at once. A speech-request queue decides the order, so new speeches follow up on earlier ones like a real debate.

```
[plan strategy] ──→ [initial speech requests]
                     │
           ┌─────────┴──────────┐
           │ REQUEST            │ PASS
      enqueue               queue empty?
           │                    │
           └─────────┬──────────┘
                 queue empty?
                  │         │
                  no        yes
                  │         │
         [next speaker talks]  [go to vote]
                  │
         [follow-up requests from other players]
                  │
           ┌──────┴──────┐
           │ REQUEST     │ PASS
        enqueue      speakers left?
           │             │
           └──────┬──────┘
                  └──→ [queue empty?] ──yes──→ [go to vote]
                              │
                              no
                              │
                     [next speaker talks]
```

### LangGraph Node Structure

`graph_runner` implements the flow above as nodes and edges in a LangGraph `StateGraph`.

```
START
  └─→ setup_match
        └─→ advance_phase ─┬─→ night_private_chat
                           │       └─→ night_consensus
                           │             └─→ night_resolution
                           │                   └─→ check_winner
                           │
                           ├─→ day_prepare_memory
                           │       └─→ day_collect_initial_requests
                           │               ├─→ day_run_next_speech ←┐
                           │               │       └─→ day_collect_followups ─┘
                           │               └─→ advance_phase (when queue is drained)
                           │
                           ├─→ vote_collect_ballots
                           │       └─→ vote_resolve
                           │             └─→ check_winner
                           │
                           └─→ finalize_state ──→ END
```

If `check_winner` finds a winner, the graph moves to `finalize_state`; otherwise it returns to `advance_phase` and starts the next round.

---

## Agent Internals

Each player is an `LLMAgent` instance. Beyond calling an LLM, it keeps its own memory and inference system.

### RoleBelief — Role Inference

Agents analyze game events and track the probability that each other player is mafia, doctor, or citizen.

```
visible_history (event log)
        │
        ├─ day events ──→ _apply_day_inference
        │    · vote records       (+0.14 mafia)
        │    · suspicious speech  (+0.20 mafia)
        │    · trusting speech    (+0.15 citizen)
        │    · protective speech  (+0.14 doctor)
        │
        └─ night events ──→ _apply_night_inference
             · mafia vote         (+0.18 mafia)
             · consensus target   (+0.10 mafia)
                    │
                    ▼
           RoleBelief { mafia: 0.xx, doctor: 0.xx, citizen: 0.xx, evidence: [...] }
```

These probabilities are injected into the LLM prompt through `build_belief_context()` when the agent decides how to vote.

### Prompt Composition

Each agent's output is built from three layers.

| Layer | Content |
|-------|---------|
| System | Role-specific persona (`persona_mafia.txt`, etc.) + shared Korean policy (`language.txt`) |
| History | Recent event context (6 day events, 3 night events, role-signal tags) |
| User | Current action prompt (strategy / speech request / vote, etc.) |

---

## Getting Started

### Requirements

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (package management)

### Installation

```bash
git clone https://github.com/your-username/llm-mafia.git
cd llm-mafia

uv sync

cp .env.sample .env
# Set EXAONE_API_KEY for the default match.
# Also set OPENROUTER_API_KEY for config.compare.yaml.
```

---

### EXAONE credentials (including Pi / macOS Keychain)

The official endpoint is `https://api.exaone.ai/v1/chat/completions`, with model ID
`k-exaone_v2.5` ([API reference](https://docs.exaone.ai/en/docs/k-exaone-2.5)).
Set `EXAONE_API_KEY` in the environment or `.env`.

If Pi already stores your key in the macOS Keychain under `pi-exaone-api-key`, reuse
it without copying the secret into this repository:

```bash
export EXAONE_API_KEY_COMMAND='security find-generic-password -s pi-exaone-api-key -w'
uv run main.py
```

`*_API_KEY` takes precedence over `*_API_KEY_COMMAND`. The command is explicitly
opt-in, runs without a shell, and its output is captured rather than logged.
The app does not read or modify Pi's configuration. Environment variables take
precedence over `.env`; do not leave placeholder keys in either source.

Both providers use non-streaming chat completions with a 4,096-token output cap
(including reasoning), a 120-second request timeout, and up to four attempts.
EXAONE uses `reasoning_effort: low`; OpenRouter uses each model's default reasoning
settings. Sampling is provider-default, so a game seed fixes roles and engine
randomness, **not** LLM outputs. Missing credentials, exhausted API retries,
truncated/empty completions, or unexpected response model IDs stop the match.
Game agents never fall back to another model or fabricate a response.

---

## Configuration

`config.yaml` defaults to eight K-EXAONE 2.5 players and defines the game rules and model seating.

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

The sum of `roles` and the sum of `models[*].count` must both equal `player_count`.
Model names must be unique. Each model can override the default `llm.provider`
with `provider: exaone` or `provider: openrouter`; see `config.compare.yaml` for a
mixed-provider example. All natural-language responses are requested in Korean;
player names and parser tokens (`CHAT:`, `VOTE_JSON:`, `REQUEST:`, `PASS:`, `VOTE:`)
remain unchanged. Role-inference heuristics recognize Korean words and particles
such as `Emma는` as well as English input. If a natural-language response has no
Hangul, the agent asks the **same model** to restate it in Korean once; a second
violation stops the match. Name-only day votes are exempt. This is a minimal
language check, not a guarantee against every English word or transliterated name.

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
# examples
uv run main.py --runner graph
uv run main.py --runner legacy --seed 7 --max-rounds 10
```

### Balanced model comparison

```bash
# Requires both EXAONE and OpenRouter credentials.
uv run main.py --config config.compare.yaml --tournament 3 --seed 42 --max-rounds 5
# Larger sample: four complete rotation cycles.
uv run main.py --config config.compare.yaml --tournament 12 --seed 42 --max-rounds 5
```

The comparison preset seats two players from each model: K-EXAONE 2.5,
`nvidia/nemotron-3-ultra-550b-a55b:free`, and `nvidia/nemotron-3.5-super-vl:free`.
The two free models were selected from OpenRouter's
[Top Weekly, free-price filter](https://openrouter.ai/models?max_price=0&order=top-weekly)
on **2026-11-10**: #1 Ultra (~3.21T tokens) and #2 Super VL (~2.90T tokens).
This ranking measures tokens processed in the last seven days, not requests or users.
The preset pins these IDs rather than silently changing models when rankings change.
It uses six players (one mafia, one police, one doctor, three citizens) and one
public speech per player per day. Free model availability and quotas change;
verify IDs with `GET https://openrouter.ai/api/v1/models`. There is no paid fallback
and no `openrouter/free` router, which would obscure model identity.

`--tournament N` requires equal model counts and a multiple of the number of
models. Each block uses the same seed and rotates model groups through the same
seats, balancing names, speaking positions, and assigned roles; the next block
increments the seed. Individual game logs are saved under `logs/`; a tournament
summary under `logs/tournaments/<run>/summary.json` records the config, seeds,
rotations, completed matches, invalid votes, and model/team/role statistics.
Completed results are saved after every match. An API failure stops the tournament,
marks the summary `failed`, and excludes the unfinished match from all scores.

**Scoring and limitations:** a player's win is their team's victory, even if that
player was eliminated. Win rate is wins / player appearances, including draws in
the denominator. Police and doctor count as the citizen team. Unplayed roles have
`win_rate: null`, not zero. Multiple copies of a model can be on opposing teams,
and appearances within one game are not independent samples. Three matches are
only a smoke/pilot comparison, not evidence of statistical superiority. Also,
the existing engine selects police investigation and doctor protection randomly;
this benchmark measures discussion/voting, not LLM-controlled special abilities.
Partial rotation cycles after a failure are not balanced and should not be ranked.

See the [2026-11-10 pilot report](docs/comparison-2026-11-10.md) for the three
completed games, language-compliance observations, and execution interruptions.

### Web dashboard (HTML + JS)

Visualizes the match in real time. The browser connects to a FastAPI server over SSE (Server-Sent Events) and streams speeches, votes, and results as they happen. The UI follows [`DESIGN.md`](DESIGN.md) and uses inline SVG icons.

```bash
uv run main.py --web
```

| Option | Default | Description |
|--------|---------|-------------|
| `--web` | off | Launch the web dashboard (instead of a CLI match) |
| `--host <str>` | `127.0.0.1` | Dashboard bind host |
| `--port <int>` | `8000` | Dashboard port |

```bash
# example
uv run main.py --web --port 9000
```

---

## Project Structure

```
llm-mafia/
├── main.py                        # CLI entrypoint
├── config.yaml                    # default K-EXAONE 2.5 match
├── config.compare.yaml            # mixed-provider balanced comparison preset
├── DESIGN.md                      # dashboard design system
│
├── src/
│   ├── agents/
│   │   ├── llm_agent.py           # LLMAgent, RoleBelief, role inference logic
│   │   └── prompt/
│   │       ├── personas.py        # role-specific system prompt loader
│   │       └── *.txt              # persona / action prompt templates
│   │
│   ├── engine/
│   │   ├── game_state.py          # immutable data models (Player, GameState, GameEvent)
│   │   ├── phase.py               # phase transition logic
│   │   ├── rules.py               # seating, win conditions, night resolution
│   │   └── vote.py                # day vote tally
│   │
│   ├── runner/
│   │   ├── match_runner.py        # graph/legacy runner dispatcher
│   │   ├── graph_runner.py        # LangGraph-based runner (default)
│   │   ├── single_match.py        # legacy loop runner + shared helpers
│   │   ├── speech_queue.py        # SpeechQueue (day discussion ordering)
│   │   └── tournament.py          # multi-match tournament orchestration
│   │
│   ├── web/
│   │   ├── server.py              # FastAPI app (static serving + REST/SSE API)
│   │   ├── match_session.py       # background match session + SSE event hub
│   │   ├── view_model.py          # progress payload → view state logic
│   │   └── static/
│   │       ├── index.html         # dashboard HTML
│   │       ├── style.css          # dashboard styles (based on DESIGN.md)
│   │       ├── icons.js           # inline SVG icons (Lucide, no emoji)
│   │       └── app.js             # SSE subscription and rendering logic
│   │
│   ├── providers/
│   │   ├── chat_client.py         # shared OpenAI-compatible HTTP transport
│   │   ├── exaone_client.py       # official EXAONE API
│   │   ├── openrouter_client.py   # OpenRouter API
│   │   ├── factory.py             # per-model provider routing
│   │   ├── settings.py            # environment / opt-in key command
│   │   └── model_registry.py      # config → model slot conversion
│   │
│   ├── metrics/
│   │   ├── collector.py           # game statistics collection
│   │   ├── comparison.py          # model/team/role win rates
│   │   └── report.py              # statistics report output
│   │
│   └── io/
│       ├── event_log.py           # JSONL event log / summary JSON writer
│       └── config.py              # config file loader
│
├── docs/
│   └── rule.md                    # detailed game rules
└── tests/                         # pytest test suite
```

---

## Development

### Running tests

```bash
uv run pytest -q
```

### Comparing the two runners

The `graph` and `legacy` runners share the same helper functions (`single_match.py`). Run both with the same seed to compare results.

```bash
uv run main.py --runner graph  --seed 42
uv run main.py --runner legacy --seed 42
```

---

## License

MIT License — see [LICENSE](LICENSE)
