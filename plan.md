# Plan: Prevent Third-Person Self-Mention in Agent Outputs

## Goal
Ensure each in-game agent has explicit self-identity context and avoids referring to itself by name (e.g., "Ava thinks..."), while preserving existing output formats and parser contracts.

## Constraints
- No schema/config changes.
- No unrelated refactors.
- Keep existing output contracts intact (`REQUEST/PASS`, `VOTE:` line, mafia `CHAT/VOTE_JSON`).

## Implementation Checkpoints

### C1) Make naming guidance self-aware in `src/runner/single_match.py`
**Changes**
- Update `_player_naming_instruction(...)` so it can generate per-speaker guidance:
  - Explicitly identify current speaker name.
  - List only **other** alive player names for name references.
  - Add rule: use first-person for self (`I/me/my`), do not refer to self by own name in third person.
- Apply per-speaker naming instruction at all prompt construction points:
  - Night chat prompt in `_append_night_phase_talk`.
  - Strategy and request prompts in `_append_day_phase_talk`.
  - Day speech prompt in `_build_day_speech_prompt` call site.
  - Follow-up request prompts in `_append_day_phase_talk`.
  - Day vote prompt in `_collect_day_vote_ballots`.

**Acceptance criteria**
- Every prompt built for a specific speaker includes:
  - Clear `You are <speaker_name>` context.
  - First-person self-reference rule.
  - "Use names only for other players" rule.

---

### C2) Strengthen day speech prompt wording in `src/runner/single_match.py`
**Changes**
- In `_build_day_speech_prompt`, keep current structure but add explicit style constraint:
  - Speak in first person.
  - Do not self-mention by own name.
- Keep current speech numbering, strategy, and prior-self-context lines unchanged.

**Acceptance criteria**
- Prompt still contains:
  - `speech #<n> out of <max>`
  - prior self statements block
- Plus explicit first-person / no-self-name instruction.

---

### C3) Add explicit self-identity placeholders in `src/agents/llm_agent.py`
**Changes**
- Ensure prompt builders pass self identity into template formatting (e.g., `self_name=self.name`).
- Keep method responsibilities the same; no behavioral changes beyond prompt text composition.

**Affected methods**
- `build_speak_request_prompt(...)`
- `build_followup_request_prompt(...)`
- `build_day_vote_prompt(...)`

**Acceptance criteria**
- Template rendering includes a speaker-specific identity variable.
- No change to response parsing interfaces or return types.

---

### C4) Update prompt templates in `src/agents/prompt/*.txt`
**Files**
- `src/agents/prompt/speak_request.txt`
- `src/agents/prompt/followup_request.txt`
- `src/agents/prompt/day_vote.txt`

**Changes**
- Add concise identity/style line in each template:
  - `You are {self_name}. Use first person (I/me/my). Do not refer to yourself as {self_name}.`
- Preserve strict output format lines exactly:
  - `REQUEST: ...` / `PASS: ...` one-line contract.
  - `VOTE: <exact alive player name>` one-line contract.

**Acceptance criteria**
- Templates contain explicit first-person/no-self-name guidance.
- Existing parser-triggering output format lines remain unchanged.

---

### C5) Test updates for prompt guarantees
**File: `tests/test_day_speech_prompt.py`**
- Extend day speech prompt test to assert first-person/no-self-name guidance is present.
- Keep existing assertions for speech numbering and prior-context behavior.

**File: `tests/test_day_vote.py`**
- Extend `test_build_day_vote_prompt_includes_role_belief_context` to assert self-identity guidance appears in rendered prompt.
- Keep assertions for:
  - `Role inferences (visible only):`
  - `VOTE: <exact alive player name>`
  - No unresolved placeholders.

**Acceptance criteria**
- New assertions validate identity guidance presence.
- Existing parse/resolve vote tests continue passing without contract changes.

## Verification Steps

1. Run targeted tests:
   - `uv run pytest tests/test_day_speech_prompt.py -q`
   - `uv run pytest tests/test_day_vote.py -q`
2. Run full suite:
   - `uv run pytest -q`

Success = all tests pass with no parser/format regressions.

## Risk Notes

- **Risk:** Strong first-person instruction could conflict with strict one-line formats.
  - **Mitigation:** Keep format constraints explicit and final in template wording.
- **Risk:** Excluding self from name list may reduce clarity when few players remain.
  - **Mitigation:** Include explicit self identity line (`You are {self_name}`) in every prompt.
- **Risk:** Prompt drift across multiple construction paths.
  - **Mitigation:** Centralize per-speaker naming instruction generation and reuse consistently.

## Rollout Order

1. `single_match.py` per-speaker naming instruction + day speech prompt wording (C1, C2).
2. `llm_agent.py` template formatting updates for self identity (C3).
3. Prompt template text updates (C4).
4. Test updates (C5).
5. Verification commands (targeted then full suite).
