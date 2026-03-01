# Draft: Agent Role Inference Voting

## Requirements (confirmed)
- [user]: "각각의 에이전트가 역할을 유추하도록"
- [user]: "지금은 너무 마피아를 찾는 것에 대해서만 대화가 쏠리는 경향"
- [user]: "마피아 뿐만 아니라 의사/시민의 여부도 중요"
- [user]: "각각의 에이전트가 서로를 어떻게 생각하는지 유지"
- [user]: "@src/agents/llm_agent.py 의 LLMAgent 클래스에서 자기가 했던 대화 히스토리, 상대방 역할에 대한 유추를 저장해서 voting 에 반영"
- [user]: "계획을 수립"

## Technical Decisions
- [decision]: Keep voting output contract as one-line `VOTE: <exact alive player name>` to preserve `_parse_day_vote` behavior.
- [decision]: Introduce per-agent role-inference memory owned by `LLMAgent` and updated from visible conversation history.
- [decision]: Integrate memory summary into day vote prompt inputs without changing `src/engine/vote.py` tally logic.
- [decision]: Preserve role-based visibility boundary by using `single_match._visible_history_for_player(...)` as upstream guardrail.

## Research Findings
- [src/runner/single_match.py]: Day strategy/speech/follow-up flow and day vote collection are centralized here.
- [src/agents/llm_agent.py]: Prompt builders and `_history_to_context(...)` are current context/memory entry points.
- [src/agents/prompt/day_vote.txt]: Current vote prompt restricts decision basis to self statements only.
- [tests/test_day_vote.py]: Existing parser/context tests are the primary extension point.
- [tests/test_history_visibility.py]: Existing visibility tests can guard against hidden-info leakage regressions.
- [tests/test_mafia_consensus.py]: Night consensus path is independent and should remain unchanged.

## Open Questions
- None (plan will apply defaults where preference-level ambiguity exists).

## Scope Boundaries
- INCLUDE: `LLMAgent` memory model, memory update policy, prompt contract extension for voting, runner wiring, unit/integration tests, docs update for behavior changes.
- EXCLUDE: Day vote tally algorithm in `src/engine/vote.py`, winner rules in `src/engine/rules.py`, and Streamlit UX redesign.