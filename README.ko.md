# llm-mafia

Language: Korean | [English](README.en.md)

LLM 에이전트들이 마피아 게임을 진행하고, 턴/페이즈 단위 이벤트를 로그로 남기는 시뮬레이터입니다.
CLI 실행(`main.py`)과 Streamlit 대시보드(`src/streamlit_app.py`)를 모두 지원합니다.

## 1. 프로젝트 개요

- 게임 루프: `setup -> night -> day -> vote -> night ...`
- 핵심 실행 함수: `src/runner/single_match.py`의 `run_single_match()`
- 주요 산출물:
  - 이벤트 로그(`events.jsonl`)
  - 요약 파일(`summary.json`)
  - 메트릭 리포트(표준 출력)

승리 조건은 밤 종료 직후와 투표 종료 직후에 즉시 검사되며, 조건 충족 시 `Phase.END`로 게임이 종료됩니다.

## 2. 발언권 요청 로직 (핵심)

이 프로젝트의 낮(`day`) 페이즈는 "아무나 자유 발언"이 아니라, **요청 기반 큐(FIFO) 발언 시스템**으로 동작합니다.
구현의 중심은 `src/runner/single_match.py`의 `_append_day_phase_talk()`와 `src/runner/speech_queue.py`입니다.

### 2.1 낮 페이즈 시작 시 초기 요청 수집

낮이 시작되면 생존 플레이어 전체를 한 번 순회하면서 다음을 수행합니다.

1. 각 플레이어가 전략 문장(`strategy`) 생성
2. 각 플레이어가 발언권 요청 의사 결정
   - LLM 응답 포맷 요구: `REQUEST: <이유>` 또는 `PASS: <이유>`
   - 파싱 함수: `_parse_speech_request()`

이때 이벤트 로그에는 아래 kind가 남습니다.

- `strategy`
- `speak_request`
- `speak_request_reason`

### 2.2 요청 파싱 규칙

`_parse_speech_request()`는 다음 규칙으로 요청 여부를 판정합니다.

- 응답에 `PASS`가 포함되면 `False`
- 응답에 `REQUEST`가 포함되면 `True`
- 둘 다 없고 텍스트가 비어있지 않으면 암묵적 요청(`True`)
- 콜론(`:`)이 있으면 뒤쪽을 이유로 사용

즉, LLM이 형식을 조금 벗어나도 완전히 빈 응답이 아니면 요청으로 처리될 수 있습니다.

### 2.3 큐 삽입 제약(중요)

요청자가 모두 큐에 들어가는 것은 아닙니다. `_enqueue_requester()`에서 아래 제약을 적용합니다.

- `day_max_speeches_per_player` 이상 이미 발언한 플레이어는 큐 삽입 불가
- 이미 큐에 있는 플레이어는 중복 삽입 불가

기본 큐는 `SpeechQueue(items: list[int])`이며 `enqueue` + `dequeue(pop(0))` 방식의 FIFO입니다.

### 2.4 실제 발언 처리 + 후속 재요청

초기 요청 수집 뒤 큐가 비지 않았다면, 큐를 소진할 때까지 반복합니다.

1. 큐 맨 앞 플레이어 dequeue
2. 공개 발언(`speech`) 1회 생성
3. 발언 직후, **현재 발언자를 제외한 모든 생존자**에게 후속 요청을 다시 받음
   - 프롬프트: 방금 발언에 대해 반박/보강할지 결정
   - 응답도 동일하게 `REQUEST/PASS`로 파싱
4. 새롭게 `REQUEST`한 플레이어를 FIFO 뒤에 enqueue

따라서 낮 페이즈의 발언 순서는 아래처럼 동적으로 확장됩니다.

- 초기 신청자들 FIFO 처리
- 각 발언 이후 생기는 후속 신청자들이 큐 뒤에 추가
- 최대 발언 횟수 제한에 걸리면 이후 재요청되어도 제외

### 2.5 발언권 로직 보장 테스트

아래 테스트가 발언권 로직의 핵심 성질을 검증합니다.

- `tests/test_single_match.py::test_day_phase_collects_requests_before_speeches`
  - 요청 이벤트가 발언 이벤트보다 먼저 발생하는지 검증
- `tests/test_single_match.py::test_day_phase_rechecks_requests_after_each_speech`
  - 각 발언 후 후속 재요청이 실제로 다시 수행되는지 검증
- `tests/test_single_match.py::test_day_phase_respects_configured_max_speeches_per_player`
  - 플레이어별 최대 발언 횟수 제한 준수 검증
- `tests/test_speech_queue.py`
  - FIFO 순서/빈 큐 동작 검증

## 3. 사용 방법

아래 명령은 저장소 루트 기준입니다.

### 3.1 요구사항

- Python 3.13+
- `uv` 사용 권장 (`uv.lock` 포함)

### 3.2 설치

```bash
uv sync --frozen
```

### 3.3 환경 변수 설정

OpenRouter 키가 필요합니다.

```bash
cp .env.sample .env
```

`.env`에 최소 아래 값을 설정하세요.

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

선택값:

- `OPENROUTER_HTTP_REFERER`
- `OPENROUTER_APP_TITLE`

### 3.4 설정 파일 준비

기본 설정 파일은 `config.yaml`입니다.

중요 필드:

- `game.player_count`
- `game.day_max_speeches_per_player`
- `game.roles`(mafia/police/doctor/citizen)
- `llm.provider` (`openrouter` 고정)
- `llm.models` (각 모델의 `name`, `model`, `count`)

검증 규칙:

- 역할 수 합 == `player_count`
- 모델 count 합 == `player_count`
- `day_max_speeches_per_player` > 0

### 3.5 CLI로 1회 매치 실행

```bash
uv run python main.py --config config.yaml --seed 42 --max-rounds 10
```

도움말:

```bash
uv run python main.py --help
```

실행 후 표준 출력으로 메트릭/로그 경로가 출력됩니다.

### 3.6 Streamlit 대시보드 실행

```bash
uv run streamlit run src/streamlit_app.py
```

도움말:

```bash
uv run streamlit run src/streamlit_app.py --help
```

대시보드에서 확인 가능한 항목:

- 턴/페이즈 진행 상태
- 플레이어 생존 상태
- 실시간 발언 큐(`speech_queue`)와 현재 발언자
- 채팅 리플레이

### 3.7 테스트 실행

```bash
uv run pytest -q
```

테스트만 수집 확인:

```bash
uv run pytest --collect-only -q
```

## 4. 이벤트 로그 참고

주요 이벤트 kind 예시:

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

`events.jsonl`를 보면 "누가 언제 발언을 요청했고, 큐에 들어갔고, 실제로 언제 말했는지"를 턴 단위로 추적할 수 있습니다.

## 5. 코드 지도

- 실행 엔트리: `main.py`
- 대시보드: `src/streamlit_app.py`
- 매치 러너: `src/runner/single_match.py`
- 발언 큐: `src/runner/speech_queue.py`
- 페이즈 전이: `src/engine/phase.py`
- 승리/야간 판정: `src/engine/rules.py`
- 투표 판정: `src/engine/vote.py`
- 설정 로더/검증: `src/config.py`
