# use_ollama

Ollama/OpenAI 호환 LLM을 사용해 `deepagents` 기반 챗봇 백엔드를 실행하는 예제다.

현재 구성:
- FastAPI + SSE 스트리밍 응답
- 메인 에이전트 + 서브에이전트 구조
- 대화 세션, 메시지, 작업 기록(trace)을 PostgreSQL에 저장

## 구성 파일

- `main.py`: FastAPI 앱과 스트리밍 API
- `deep_agent.py`: 메인/서브 에이전트 정의
- `deep_agent_with_langgraph.py`: Deep Agent와 LangGraph를 섞은 예시 workflow
- `session_store.py`: PostgreSQL 세션 저장소
- `config/settings.py`: `.env` 설정 로딩
- `POSTGRESQL.md`: PostgreSQL 명령/예시 문서
- `DEEP_AGENT_LANGGRAPH_EXAMPLE.md`: LangGraph 예시 구조 설명

## 사전 준비

### 1. conda 환경

이 프로젝트는 `harness` 환경 기준으로 테스트했다.

```bash
conda activate harness
```

### 2. PostgreSQL

로컬 PostgreSQL 16 기준으로 확인했다.

서비스 실행:

```bash
brew services start postgresql@16
```

DB 생성:

```bash
createdb deepagent_chat
```

테이블은 앱 시작 시 자동 생성된다.

## 설치

`harness` 환경에서 의존성을 설치한다.

```bash
cd use_ollama
pip install -r requirements.txt
```

`psycopg[binary]`가 포함되어 있어 PostgreSQL 연결에 필요하다.

## 환경 변수

`.env`는 현재 디렉토리에 두고 아래처럼 설정한다.

```env
MODEL_MAP_JSON={"model_1":"gemma4:31b","model_2":"minimax-m2.7:cloud","model_3":"qwen3.5:cloud","model_4":"glm-5:cloud"}
API_KEY=your_api_key
API_HOST=https://ollama.com
DB_HOST=localhost
DB_PORT=5432
DB_NAME=deepagent_chat
DB_USER=bogeun
DB_PASSWORD=
```

설명:
- `MODEL_MAP_JSON`: 모델 키와 실제 모델명 매핑
- `API_KEY`: Ollama Cloud 또는 호환 API 키
- `API_HOST`: LLM API base URL
- `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`: PostgreSQL 연결 정보

앱 내부에서는 위 값을 조합해서 PostgreSQL 연결 문자열을 만든다.

## 실행

백엔드 실행:

```bash
conda activate harness
cd use_ollama
python main.py
```

기본 주소:

- API: `http://localhost:8000`
- OpenAPI docs: `http://localhost:8000/docs`

## 주요 API

- `GET /api/sessions`: 세션 목록 조회
- `POST /api/sessions`: 새 세션 생성
- `GET /api/sessions/{session_id}`: 세션 상세 조회
- `DELETE /api/sessions/{session_id}`: 세션 삭제
- `POST /api/chat/stream`: SSE 기반 스트리밍 채팅

## DB 저장 구조

현재 PostgreSQL 테이블:

- `chat_sessions`
- `chat_messages`
- `chat_message_traces`

저장되는 내용:
- 세션 제목, 생성/수정 시각
- user/assistant 메시지
- assistant 응답 중 발생한 tool/subagent/LangGraph trace

즉, 새로고침 후에도 대화 내용과 작업 기록 보기가 복원된다.

## LangGraph Trace

`deep_agent_with_langgraph.py`를 사용할 때는 `main.py`가 아래 이벤트를 작업 기록으로 저장한다.

- `on_chain_start`, `on_chain_end`: LangGraph 노드 진행
- `on_tool_start`, `on_tool_end`: tool 호출
- `on_retriever_start`, `on_retriever_end`: RAG/retriever 호출

프런트에서는 스트리밍 중에는 최신 raw trace를 보여주고, 응답 완료 후 `작업 기록 보기`에서는 같은 tool 호출의 시작/결과를 한 카드로 합쳐 보여준다.

최종 답변 본문은 중간 model chunk가 아니라 `analysis_deep_agent` 또는 `general_answer` 노드의 최종 출력만 사용한다. 이 방식은 Deep Agent 내부 중간 멘트가 사용자 답변에 섞이는 것을 막는다.

## 확인한 동작

로컬에서 아래 항목을 확인했다.

- PostgreSQL `deepagent_chat` DB 생성
- 테이블 자동 생성
- 세션/메시지/trace CRUD 동작
- `harness` 환경에서 `psycopg`, `pydantic-settings` import

## 참고

PostgreSQL 명령어 예시와 테이블 설명은 `POSTGRESQL.md`를 보면 된다.
