# PostgreSQL Quick Guide

이 문서는 로컬 네이티브 PostgreSQL 기준으로 `deepagent_chat` 데이터베이스를 다루는 최소 명령과 예시를 정리한 것이다.

## 1. 서비스 관리

```bash
brew services list | rg postgresql
brew services start postgresql@16
brew services restart postgresql@16
brew services stop postgresql@16
```

## 2. 기본 접속

`psql`이 `PATH`에 잡혀 있다면 아래처럼 바로 접속하면 된다.

```bash
psql -d postgres
```

접속 확인 예시:

```bash
psql -d postgres -c "select current_user, current_database();"
```

만약 `psql` 명령을 찾지 못하면 PostgreSQL 설치 경로를 직접 써도 된다.

Homebrew 예시:

```bash
/opt/homebrew/opt/postgresql@16/bin/psql -d postgres
```

## 3. 앱용 DB 생성

프로젝트 DB 이름은 `deepagent_chat`으로 사용한다.

```bash
createdb deepagent_chat
psql -d deepagent_chat -c "select current_database();"
```

이미 만들어져 있는지 확인:

```bash
psql -lqt | grep deepagent_chat
```

삭제 후 다시 만들기:

```bash
dropdb deepagent_chat
createdb deepagent_chat
```

## 4. 자주 쓰는 psql 명령

`psql`에 들어간 뒤 사용하는 메타 명령:

```text
\l           -- DB 목록
\c deepagent_chat
\dt          -- 테이블 목록
\d chat_sessions  -- 테이블 구조
\q           -- 종료
```

예시:

```bash
psql -d deepagent_chat
```

```sql
\dt
select now();
\q
```

## 5. 권장 테이블 이름

현재 챗봇 구조라면 아래 3개로 시작하는 것이 가장 무난하다.

- `chat_sessions`
- `chat_messages`
- `chat_message_traces`

이유:

- `sessions`, `messages`처럼 너무 일반적인 이름보다 충돌 가능성이 낮다
- 테이블 역할이 이름만 보고 바로 드러난다
- 추후 다른 기능용 테이블이 추가되어도 구분이 쉽다

권장 관계:

- `chat_sessions`: 대화방 단위
- `chat_messages`: 사용자/assistant 메시지 단위
- `chat_message_traces`: assistant 메시지에 연결된 작업 기록 단위

## 6. 추천 스키마 방향

### `chat_sessions`

- `id`: `text` 또는 `uuid`
- `title`: `text`
- `created_at`: `timestamptz`
- `updated_at`: `timestamptz`

### `chat_messages`

- `id`: `bigserial` 또는 `uuid`
- `session_id`: `chat_sessions.id` FK
- `role`: `text`
- `content`: `text`
- `created_at`: `timestamptz`

### `chat_message_traces`

- `id`: `bigserial` 또는 `uuid`
- `message_id`: `chat_messages.id` FK
- `trace_order`: 메시지 내부 순서
- `trace_uid`: 프런트 표시용 trace id
- `trace_kind`: `text`
- `trace_name`: `text`
- `status`: `text`
- `summary`: `text`
- `detail`: `text`
- `agent_name`: `text`
- `subagent_name`: `text`
- `input_payload`: `jsonb`
- `output_payload`: `jsonb`
- `created_at`: `timestamptz`

## 7. SQL 예시

테이블 생성 예시:

```sql
create table if not exists chat_sessions (
  id text primary key,
  title text not null default '새 대화',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table if not exists chat_messages (
  id bigserial primary key,
  session_id text not null references chat_sessions(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null default '',
  created_at timestamptz not null default now()
);

create table if not exists chat_message_traces (
  id bigserial primary key,
  message_id bigint not null references chat_messages(id) on delete cascade,
  trace_order integer not null,
  trace_uid text not null,
  trace_kind text,
  trace_name text,
  status text not null,
  summary text,
  detail text,
  agent_name text,
  subagent_name text,
  input_payload jsonb,
  output_payload jsonb,
  created_at timestamptz not null default now()
);

create index if not exists idx_chat_messages_session_id
  on chat_messages(session_id, created_at);

create index if not exists idx_chat_message_traces_message_id
  on chat_message_traces(message_id, trace_order);
```

## 8. 환경 변수 예시

앱에서는 아래 분리형 환경 변수를 사용한다.

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=deepagent_chat
DB_USER=bogeun
DB_PASSWORD=
```

비밀번호가 있는 별도 계정을 만들면:

```env
DB_HOST=localhost
DB_PORT=5432
DB_NAME=deepagent_chat
DB_USER=app_user
DB_PASSWORD=secret
```

## 9. 추천 운영 방식

지금 단계에서는 다음 기준으로 가면 된다.

- DB 이름: `deepagent_chat`
- 테이블 이름: `chat_sessions`, `chat_messages`, `chat_message_traces`
- 애플리케이션 설정: `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`

이 구조면 이후 `session_store.py`를 PostgreSQL 기반으로 바꿀 때도 모델이 단순하고 확장하기 쉽다.
