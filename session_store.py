from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import psycopg
from psycopg.rows import dict_row


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_title_from_messages(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue

        content = " ".join(str(message.get("content", "")).split()).strip()
        if content:
            return content[:36] + "..." if len(content) > 36 else content

    return "새 대화"


@dataclass
class ChatSessionStore:
    database_url: str

    def __post_init__(self) -> None:
        # 앱 시작 시 필요한 테이블과 인덱스를 자동으로 맞춘다.
        self._ensure_schema()

    def _connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _ensure_schema(self) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                # 세션, 메시지, trace를 분리해 저장해 두면
                # 목록 조회와 상세 복원이 단순해진다.
                cur.execute(
                    """
                    create table if not exists chat_sessions (
                      id text primary key,
                      title text not null default '새 대화',
                      created_at timestamptz not null default now(),
                      updated_at timestamptz not null default now()
                    )
                    """
                )
                cur.execute(
                    """
                    create table if not exists chat_messages (
                      id bigserial primary key,
                      session_id text not null references chat_sessions(id) on delete cascade,
                      role text not null check (role in ('user', 'assistant')),
                      content text not null default '',
                      created_at timestamptz not null default now()
                    )
                    """
                )
                cur.execute(
                    """
                    create table if not exists chat_message_traces (
                      id bigserial primary key,
                      message_id bigint not null references chat_messages(id) on delete cascade,
                      trace_order integer not null,
                      trace_uid text not null,
                      call_id text,
                      trace_kind text,
                      trace_name text,
                      title text not null,
                      status text not null,
                      summary text,
                      detail text,
                      agent_name text,
                      subagent_name text,
                      input_payload jsonb,
                      output_payload jsonb,
                      trace_timestamp bigint,
                      created_at timestamptz not null default now()
                    )
                    """
                )
                cur.execute(
                    """
                    create index if not exists idx_chat_sessions_updated_at
                      on chat_sessions(updated_at desc)
                    """
                )
                cur.execute(
                    """
                    create index if not exists idx_chat_messages_session_id_created_at
                      on chat_messages(session_id, created_at, id)
                    """
                )
                cur.execute(
                    """
                    create index if not exists idx_chat_message_traces_message_id_trace_order
                      on chat_message_traces(message_id, trace_order)
                    """
                )

    def create_session(self, session_id: str, title: str | None = None) -> dict[str, Any]:
        now = utc_now_iso()
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into chat_sessions (id, title, created_at, updated_at)
                    values (%s, %s, %s, %s)
                    on conflict (id) do nothing
                    """,
                    (session_id, title or "새 대화", now, now),
                )
                cur.execute(
                    """
                    select id, title, created_at, updated_at
                    from chat_sessions
                    where id = %s
                    """,
                    (session_id,),
                )
                session = cur.fetchone()

        return {
            "id": session["id"],
            "title": session["title"],
            "created_at": session["created_at"].isoformat(),
            "updated_at": session["updated_at"].isoformat(),
            "messages": [],
        }

    def ensure_session(self, session_id: str) -> dict[str, Any]:
        return self.create_session(session_id)

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        traces: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        # 메시지 저장 전에 세션이 없으면 먼저 만든다.
        self.create_session(session_id)

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into chat_messages (session_id, role, content)
                    values (%s, %s, %s)
                    returning id
                    """,
                    (session_id, role, content),
                )
                message_id = cur.fetchone()["id"]

                if traces:
                    # assistant 응답에 연결된 작업 기록을 순서대로 보존한다.
                    for index, trace in enumerate(traces):
                        cur.execute(
                            """
                            insert into chat_message_traces (
                              message_id,
                              trace_order,
                              trace_uid,
                              call_id,
                              trace_kind,
                              trace_name,
                              title,
                              status,
                              summary,
                              detail,
                              agent_name,
                              subagent_name,
                              input_payload,
                              output_payload,
                              trace_timestamp
                            )
                            values (
                              %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                              %s::jsonb, %s::jsonb, %s
                            )
                            """,
                            (
                                message_id,
                                index,
                                trace.get("id") or f"{message_id}-{index}",
                                trace.get("call_id"),
                                trace.get("kind"),
                                trace.get("name"),
                                trace.get("title") or trace.get("name") or "trace",
                                trace.get("status") or "done",
                                trace.get("summary"),
                                trace.get("detail"),
                                trace.get("agent_name"),
                                trace.get("subagent_name"),
                                self._json_value(trace.get("input")),
                                self._json_value(trace.get("output")),
                                trace.get("timestamp"),
                            ),
                        )

                # 세션 제목은 첫 user 메시지 기준으로 다시 계산한다.
                title = self._recompute_title(cur, session_id)
                cur.execute(
                    """
                    update chat_sessions
                    set title = %s, updated_at = now()
                    where id = %s
                    returning id, title, created_at, updated_at
                    """,
                    (title, session_id),
                )
                session = cur.fetchone()

        return {
            "id": session["id"],
            "title": session["title"],
            "created_at": session["created_at"].isoformat(),
            "updated_at": session["updated_at"].isoformat(),
        }

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._connect() as conn:
            with conn.cursor() as cur:
                # 목록 화면에서는 최신 미리보기와 메시지 개수만 빠르게 가져온다.
                cur.execute(
                    """
                    select
                      s.id,
                      s.title,
                      s.created_at,
                      s.updated_at,
                      count(m.id)::int as message_count,
                      coalesce(
                        (
                          select case
                            when length(trim(m2.content)) > 80 then left(trim(m2.content), 80) || '...'
                            else trim(m2.content)
                          end
                          from chat_messages m2
                          where m2.session_id = s.id
                            and trim(m2.content) <> ''
                          order by m2.created_at desc, m2.id desc
                          limit 1
                        ),
                        ''
                      ) as preview
                    from chat_sessions s
                    left join chat_messages m on m.session_id = s.id
                    group by s.id, s.title, s.created_at, s.updated_at
                    order by s.updated_at desc
                    """
                )
                rows = cur.fetchall()

        return [
            {
                "id": row["id"],
                "title": row["title"] or "새 대화",
                "created_at": row["created_at"].isoformat(),
                "updated_at": row["updated_at"].isoformat(),
                "message_count": row["message_count"],
                "preview": row["preview"] or "",
            }
            for row in rows
        ]

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    select id, title, created_at, updated_at
                    from chat_sessions
                    where id = %s
                    """,
                    (session_id,),
                )
                session = cur.fetchone()
                if not session:
                    return None

                # 메시지와 trace를 한 번에 묶어서 프런트 복원 형태로 가져온다.
                cur.execute(
                    """
                    select
                      m.id,
                      m.role,
                      m.content,
                      coalesce(
                        json_agg(
                          json_build_object(
                            'id', t.trace_uid,
                            'call_id', t.call_id,
                            'kind', t.trace_kind,
                            'name', t.trace_name,
                            'title', t.title,
                            'status', t.status,
                            'summary', t.summary,
                            'detail', t.detail,
                            'agent_name', t.agent_name,
                            'subagent_name', t.subagent_name,
                            'input', t.input_payload,
                            'output', t.output_payload,
                            'timestamp', t.trace_timestamp
                          )
                          order by t.trace_order
                        ) filter (where t.id is not null),
                        '[]'::json
                      ) as traces
                    from chat_messages m
                    left join chat_message_traces t on t.message_id = m.id
                    where m.session_id = %s
                    group by m.id, m.role, m.content, m.created_at
                    order by m.created_at asc, m.id asc
                    """,
                    (session_id,),
                )
                messages = cur.fetchall()

        return {
            "id": session["id"],
            "title": session["title"] or "새 대화",
            "created_at": session["created_at"].isoformat(),
            "updated_at": session["updated_at"].isoformat(),
            "messages": [
                {
                    "role": message["role"],
                    "content": message["content"],
                    **({"traces": message["traces"]} if message["traces"] else {}),
                }
                for message in messages
            ],
        }

    def delete_session(self, session_id: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("delete from chat_sessions where id = %s", (session_id,))
                deleted = cur.rowcount > 0
        return deleted

    def _recompute_title(self, cur: Any, session_id: str) -> str:
        cur.execute(
            """
            select role, content
            from chat_messages
            where session_id = %s
            order by created_at asc, id asc
            """,
            (session_id,),
        )
        return make_title_from_messages(cur.fetchall())

    @staticmethod
    def _json_value(value: Any) -> str | None:
        if value is None:
            return None
        # trace payload는 jsonb 컬럼에 넣기 전에 문자열로 직렬화한다.
        return json.dumps(value, ensure_ascii=False)
