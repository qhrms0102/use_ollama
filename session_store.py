from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_title_from_messages(messages: list[dict[str, str]]) -> str:
    for message in messages:
        if message.get("role") != "user":
            continue

        content = " ".join(message.get("content", "").split()).strip()
        if content:
            return content[:36] + "..." if len(content) > 36 else content

    return "새 대화"


@dataclass
class ChatSessionStore:
    file_path: Path

    def __post_init__(self) -> None:
        self._lock = Lock()
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.file_path.exists():
            self.file_path.write_text("{}", encoding="utf-8")

    def _load(self) -> dict[str, dict[str, Any]]:
        try:
            raw = self.file_path.read_text(encoding="utf-8").strip()
            if not raw:
                return {}
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except (OSError, json.JSONDecodeError):
            pass

        return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self.file_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def create_session(self, session_id: str, title: str | None = None) -> dict[str, Any]:
        with self._lock:
            data = self._load()
            existing = data.get(session_id)
            if existing:
                return existing

            now = utc_now_iso()
            session = {
                "id": session_id,
                "title": title or "새 대화",
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
            data[session_id] = session
            self._save(data)
            return session

    def ensure_session(self, session_id: str) -> dict[str, Any]:
        return self.create_session(session_id)

    def append_message(self, session_id: str, role: str, content: str) -> dict[str, Any]:
        with self._lock:
            data = self._load()
            session = data.get(session_id)
            if not session:
                now = utc_now_iso()
                session = {
                    "id": session_id,
                    "title": "새 대화",
                    "created_at": now,
                    "updated_at": now,
                    "messages": [],
                }
                data[session_id] = session

            session_messages = session.setdefault("messages", [])
            session_messages.append({"role": role, "content": content})
            session["updated_at"] = utc_now_iso()
            session["title"] = make_title_from_messages(session_messages)
            self._save(data)
            return session

    def list_sessions(self) -> list[dict[str, Any]]:
        with self._lock:
            data = self._load()

        sessions = []
        for session in data.values():
            messages = session.get("messages", [])
            preview = ""
            for message in reversed(messages):
                content = " ".join(message.get("content", "").split()).strip()
                if content:
                    preview = content[:80] + "..." if len(content) > 80 else content
                    break

            sessions.append(
                {
                    "id": session["id"],
                    "title": session.get("title") or "새 대화",
                    "created_at": session.get("created_at"),
                    "updated_at": session.get("updated_at"),
                    "message_count": len(messages),
                    "preview": preview,
                }
            )

        sessions.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return sessions

    def get_session(self, session_id: str) -> dict[str, Any] | None:
        with self._lock:
            data = self._load()

        session = data.get(session_id)
        if not session:
            return None

        return {
            "id": session["id"],
            "title": session.get("title") or "새 대화",
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
            "messages": session.get("messages", []),
        }

    def delete_session(self, session_id: str) -> bool:
        with self._lock:
            data = self._load()
            if session_id not in data:
                return False

            del data[session_id]
            self._save(data)
            return True
