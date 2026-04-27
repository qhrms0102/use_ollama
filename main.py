import json
import uvicorn
import logging
import re
import uuid
import ast
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from config.settings import settings
from deep_agent_with_langgraph import create_agent_async
from session_store import ChatSessionStore

app = FastAPI(title="Deep Agent Context API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

chat_store = ChatSessionStore(settings.database_url)


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SessionCreateRequest(BaseModel):
    title: str | None = None


# ==============================
# 🔹 Utility
# ==============================

def clean_preview_text(text: str, max_len: int = 160) -> str:
    text = text.replace("\\n", "\n").strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    if len(text) > max_len:
        return text[:max_len].rstrip() + "..."
    return text


def extract_repr_content(text: str, max_len: int = 160) -> str | None:
    for pattern in (
        r"ToolMessage\(content=(['\"])((?:\\.|(?!\1).)*?)\1",
        r"content=(['\"])((?:\\.|(?!\1).)*?)\1",
    ):
        m = re.search(pattern, text, re.DOTALL)
        if m:
            quote = m.group(1)
            raw_content = m.group(2)
            try:
                raw_content = ast.literal_eval(f"{quote}{raw_content}{quote}")
            except (SyntaxError, ValueError):
                pass
            return clean_preview_text(raw_content, max_len=max_len)

    return None


def extract_human_content(output: Any, max_len: int = 160) -> str | None:
    if output is None:
        return None

    if isinstance(output, str):
        text = output.strip()
        content = extract_repr_content(text, max_len=max_len)
        if content:
            return content

        return clean_preview_text(text, max_len=max_len)

    if isinstance(output, dict):
        if "content" in output and isinstance(output["content"], str):
            return clean_preview_text(output["content"], max_len=max_len)

        if {"cond", "temp", "unit"} <= set(output.keys()):
            return f"{output['cond']}, {output['temp']}°{output['unit']}"

        return clean_preview_text(str(output), max_len=max_len)

    if hasattr(output, "content"):
        try:
            content = getattr(output, "content")
            if isinstance(content, str) and content.strip():
                return clean_preview_text(content, max_len=max_len)
        except Exception:
            pass

    content = extract_repr_content(str(output), max_len=max_len)
    if content:
        return content

    return clean_preview_text(str(output), max_len=max_len)


def safe_jsonable(data: Any):
    try:
        json.dumps(data, ensure_ascii=False)
        return data
    except Exception:
        return str(data)


def extract_agent_name(event: dict[str, Any]) -> str:
    metadata = event.get("metadata") or {}
    if isinstance(metadata, dict):
        agent_name = metadata.get("lc_agent_name")
        if isinstance(agent_name, str) and agent_name.strip():
            return agent_name
    return "main-agent"


def utc_now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


# ==============================
# 🔹 Streaming
# ==============================

async def generate_chat_events(session_id: str, message: str):
    agent = create_agent_async()
    session = chat_store.ensure_session(session_id)
    history = list(session.get("messages", []))
    history.append({"role": "user", "content": message})
    chat_store.append_message(session_id, "user", message)

    active_tools = 0
    final_answer_buffer = ""
    subagent_call_names: dict[str, str] = {}
    trace_buffer: list[dict[str, Any]] = []

    try:
        async for event in agent.astream_events({"messages": history}, version="v2"):
            kind = event["event"]

            # ======================
            # 🔹 LLM Streaming
            # ======================
            if kind == "on_chat_model_stream":
                if active_tools == 0:
                    chunk = event["data"]["chunk"]
                    if chunk.content and isinstance(chunk.content, str):
                        final_answer_buffer += chunk.content
                        yield f"data: {json.dumps({'type': 'chunk', 'content': chunk.content}, ensure_ascii=False)}\n\n"

            # ======================
            # 🔹 Tool Start
            # ======================
            elif kind == "on_tool_start":
                active_tools += 1
                name = event["name"]
                tool_input = event["data"].get("input", {})
                call_id = str(event["data"].get("tool_call_id") or event.get("run_id") or f"{name}-{active_tools}")
                agent_name = extract_agent_name(event)
                subagent_name = None
                if isinstance(tool_input, dict):
                    raw_subagent_name = tool_input.get("subagent_type")
                    if isinstance(raw_subagent_name, str) and raw_subagent_name.strip():
                        subagent_name = raw_subagent_name
                        subagent_call_names[call_id] = subagent_name

                payload = {
                    "type": "trace",
                    "trace": {
                        "id": uuid.uuid4().hex,
                        "call_id": call_id,
                        "kind": "tool_start",
                        "name": name,
                        "title": name,
                        "status": "running",
                        "summary": f"{name} 실행 중...",
                        "agent_name": agent_name,
                        "subagent_name": subagent_name,
                        "input": safe_jsonable(tool_input),
                        "timestamp": utc_now_ms(),
                    },
                }
                trace_buffer.append(payload["trace"])

                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

            # ======================
            # 🔹 Tool End
            # ======================
            elif kind == "on_tool_end":
                active_tools = max(0, active_tools - 1)
                name = event["name"]
                output = event["data"].get("output")
                call_id = str(event["data"].get("tool_call_id") or event.get("run_id") or f"{name}-done")
                agent_name = extract_agent_name(event)
                subagent_name = subagent_call_names.pop(call_id, None)

                summary = extract_human_content(output)
                detail = extract_human_content(output, max_len=4000)

                payload = {
                    "type": "trace",
                    "trace": {
                        "id": uuid.uuid4().hex,
                        "call_id": call_id,
                        "kind": "tool_end",
                        "name": name,
                        "title": name,
                        "status": "done",
                        "summary": summary,
                        "detail": detail,
                        "agent_name": agent_name,
                        "subagent_name": subagent_name,
                        "output": safe_jsonable(output),
                        "timestamp": utc_now_ms(),
                    },
                }
                trace_buffer.append(payload["trace"])

                yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    except Exception as e:
        logging.exception("Streaming error")
        error_trace = {
            "id": uuid.uuid4().hex,
            "title": "stream_error",
            "status": "error",
            "summary": str(e),
            "timestamp": utc_now_ms(),
        }
        trace_buffer.append(error_trace)
        yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    finally:
        assistant_content = final_answer_buffer.strip()
        if assistant_content or trace_buffer:
            chat_store.append_message(
                session_id,
                "assistant",
                assistant_content,
                traces=trace_buffer,
            )


@app.get("/api/sessions")
async def list_sessions_endpoint():
    return {"sessions": chat_store.list_sessions()}


@app.post("/api/sessions")
async def create_session_endpoint(request: SessionCreateRequest | None = None):
    session_id = uuid.uuid4().hex
    session = chat_store.create_session(session_id, title=request.title if request else None)
    return {
        "id": session["id"],
        "title": session["title"],
        "created_at": session["created_at"],
        "updated_at": session["updated_at"],
        "messages": session["messages"],
    }


@app.get("/api/sessions/{session_id}")
async def get_session_endpoint(session_id: str):
    session = chat_store.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="session_not_found")
    return session


@app.delete("/api/sessions/{session_id}")
async def delete_session_endpoint(session_id: str):
    if not chat_store.delete_session(session_id):
        raise HTTPException(status_code=404, detail="session_not_found")
    return {"ok": True}


@app.post("/api/chat/stream")
async def chat_stream_endpoint(request: ChatRequest):
    return StreamingResponse(
        generate_chat_events(request.session_id, request.message),
        media_type="text/event-stream",
    )


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
