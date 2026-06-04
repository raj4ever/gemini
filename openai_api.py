"""OpenAI-compatible API for AI editors and Hermes Agent (with tool calling)."""

from __future__ import annotations

import asyncio
import json
import os
import time
import uuid
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from agent_bridge import (
    build_choice,
    build_stream_tool_chunks,
    messages_to_agent_prompt,
    parse_tool_calls,
)
from gemini_webapi.constants import Model

API_KEY = os.environ.get("GEMINI_API_KEY", "sk-gemini-cookie-local")
HOST = os.environ.get("GEMINI_HOST", "127.0.0.1")
PORT = int(os.environ.get("GEMINI_PORT", "8765"))
BASE_URL = os.environ.get("GEMINI_BASE_URL", f"http://{HOST}:{PORT}/v1")

MODEL_ALIASES: dict[str, str] = {
    "gpt-4": "gemini-3-pro",
    "gpt-4o": "gemini-3-pro",
    "gpt-4o-mini": "gemini-3-flash",
    "gpt-4-turbo": "gemini-3-pro",
    "gpt-3.5-turbo": "gemini-3-flash",
    "gemini-pro": "gemini-3-pro",
    "gemini-1.5-pro": "gemini-3-pro",
    "gemini-1.5-flash": "gemini-3-flash",
    "gemini-2.0-flash": "gemini-3-flash",
    "gemini-2.5-pro": "gemini-3-pro",
    "gemini-2.5-flash": "gemini-3-flash",
    "gemini-flash": "gemini-3-flash",
    "gemini-flash-thinking": "gemini-3-flash-thinking",
    "default": "gemini-3-flash",
}

DEFAULT_MODEL = os.environ.get("GEMINI_DEFAULT_MODEL", "gemini-3-flash")

_bearer = HTTPBearer(auto_error=False)
router = APIRouter(prefix="/v1")


def verify_api_key(
    request: Request,
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    token: str | None = None
    if credentials and credentials.scheme.lower() == "bearer":
        token = credentials.credentials
    else:
        token = request.headers.get("x-api-key") or request.headers.get("api-key")
    if not token or token != API_KEY:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key. Use Authorization: Bearer <GEMINI_API_KEY>",
        )


def all_model_names() -> list[str]:
    names = [m.model_name for m in Model if m is not Model.UNSPECIFIED]
    names.extend(MODEL_ALIASES.keys())
    return sorted(set(names))


def resolve_model(name: str | None) -> str:
    raw = (name or DEFAULT_MODEL).strip()
    return MODEL_ALIASES.get(raw, raw)


def resolve_model_enum(name: str | None):
    resolved = resolve_model(name)
    try:
        return Model.from_name(resolved)
    except ValueError:
        return Model.UNSPECIFIED


def extract_message_text(content: str | list[Any] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif "text" in block:
                parts.append(str(block["text"]))
        elif isinstance(block, str):
            parts.append(block)
    return "\n".join(p for p in parts if p)


def messages_to_prompt(messages: list[dict[str, Any]]) -> str:
    """Simple chat prompt (no tools)."""
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "user")
        text = extract_message_text(msg.get("content"))
        if not text.strip() and role not in ("assistant", "tool"):
            continue
        if role == "system":
            lines.append(f"[System]\n{text}")
        elif role == "user":
            lines.append(f"[User]\n{text}")
        elif role == "assistant":
            lines.append(f"[Assistant]\n{text}")
        else:
            lines.append(f"[{role}]\n{text}")
    if not lines:
        return ""
    lines.append("[Assistant]")
    return "\n\n".join(lines)


class ChatMessage(BaseModel):
    role: str
    content: str | list[Any] | None = None
    tool_calls: list[Any] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=DEFAULT_MODEL)
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None


def _completion_id() -> str:
    return f"chatcmpl-{uuid.uuid4().hex[:24]}"


def _chunk(
    completion_id: str,
    model: str,
    delta: dict[str, Any] | None,
    finish_reason: str | None = None,
) -> str:
    payload = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        ],
    }
    return f"data: {json.dumps(payload)}\n\n"


def _build_prompt(body: ChatCompletionRequest) -> str:
    raw_messages = [m.model_dump(exclude_none=True) for m in body.messages]
    if body.tools:
        return messages_to_agent_prompt(raw_messages, body.tools, body.tool_choice)
    return messages_to_prompt(raw_messages)


async def _generate_content_text(prompt: str, model_enum: Any) -> str:
    """Non-stream Gemini call with lock, timeout, and 1097-style error recovery."""
    from server import (
        _SEND_TIMEOUT_SEC,
        _gemini_call_lock,
        _is_recoverable_gemini_error,
        gemini_client,
        persist_cookies,
        reset_gemini_client,
    )

    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not initialized")

    last_exc: Exception | None = None
    for attempt in range(2):
        client = gemini_client
        if not client:
            raise HTTPException(status_code=503, detail="Gemini client not initialized")
        try:
            async with _gemini_call_lock:
                response = await asyncio.wait_for(
                    client.generate_content(prompt, model=model_enum),
                    timeout=_SEND_TIMEOUT_SEC,
                )
            try:
                persist_cookies(client)
            except OSError:
                pass
            return response.text or ""
        except Exception as exc:
            last_exc = exc
            if attempt == 0 and _is_recoverable_gemini_error(exc):
                await reset_gemini_client()
                continue
            raise
    if last_exc:
        raise last_exc
    return ""


def _finish_agent_stream(
    completion_id: str,
    model_name: str,
    full: str,
) -> list[str]:
    lines: list[str] = []
    cleaned, tool_calls = parse_tool_calls(full)
    if cleaned:
        lines.append(
            _chunk(
                completion_id,
                model_name,
                {"role": "assistant", "content": cleaned},
            )
        )
    if tool_calls:
        lines.extend(build_stream_tool_chunks(completion_id, model_name, tool_calls))
    else:
        lines.append(_chunk(completion_id, model_name, {}, finish_reason="stop"))
        lines.append("data: [DONE]\n\n")
    return lines


@router.get("/models")
async def list_models(_: None = Depends(verify_api_key)) -> dict[str, Any]:
    models = []
    seen: set[str] = set()
    for name in all_model_names():
        if name in seen:
            continue
        seen.add(name)
        models.append(
            {
                "id": name,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "google-gemini-cookie",
            }
        )
    return {"object": "list", "data": models}


@router.post("/chat/completions")
async def chat_completions(
    body: ChatCompletionRequest,
    _: None = Depends(verify_api_key),
) -> Any:
    from server import gemini_client, persist_cookies

    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not initialized")

    prompt = _build_prompt(body)
    if not prompt.strip():
        raise HTTPException(status_code=400, detail="No messages with content")

    model_name = resolve_model(body.model)
    model_enum = resolve_model_enum(body.model)
    completion_id = _completion_id()
    agent_mode = bool(body.tools)

    if body.stream:
        from server import _gemini_call_lock, _is_recoverable_gemini_error, reset_gemini_client

        async def sse_stream() -> AsyncIterator[str]:
            full = ""
            try:
                async with _gemini_call_lock:
                    async for chunk in gemini_client.generate_content_stream(
                        prompt, model=model_enum
                    ):
                        delta = getattr(chunk, "text_delta", None) or ""
                        if delta:
                            full += delta
                            if not agent_mode:
                                yield _chunk(
                                    completion_id,
                                    model_name,
                                    {"role": "assistant", "content": delta},
                                )
                try:
                    persist_cookies(gemini_client)
                except OSError:
                    pass

                if agent_mode:
                    for line in _finish_agent_stream(completion_id, model_name, full):
                        yield line
                else:
                    yield _chunk(completion_id, model_name, {}, finish_reason="stop")
                    yield "data: [DONE]\n\n"
            except Exception as exc:
                if _is_recoverable_gemini_error(exc):
                    try:
                        await reset_gemini_client()
                        full = await _generate_content_text(prompt, model_enum)
                        if agent_mode:
                            for line in _finish_agent_stream(
                                completion_id, model_name, full
                            ):
                                yield line
                        else:
                            if full:
                                yield _chunk(
                                    completion_id,
                                    model_name,
                                    {"role": "assistant", "content": full},
                                )
                            yield _chunk(
                                completion_id, model_name, {}, finish_reason="stop"
                            )
                            yield "data: [DONE]\n\n"
                        return
                    except Exception as fallback_exc:
                        exc = fallback_exc
                yield _chunk(
                    completion_id,
                    model_name,
                    {"role": "assistant", "content": f"\n[Error: {exc}]"},
                )
                yield _chunk(completion_id, model_name, {}, finish_reason="stop")
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            sse_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    try:
        text = await _generate_content_text(prompt, model_enum)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    if agent_mode:
        cleaned, tool_calls = parse_tool_calls(text)
        choice = build_choice(cleaned, tool_calls)
    else:
        choice = {
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }

    return JSONResponse(
        {
            "id": completion_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            },
        }
    )
