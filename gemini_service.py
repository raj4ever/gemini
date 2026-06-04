"""Single Gemini execution path — non-stream by default (fast/local-stable, avoids 1097 on VPS)."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from fastapi import HTTPException
from gemini_webapi.constants import Model

# Google's stream API often returns 1097 from VPS/datacenter IPs. Local uses send_message (fast).
USE_NATIVE_STREAM = os.environ.get("GEMINI_NATIVE_STREAM", "false").lower() in (
    "1",
    "true",
    "yes",
)

SEND_TIMEOUT_SEC = float(os.environ.get("GEMINI_SEND_TIMEOUT", "120"))
MAX_ATTEMPTS = int(os.environ.get("GEMINI_MAX_ATTEMPTS", "3"))
RETRY_DELAY_SEC = float(os.environ.get("GEMINI_RETRY_DELAY", "2"))
FAKE_STREAM_CHUNK = int(os.environ.get("GEMINI_FAKE_STREAM_CHUNK", "48"))

_call_lock = asyncio.Lock()


def is_recoverable_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "1097" in msg
        or "failed to generate" in msg
        or "usage limit" in msg
        or "temporarily blocked" in msg
        or "closed" in msg
        or "1013" in msg
        or "unauthenticated" in msg
    )


def fake_stream_text(text: str, chunk_size: int = FAKE_STREAM_CHUNK) -> list[str]:
    if not text:
        return []
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


async def generate_text(prompt: str, model: Model | Any = Model.UNSPECIFIED) -> str:
    """OpenAI/Hermes: generate_content (non-stream) with retries."""
    from server import gemini_client, persist_cookies, reset_gemini_client

    if not prompt.strip():
        return ""

    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        client = gemini_client
        if client is None:
            raise HTTPException(status_code=503, detail="Gemini client not initialized")
        try:
            async with _call_lock:
                if USE_NATIVE_STREAM:
                    parts: list[str] = []
                    async for chunk in client.generate_content_stream(prompt, model=model):
                        delta = getattr(chunk, "text_delta", None) or ""
                        if delta:
                            parts.append(delta)
                    text = "".join(parts)
                else:
                    response = await asyncio.wait_for(
                        client.generate_content(prompt, model=model),
                        timeout=SEND_TIMEOUT_SEC,
                    )
                    text = response.text or ""
            try:
                persist_cookies(client)
            except OSError:
                pass
            return text
        except Exception as exc:
            last_exc = exc
            if attempt < MAX_ATTEMPTS - 1 and is_recoverable_error(exc):
                await asyncio.sleep(RETRY_DELAY_SEC)
                await reset_gemini_client()
                continue
            break

    if last_exc:
        raise last_exc
    raise HTTPException(status_code=502, detail="Gemini request failed")


async def send_chat_message(session_id: str, text: str) -> str:
    """Web UI /api/chat — chat.send_message (non-stream) with retries."""
    from server import chat_sessions, gemini_client, get_chat, persist_cookies, reset_gemini_client

    last_exc: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        if gemini_client is None:
            raise HTTPException(status_code=503, detail="Gemini client not initialized")
        try:
            async with _call_lock:
                chat = get_chat(session_id)
                if USE_NATIVE_STREAM:
                    parts: list[str] = []
                    async for chunk in chat.send_message_stream(text):
                        delta = getattr(chunk, "text_delta", None) or ""
                        if delta:
                            parts.append(delta)
                    reply = "".join(parts)
                else:
                    response = await asyncio.wait_for(
                        chat.send_message(text),
                        timeout=SEND_TIMEOUT_SEC,
                    )
                    reply = response.text or ""
            try:
                persist_cookies(gemini_client)
            except OSError:
                pass
            return reply
        except TimeoutError as exc:
            raise HTTPException(
                status_code=504,
                detail="Gemini reply timed out — try again",
            ) from exc
        except Exception as exc:
            last_exc = exc
            chat_sessions.pop(session_id, None)
            if attempt < MAX_ATTEMPTS - 1 and is_recoverable_error(exc):
                await asyncio.sleep(RETRY_DELAY_SEC)
                await reset_gemini_client()
                continue
            break

    if last_exc:
        raise last_exc
    raise HTTPException(status_code=502, detail="Gemini request failed")
