"""Local Gemini chat server using cookies from cookies.json."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import shutil
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from gemini_webapi import GeminiClient
from openai_api import API_KEY, BASE_URL, DEFAULT_MODEL, PORT, router as openai_router

ROOT = Path(__file__).resolve().parent
COOKIES_PATH = Path(os.environ.get("GEMINI_COOKIES_PATH", ROOT / "cookies.json"))
PUBLIC_DIR = ROOT / "public"
CACHE_DIR = ROOT / ".gemini_cache"

gemini_client: GeminiClient | None = None
chat_sessions: dict[str, object] = {}
_reset_lock = asyncio.Lock()
_last_reset_at = 0.0
_RESET_COOLDOWN_SEC = 15.0


def _is_recoverable_gemini_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return (
        "1097" in msg
        or "failed to generate" in msg
        or "usage limit" in msg
        or "temporarily blocked" in msg
        or "closed" in msg
    )


async def reset_gemini_client() -> GeminiClient:
    """Re-init after gemini_webapi calls client.close() on API errors."""
    global gemini_client, chat_sessions, _last_reset_at
    async with _reset_lock:
        now = time.monotonic()
        if now - _last_reset_at < _RESET_COOLDOWN_SEC and gemini_client is not None:
            return gemini_client
        _last_reset_at = now
        chat_sessions.clear()
        if gemini_client is not None:
            try:
                await gemini_client.close()
            except Exception:
                pass
        gemini_client = await init_gemini()
        return gemini_client


def get_chat(session_id: str):
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not initialized")
    if session_id not in chat_sessions:
        chat_sessions[session_id] = gemini_client.start_chat()
    return chat_sessions[session_id]


async def send_message_with_retry(session_id: str, text: str):
    last_exc: Exception | None = None
    for attempt in range(2):
        try:
            chat = get_chat(session_id)
            return await chat.send_message(text)
        except Exception as exc:
            last_exc = exc
            if attempt == 0 and _is_recoverable_gemini_error(exc):
                chat_sessions.pop(session_id, None)
                await reset_gemini_client()
                continue
            raise
    if last_exc:
        raise last_exc
    raise HTTPException(status_code=502, detail="Gemini request failed")


def _cookies_from_raw(raw: object) -> dict[str, str]:
    if isinstance(raw, list):
        return {item["name"]: item["value"] for item in raw if item.get("name") and item.get("value")}
    if isinstance(raw, dict):
        return raw
    raise ValueError("cookies must be a list of {name, value} objects or a flat object")


def parse_cookies_file() -> dict[str, str]:
    if not COOKIES_PATH.exists():
        raise FileNotFoundError(
            f"{COOKIES_PATH} not found. Copy cookies.example.json to cookies.json and paste your cookies."
        )
    if COOKIES_PATH.is_dir():
        raise FileNotFoundError(
            f"{COOKIES_PATH} is a directory (bad Docker mount). Remove it or set GEMINI_COOKIES_B64."
        )
    return _cookies_from_raw(json.loads(COOKIES_PATH.read_text(encoding="utf-8")))


def load_cookies() -> dict[str, str]:
    """Load cookies from GEMINI_COOKIES_B64 env (VPS deploy) or cookies.json file."""
    b64 = os.environ.get("GEMINI_COOKIES_B64", "").strip()
    if b64:
        raw = json.loads(base64.b64decode(b64))
        cookies = _cookies_from_raw(raw)
        try:
            if COOKIES_PATH.is_dir():
                shutil.rmtree(COOKIES_PATH)
            if not COOKIES_PATH.exists():
                COOKIES_PATH.parent.mkdir(parents=True, exist_ok=True)
                payload = raw if isinstance(raw, list) else [
                    {"name": k, "value": v} for k, v in cookies.items()
                ]
                COOKIES_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except OSError:
            pass
        return cookies
    return parse_cookies_file()


def persist_cookies(client: GeminiClient) -> None:
    jar = client.cookies
    items: list[tuple[str, str]] = []
    if hasattr(jar, "items"):
        items = list(jar.items())  # type: ignore[arg-type]
    elif isinstance(jar, dict):
        items = list(jar.items())
    else:
        for cookie in jar:
            items.append((cookie.name, cookie.value))
    COOKIES_PATH.write_text(
        json.dumps([{"name": k, "value": v} for k, v in items], indent=2),
        encoding="utf-8",
    )


async def init_gemini() -> GeminiClient:
    cookies = load_cookies()
    psid = cookies.get("__Secure-1PSID", "")
    psidts = cookies.get("__Secure-1PSIDTS", "")
    if not psid:
        raise ValueError("cookies.json must include __Secure-1PSID")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["GEMINI_COOKIE_PATH"] = str(CACHE_DIR)

    client = GeminiClient(psid, psidts or None)
    client.cookies = cookies
    await client.init(timeout=120, auto_close=False, auto_refresh=True)
    return client


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global gemini_client
    try:
        gemini_client = await init_gemini()
        print("Gemini client ready.")
    except Exception as exc:
        gemini_client = None
        print(f"Gemini init failed: {exc}")
    yield
    if gemini_client:
        await gemini_client.close()


app = FastAPI(title="Gemini Cookie Chat", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(openai_router)


class ChatRequest(BaseModel):
    message: str
    session_id: str = Field(default="default")


class NewSessionResponse(BaseModel):
    session_id: str


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict[str, object]:
    probes_rejected: list[str] = []
    if gemini_client is not None:
        try:
            status = await gemini_client.inspect_account_status()
            probes_rejected = (status.get("summary") or {}).get("rejected_probes") or []
        except Exception:
            pass
    return {
        "ok": gemini_client is not None,
        "probes_rejected": probes_rejected,
        "cookies_path": str(COOKIES_PATH),
        "openai_base_url": BASE_URL,
        "api_key_hint": f"{API_KEY[:12]}..." if len(API_KEY) > 12 else "(set GEMINI_API_KEY)",
        "default_model": DEFAULT_MODEL,
    }


@app.get("/api/hello")
async def hello() -> dict[str, str]:
    """Quick cloud check — browser ya curl se 'hello' verify karo."""
    if not gemini_client:
        raise HTTPException(
            status_code=503,
            detail="Gemini not ready — cookies.json check karo (see /api/health)",
        )
    try:
        chat = gemini_client.start_chat()
        response = await chat.send_message("Say hello in one short friendly sentence.")
        text = (response.text or "").strip()
        try:
            persist_cookies(gemini_client)
        except OSError:
            pass
        return {"status": "ok", "message": text or "hello"}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/session/new", response_model=NewSessionResponse)
async def new_session() -> NewSessionResponse:
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not initialized")
    session_id = str(uuid.uuid4())
    chat_sessions[session_id] = gemini_client.start_chat()
    return NewSessionResponse(session_id=session_id)


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict[str, str]:
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    try:
        response = await send_message_with_retry(req.session_id, text)
        try:
            if gemini_client:
                persist_cookies(gemini_client)
        except OSError:
            pass
        return {"text": response.text or "", "session_id": req.session_id}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    session_id = req.session_id

    async def event_stream():
        full = ""
        try:
            chat = get_chat(session_id)
            async for chunk in chat.send_message_stream(text):
                delta = getattr(chunk, "text_delta", None) or ""
                if delta:
                    full += delta
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
            try:
                if gemini_client:
                    persist_cookies(gemini_client)
            except OSError:
                pass
            yield f"data: {json.dumps({'done': True, 'text': full})}\n\n"
        except Exception as exc:
            if _is_recoverable_gemini_error(exc):
                try:
                    chat_sessions.pop(session_id, None)
                    response = await send_message_with_retry(session_id, text)
                    full = response.text or ""
                    yield f"data: {json.dumps({'delta': full})}\n\n"
                    yield f"data: {json.dumps({'done': True, 'text': full})}\n\n"
                    return
                except Exception as retry_exc:
                    yield f"data: {json.dumps({'error': str(retry_exc)})}\n\n"
                    return
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("GEMINI_HOST", "127.0.0.1")
    port = int(os.environ.get("GEMINI_PORT", "8765"))
    print(f"OpenAI-compatible API: {BASE_URL}")
    print(f"API key: {API_KEY}")
    print(f"Default model: {DEFAULT_MODEL}")
    uvicorn.run("server:app", host=host, port=port, reload=False)
