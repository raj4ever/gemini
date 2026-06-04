"""Local Gemini chat server using cookies from cookies.json."""

from __future__ import annotations

import json
import os
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


def parse_cookies_file() -> dict[str, str]:
    if not COOKIES_PATH.exists():
        raise FileNotFoundError(
            f"{COOKIES_PATH} not found. Copy cookies.example.json to cookies.json and paste your cookies."
        )
    raw = json.loads(COOKIES_PATH.read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return {item["name"]: item["value"] for item in raw if item.get("name") and item.get("value")}
    if isinstance(raw, dict):
        return raw
    raise ValueError("cookies.json must be a list of {name, value} objects or a flat object")


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
    cookies = parse_cookies_file()
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
    return {
        "ok": gemini_client is not None,
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
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not initialized — check cookies.json")
    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    session_id = req.session_id
    if session_id not in chat_sessions:
        chat_sessions[session_id] = gemini_client.start_chat()
    chat = chat_sessions[session_id]

    try:
        response = await chat.send_message(text)
        try:
            persist_cookies(gemini_client)
        except OSError:
            pass
        return {"text": response.text or "", "session_id": session_id}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest) -> StreamingResponse:
    if not gemini_client:
        raise HTTPException(status_code=503, detail="Gemini client not initialized")

    text = req.message.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Empty message")

    session_id = req.session_id
    if session_id not in chat_sessions:
        chat_sessions[session_id] = gemini_client.start_chat()
    chat = chat_sessions[session_id]

    async def event_stream():
        full = ""
        try:
            async for chunk in chat.send_message_stream(text):
                delta = getattr(chunk, "text_delta", None) or ""
                if delta:
                    full += delta
                    yield f"data: {json.dumps({'delta': delta})}\n\n"
            try:
                persist_cookies(gemini_client)
            except OSError:
                pass
            yield f"data: {json.dumps({'done': True, 'text': full})}\n\n"
        except Exception as exc:
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
