# Gemini Cookie Chat + OpenAI API

Local server: browser cookies se **gemini.google.com** chat, plus **OpenAI-compatible API** for Cursor, Continue, Cline, Roo, etc.

## Quick start (Docker — recommended with Hermes)

```bash
cd /Volumes/code/chrome
# Requires ~/.hermes/hermes-agent (installed automatically on first setup)
docker compose up -d --build
```

> **First start** takes 2–5 minutes while Hermes WebUI installs `hermes-agent` Python deps inside the container.

Web UI: **http://127.0.0.1:8765**

### Hermes Agent / Hermes WebUI (Docker) settings

Hermes WebUI container (`hermes-webui`) same Docker network par hai — yeh values use karo:

| Setting | Value |
|--------|--------|
| **Provider** | Custom endpoint (OpenAI-compatible) |
| **Base URL** | `http://gemini-cookie:8765/v1` |
| **API Key** | `sk-gemini-cookie-local` |
| **Model** | `gemini-3-flash` (ya `gemini-3-pro`) |

`~/.hermes/config.yaml` example: `hermes-gemini-config.example.yaml`

Hermes WebUI → Settings / Model → **Custom endpoint** → upar wale values.

Mac host se (terminal `hermes` CLI): `http://127.0.0.1:8765/v1`

---

## VPS / cloud (Hermes WebUI alag server par)

Problem: VPS par Hermes WebUI hai, lekin `http://127.0.0.1:8765` VPS ke andar Mac ka local API nahi pakadta. **Gemini Cookie API bhi usi VPS par Docker mein chalao**, phir Hermes ko container name se URL do.

### Step 1 — GitHub par repo push

Repo: **https://github.com/raj4ever/gemini**

### Step 2 — GitHub compose URL (Hermes / manual install)

Compose file ka **raw URL**:

```text
https://raw.githubusercontent.com/raj4ever/gemini/main/docker-compose.vps.yml
```

Hermes WebUI jahan “install from URL” ho, wahan yeh URL paste kar sakte ho — ya VPS par:

```bash
curl -fsSL https://raw.githubusercontent.com/raj4ever/gemini/main/install-vps.sh | bash
```

Ya manually:

```bash
mkdir -p ~/gemini-cookie/data && cd ~/gemini-cookie
curl -fsSL -o docker-compose.yml \
  "https://raw.githubusercontent.com/raj4ever/gemini/main/docker-compose.vps.yml"
# cookies.json → data/cookies.json (Mac se scp)
export GITHUB_REPO=raj4ever/gemini
docker network create hermes-net 2>/dev/null || true
docker compose up -d --build
docker network connect hermes-net hermes-webui
```

### Step 3 — Hermes WebUI model settings (VPS container)

| Setting | Value |
|--------|--------|
| **Base URL** | `http://gemini-cookie:8765/v1` |
| **API Key** | `sk-gemini-cookie-local` |
| **Model** | `gemini-3-flash` |

`127.0.0.1:8765` mat use karo jab Hermes bhi Docker mein ho — hostname **`gemini-cookie`** hona chahiye.

Agar WebUI ka network alag hai: `docker network connect hermes-net <webui-container-name>`

### Cookies VPS par

```bash
scp cookies.json user@YOUR_VPS:~/gemini-cookie/data/cookies.json
```

Example format: `cookies.json.example`

---

### Telegram gateway

Profile `gemini-cookie` ke liye `~/.hermes/profiles/gemini-cookie/.env` mein:

- `TELEGRAM_BOT_TOKEN` — BotFather token
- `TELEGRAM_ALLOWED_USERS` — aapka numeric user ID (DM ke liye; chat ID alag ho sakta hai groups mein)

Gateway container: `docker compose up -d hermes-gateway` (service `hermes-gateway` in `docker-compose.yml`). Logs: `docker logs -f hermes-gateway` ya `~/.hermes/profiles/gemini-cookie/logs/gateway.log`.

Bot ko Telegram par message bhej kar test karo. Pehli baar `/start` ya koi message.

### Hermes Agent mode (tools / terminal)

Cookie Gemini ab **agent backend** support karta hai: Hermes ke tools ko prompt mein bhej kar jawab se `tool_calls` parse kiye jate hain.

- Complex tasks slow ho sakte hain (bada prompt + multiple turns)
- Kabhi-kabhi model tool format miss kar sakta hai — dubara try karo ya `gemini-3-pro` use karo
- Simple commands (`ls`, `pwd`) sabse reliable hain

---

## Quick start (local Python)

```bash
cd /Volumes/code/chrome
python3 -m venv .venv
source .venv/bin/activate 
pip install -r requirements.txt
python server.py
```

## AI editor mein custom API (OpenAI format)

Server start par terminal mein **Base URL**, **API Key**, aur **Model** print hote hain.

| Setting | Value |
|--------|--------|
| **Base URL** | `http://127.0.0.1:8765/v1` |
| **API Key** | `sk-gemini-cookie-local` (default) ya apna `GEMINI_API_KEY` |
| **Model** | `gemini-3-pro`, `gemini-3-flash`, `gemini-3-flash-thinking`, … |

### Apni API key set karna

```bash
export GEMINI_API_KEY="sk-my-secret-key-12345"
python server.py
```

### Available models

| Model ID | Description |
|----------|-------------|
| `gemini-3-pro` | Pro (basic tier) |
| `gemini-3-flash` | Fast |
| `gemini-3-flash-thinking` | Thinking |
| `gemini-3-pro-plus` | Pro Plus |
| `gemini-3-flash-plus` | Flash Plus |
| `gemini-3-pro-advanced` | Pro Advanced |
| `gemini-3-flash-advanced` | Flash Advanced |

Aliases bhi chalte hain: `gpt-4o` → `gemini-3-pro`, `gpt-4o-mini` → `gemini-3-flash`.

Model list: `GET http://127.0.0.1:8765/v1/models` with header `Authorization: Bearer <API_KEY>`.

---

### Cursor

1. **Settings** → **Models** → **OpenAI API Key** section ya custom OpenAI-compatible provider
2. **Override OpenAI Base URL**: `http://127.0.0.1:8765/v1`
3. **API Key**: `sk-gemini-cookie-local` (ya jo `GEMINI_API_KEY` set kiya ho)
4. **Model**: `gemini-3-pro` ya `gemini-3-flash`

> Cursor version ke hisaab se UI alag ho sakta hai — "OpenAI-compatible" / "Custom endpoint" dhundho.

### Continue (VS Code)

`~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Gemini Cookie",
      "provider": "openai",
      "model": "gemini-3-pro",
      "apiKey": "sk-gemini-cookie-local",
      "apiBase": "http://127.0.0.1:8765/v1"
    }
  ]
}
```

### Cline / Roo Code

- **API Provider**: OpenAI Compatible  
- **Base URL**: `http://127.0.0.1:8765/v1`  
- **API Key**: `sk-gemini-cookie-local`  
- **Model**: `gemini-3-flash`

### curl test

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Authorization: Bearer sk-gemini-cookie-local" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3-flash",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Streaming:

```bash
curl http://127.0.0.1:8765/v1/chat/completions \
  -H "Authorization: Bearer sk-gemini-cookie-local" \
  -H "Content-Type: application/json" \
  -d '{"model":"gemini-3-flash","stream":true,"messages":[{"role":"user","content":"Hi"}]}'
```

---

## Web chat UI

Browser: **http://127.0.0.1:8765**

## Cookies

`cookies.json` (gitignored) — DevTools se export. Expire hone par refresh karo.

## Security

- Cookies = full Google login — share/commit mat karo  
- API key sirf local machine par rakho  
- Unofficial API — Google kabhi break kar sakta hai
