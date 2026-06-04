FROM python:3.12-slim-bookworm

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py openai_api.py agent_bridge.py ./
COPY public ./public/

ENV GEMINI_HOST=0.0.0.0
ENV GEMINI_PORT=8765
ENV GEMINI_COOKIES_PATH=/app/cookies.json
ENV GEMINI_API_KEY=sk-gemini-cookie-local
ENV GEMINI_DEFAULT_MODEL=gemini-3-flash

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -sf http://127.0.0.1:8765/api/health >/dev/null || exit 1

CMD ["python", "server.py"]
