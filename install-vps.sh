#!/usr/bin/env bash
# VPS par Gemini Cookie API install (Hermes WebUI cloud + API same server)
set -euo pipefail

GITHUB_USER="${GITHUB_USER:-raj4ever}"
GITHUB_REPO="${GITHUB_REPO:-gemini}"
BRANCH="${BRANCH:-main}"
HERMES_WEBUI_CONTAINER="${HERMES_WEBUI_CONTAINER:-hermes-webui}"
HERMES_NETWORK="${HERMES_DOCKER_NETWORK:-hermes-net}"
INSTALL_DIR="${INSTALL_DIR:-$HOME/gemini-cookie}"

COMPOSE_URL="https://raw.githubusercontent.com/${GITHUB_USER}/${GITHUB_REPO}/${BRANCH}/docker-compose.vps.yml"

echo "==> Install dir: ${INSTALL_DIR}"
mkdir -p "${INSTALL_DIR}/data"
cd "${INSTALL_DIR}"

if [[ ! -f docker-compose.yml ]] || [[ "${REFRESH_COMPOSE:-0}" == "1" ]]; then
  echo "==> Downloading compose from GitHub..."
  curl -fsSL -o docker-compose.yml "${COMPOSE_URL}"
fi

if [[ ! -f data/cookies.json ]]; then
  echo "ERROR: data/cookies.json missing."
  echo "  Mac se copy karo: scp cookies.json user@vps:${INSTALL_DIR}/data/cookies.json"
  exit 1
fi

echo "==> Docker network: ${HERMES_NETWORK}"
docker network inspect "${HERMES_NETWORK}" >/dev/null 2>&1 || docker network create "${HERMES_NETWORK}"

export GITHUB_REPO="${GITHUB_USER}/${GITHUB_REPO}"
export GEMINI_DATA_DIR=./data
export HERMES_DOCKER_NETWORK="${HERMES_NETWORK}"

echo "==> Building & starting gemini-cookie..."
docker compose up -d --build

if docker ps --format '{{.Names}}' | grep -qx "${HERMES_WEBUI_CONTAINER}"; then
  if ! docker inspect "${HERMES_WEBUI_CONTAINER}" --format '{{range $k,$v := .NetworkSettings.Networks}}{{$k}} {{end}}' | grep -q "${HERMES_NETWORK}"; then
    echo "==> Connecting ${HERMES_WEBUI_CONTAINER} to ${HERMES_NETWORK}..."
    docker network connect "${HERMES_NETWORK}" "${HERMES_WEBUI_CONTAINER}" || true
  fi
else
  echo "NOTE: Container '${HERMES_WEBUI_CONTAINER}' not running — baad mein connect karo:"
  echo "  docker network connect ${HERMES_NETWORK} <hermes-webui-container-name>"
fi

echo ""
echo "Done. Hermes WebUI settings:"
echo "  Base URL:  http://gemini-cookie:8765/v1"
echo "  API Key:   sk-gemini-cookie-local  (ya apna GEMINI_API_KEY)"
echo "  Model:     gemini-3-flash"
echo ""
echo "Health (VPS par): curl -sS http://127.0.0.1:8765/api/health"
