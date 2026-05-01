#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/excel-agent-backend"
PORT="${BACKEND_PORT:-8000}"
HOST="${BACKEND_HOST:-127.0.0.1}"

cd "$BACKEND_DIR"

if [[ -d ".venv" ]]; then
  VENV_DIR=".venv"
elif [[ -d "venv" ]]; then
  VENV_DIR="venv"
else
  VENV_DIR=".venv"
  echo "Creating backend virtual environment at $BACKEND_DIR/$VENV_DIR"
  if command -v uv >/dev/null 2>&1; then
    uv venv "$VENV_DIR"
  else
    python3 -m venv "$VENV_DIR"
  fi
fi

source "$VENV_DIR/bin/activate"

if ! python -c "import fastapi, uvicorn" >/dev/null 2>&1; then
  echo "Installing backend dependencies into $VENV_DIR"
  if command -v uv >/dev/null 2>&1; then
    uv pip install -r requirements.txt
  else
    python -m pip install -r requirements.txt
  fi
fi

echo "Backend venv: $BACKEND_DIR/$VENV_DIR"
echo "Backend URL:  http://$HOST:$PORT"
exec python -m uvicorn main:app --reload --host "$HOST" --port "$PORT"
