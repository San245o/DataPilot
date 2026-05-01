#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FRONTEND_DIR="$ROOT_DIR/excel-agent-dashboard"
PORT="${FRONTEND_PORT:-3000}"

cd "$FRONTEND_DIR"

if [[ ! -d "node_modules" ]]; then
  echo "Installing frontend dependencies"
  if command -v pnpm >/dev/null 2>&1; then
    pnpm install
  elif [[ -f package-lock.json ]]; then
    npm ci
  else
    npm install
  fi
fi

echo "Frontend directory: $FRONTEND_DIR"
echo "Frontend URL:       http://127.0.0.1:$PORT"

if command -v pnpm >/dev/null 2>&1; then
  exec pnpm dev -- -p "$PORT"
else
  exec npm run dev -- -p "$PORT"
fi
