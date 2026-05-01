#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_SCRIPT="$ROOT_DIR/scripts/start-backend-linux.sh"
FRONTEND_SCRIPT="$ROOT_DIR/scripts/start-frontend-linux.sh"

open_terminal() {
  local title="$1"
  local command="$2"

  if command -v gnome-terminal >/dev/null 2>&1; then
    gnome-terminal --title="$title" -- bash -lc "$command; echo; echo 'Press Enter to close this terminal.'; read -r"
  elif command -v x-terminal-emulator >/dev/null 2>&1; then
    x-terminal-emulator -T "$title" -e bash -lc "$command; echo; echo 'Press Enter to close this terminal.'; read -r"
  elif command -v mate-terminal >/dev/null 2>&1; then
    mate-terminal --title="$title" -- bash -lc "$command; echo; echo 'Press Enter to close this terminal.'; read -r"
  elif command -v xfce4-terminal >/dev/null 2>&1; then
    xfce4-terminal --title="$title" --command "bash -lc \"$command; echo; echo 'Press Enter to close this terminal.'; read -r\""
  elif command -v konsole >/dev/null 2>&1; then
    konsole --new-window --title "$title" -e bash -lc "$command; echo; echo 'Press Enter to close this terminal.'; read -r"
  elif command -v xterm >/dev/null 2>&1; then
    xterm -T "$title" -e bash -lc "$command; echo; echo 'Press Enter to close this terminal.'; read -r"
  else
    echo "No supported terminal emulator found." >&2
    echo "Run these manually in two terminals:" >&2
    echo "  $BACKEND_SCRIPT" >&2
    echo "  $FRONTEND_SCRIPT" >&2
    exit 1
  fi
}

chmod +x "$BACKEND_SCRIPT" "$FRONTEND_SCRIPT"

printf -v BACKEND_COMMAND '%q' "$BACKEND_SCRIPT"
printf -v FRONTEND_COMMAND '%q' "$FRONTEND_SCRIPT"

open_terminal "Data Pilot Backend" "$BACKEND_COMMAND"
open_terminal "Data Pilot Frontend" "$FRONTEND_COMMAND"

echo "Opened Data Pilot in two terminal windows:"
echo "  Backend:  http://127.0.0.1:${BACKEND_PORT:-8000}"
echo "  Frontend: http://127.0.0.1:${FRONTEND_PORT:-3000}"
echo
echo "If you are using VS Code and want integrated terminal panes, run task: DataPilot: Start All"
