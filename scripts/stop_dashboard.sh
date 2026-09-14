#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$ROOT_DIR/.dashboard.pid"
PORT="${1:-8502}"

if [[ -f "$PID_FILE" ]]; then
  PID="$(cat "$PID_FILE")"
  if kill -0 "$PID" >/dev/null 2>&1; then
    kill "$PID"
    rm -f "$PID_FILE"
    echo "Stopped dashboard process $PID."
    exit 0
  fi
fi

PIDS="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN || true)"
if [[ -z "$PIDS" ]]; then
  echo "No dashboard process is listening on port $PORT."
  exit 0
fi

for PID in $PIDS; do
  kill "$PID"
  echo "Stopped process $PID on port $PORT."
done
rm -f "$PID_FILE"
