#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${1:-8502}"
HOST="127.0.0.1"
LOG_DIR="$ROOT_DIR/logs"
PID_FILE="$ROOT_DIR/.dashboard.pid"

mkdir -p "$LOG_DIR"

if lsof -tiTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Dashboard is already running at http://$HOST:$PORT/"
  exit 0
fi

cd "$ROOT_DIR"
unset PYTHONHOME
QTF_AUTO_REFRESH_DATA="${QTF_AUTO_REFRESH_DATA:-1}" \
HOME="/Users/wangyunfei" \
STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
STREAMLIT_SERVER_HEADLESS=true \
nohup .venv/bin/python -m streamlit run "$ROOT_DIR/app.py" \
  --server.port "$PORT" \
  --server.address "$HOST" \
  --server.headless true \
  --browser.gatherUsageStats false \
  > "$LOG_DIR/dashboard-$PORT.log" 2>&1 &

echo "$!" > "$PID_FILE"
echo "Dashboard started at http://$HOST:$PORT/"
echo "Log: $LOG_DIR/dashboard-$PORT.log"
