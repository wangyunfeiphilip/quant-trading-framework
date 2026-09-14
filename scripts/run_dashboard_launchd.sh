#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/Users/wangyunfei/AIQuantRuntime"
LOG_DIR="$ROOT_DIR/logs"

mkdir -p "$LOG_DIR"
{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] starting AI Quant dashboard"
  echo "root=$ROOT_DIR"
  echo "python=$ROOT_DIR/.venv/bin/python"
} >> "$LOG_DIR/dashboard-wrapper.log"

# Do not rely on launchd's inherited working directory. The browser/Codex
# session may disappear, and macOS can report getcwd errors for it.
cd "$ROOT_DIR"
unset PYTHONHOME

export HOME="/Users/wangyunfei"
export QTF_AUTO_REFRESH_DATA="${QTF_AUTO_REFRESH_DATA:-1}"
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true

exec "$ROOT_DIR/.venv/bin/python" -m streamlit run "$ROOT_DIR/app.py" \
  --server.port 8502 \
  --server.address 127.0.0.1 \
  --server.headless true \
  --browser.gatherUsageStats false
