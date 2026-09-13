#!/usr/bin/env bash
set -euo pipefail

SOURCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNTIME_ROOT="/Users/wangyunfei/AIQuantRuntime"

mkdir -p "$RUNTIME_ROOT"
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '__pycache__/' \
  --exclude 'logs/' \
  --exclude '.dashboard.pid' \
  --exclude '.venv-financial-research-agent/' \
  --exclude '.venv-finrl/' \
  --exclude '.venv-tradingagents/' \
  "$SOURCE_ROOT/" "$RUNTIME_ROOT/"

mkdir -p "$RUNTIME_ROOT/logs"
echo "Runtime synchronized to $RUNTIME_ROOT"
