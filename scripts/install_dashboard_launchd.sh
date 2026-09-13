#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="com.wyf.aiquant.dashboard"
SOURCE_PLIST="$ROOT_DIR/scripts/$LABEL.plist"
USER_HOME="/Users/wangyunfei"
AGENT_DIR="$USER_HOME/Library/LaunchAgents"
PLIST="$AGENT_DIR/$LABEL.plist"
DOMAIN="gui/$(id -u)"

mkdir -p "$ROOT_DIR/logs"
mkdir -p "$AGENT_DIR"

"$ROOT_DIR/scripts/sync_runtime.sh"

cp "$SOURCE_PLIST" "$PLIST"

if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "$DOMAIN/$LABEL" >/dev/null 2>&1 || true
fi

launchctl bootstrap "$DOMAIN" "$PLIST"
launchctl enable "$DOMAIN/$LABEL"
launchctl kickstart -k "$DOMAIN/$LABEL"

HEALTH_URL="http://127.0.0.1:8502/_stcore/health"
for attempt in {1..30}; do
  if curl --silent --show-error --fail --max-time 2 "$HEALTH_URL" >/dev/null 2>&1; then
    echo "Installed and verified launchd service: $LABEL"
    echo "Dashboard URL: http://127.0.0.1:8502/"
    echo "LaunchAgent: $PLIST"
    echo "Logs: $ROOT_DIR/logs/dashboard-launchd.log"
    exit 0
  fi
  sleep 1
done

echo "ERROR: launchd service was registered but the dashboard did not become healthy." >&2
echo "Check: $ROOT_DIR/logs/dashboard-launchd.err.log" >&2
exit 1
