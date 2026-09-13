#!/usr/bin/env bash
set -euo pipefail

LABEL="com.wyf.aiquant.dashboard"
DOMAIN="gui/$(id -u)"

if launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1; then
  launchctl bootout "$DOMAIN/$LABEL"
  echo "Uninstalled launchd service: $LABEL"
else
  echo "Service is not installed: $LABEL"
fi
