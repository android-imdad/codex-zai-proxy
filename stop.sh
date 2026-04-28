#!/bin/bash
set -e

PIDFILE="/tmp/codex-zai-proxy.pid"

echo "=== Stopping Codex -> Z.AI Proxy ==="

if [[ -f "$PIDFILE" ]]; then
    PROXY_PID=$(cat "$PIDFILE")
    if kill -0 "$PROXY_PID" 2>/dev/null; then
        echo "Stopping proxy (PID $PROXY_PID)..."
        kill "$PROXY_PID" || true
        sleep 1
    fi
    rm -f "$PIDFILE"
fi

echo "Proxy stopped."
