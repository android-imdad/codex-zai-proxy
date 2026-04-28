#!/bin/bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

PIDFILE="/tmp/codex-zai-proxy.pid"
LOGFILE="$DIR/http-proxy.log"

echo "=== Codex -> Z.AI Proxy Launcher ==="

if [[ -z "${OPENCODE_API_KEY:-${ZAI_API_KEY:-}}" ]]; then
    echo "ERROR: Set OPENCODE_API_KEY or ZAI_API_KEY before starting." >&2
    exit 1
fi

if [[ -f "$PIDFILE" ]]; then
    OLD_PID=$(cat "$PIDFILE")
    if kill -0 "$OLD_PID" > /dev/null 2>&1; then
        echo "Proxy already running (PID $OLD_PID). Stopping first..."
        kill "$OLD_PID" || true
        sleep 1
    fi
fi

echo "Starting proxy on 127.0.0.1:8080..."
node http-proxy.js >> "$LOGFILE" 2>&1 &
PROXY_PID=$!
echo $PROXY_PID > "$PIDFILE"

sleep 2

if kill -0 "$PROXY_PID" 2>/dev/null; then
    echo "Proxy running (PID $PROXY_PID)"
else
    echo "ERROR: Proxy failed to start. Check $LOGFILE:"
    tail -n 40 "$LOGFILE"
    exit 1
fi

echo "Configure Codex to use base_url=http://127.0.0.1:8080 with wire_api=responses."
