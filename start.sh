#!/usr/bin/env bash
set -e

echo "Starting LocalForge AI Coding Assistant..."
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

exec python3 launch.py --no-browser "$@"
