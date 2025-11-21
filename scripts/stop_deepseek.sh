#!/bin/bash
#
# stop_deepseek.sh - Stop DeepSeek-R1 vLLM server
# ===============================================

set -e

VLLM_PORT="${VLLM_PORT:-8000}"

echo "Stopping vLLM server on port $VLLM_PORT..."

# Find and kill processes using the port
if lsof -i :$VLLM_PORT &> /dev/null; then
    PIDS=$(lsof -ti :$VLLM_PORT)
    echo "Found processes: $PIDS"
    echo $PIDS | xargs kill -TERM || true
    sleep 2

    # Force kill if still running
    if lsof -i :$VLLM_PORT &> /dev/null; then
        echo "Forcing kill..."
        lsof -ti :$VLLM_PORT | xargs kill -9 || true
    fi

    echo "✓ vLLM server stopped"
else
    echo "No vLLM server found on port $VLLM_PORT"
fi
