#!/bin/bash
#
# start_deepseek.sh - Launch DeepSeek-R1 with vLLM for MAGNET
# ===========================================================
#
# Launches DeepSeek-R1-Distill-Qwen-32B with 4-bit quantization
# optimized for 2x A40 GPUs (48GB each).
#
# Usage:
#   ./scripts/start_deepseek.sh [model_path]
#
# Prerequisites:
#   - vLLM installed (pip install vllm)
#   - Model downloaded to MODEL_PATH
#   - CUDA available (nvidia-smi should show GPUs)

set -e  # Exit on error

# Configuration
MODEL_NAME="${1:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}"
MODEL_PATH="${MODEL_PATH:-/workspace/models/$MODEL_NAME}"
VLLM_PORT="${VLLM_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
QUANTIZATION="${QUANTIZATION:-awq}"  # 4-bit quantization (awq or gptq)

# GPU configuration for 2x A40
TENSOR_PARALLEL_SIZE=2  # Use both GPUs
CUDA_DEVICES="0,1"      # GPU IDs

# Log file
LOG_DIR="/workspace/magnet/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/deepseek_vllm_$(date +%Y%m%d_%H%M%S).log"

echo "========================================"
echo "Starting DeepSeek-R1 with vLLM"
echo "========================================"
echo "Model: $MODEL_NAME"
echo "Model Path: $MODEL_PATH"
echo "Port: $VLLM_PORT"
echo "GPUs: $CUDA_DEVICES (Tensor Parallel Size: $TENSOR_PARALLEL_SIZE)"
echo "Quantization: $QUANTIZATION"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Log File: $LOG_FILE"
echo "========================================"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "ERROR: Model not found at $MODEL_PATH"
    echo ""
    echo "To download the model, run:"
    echo "  huggingface-cli download $MODEL_NAME --local-dir $MODEL_PATH"
    echo ""
    echo "Or set MODEL_PATH environment variable to the correct location:"
    echo "  export MODEL_PATH=/path/to/your/model"
    echo "  ./scripts/start_deepseek.sh"
    exit 1
fi

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. CUDA may not be available."
    exit 1
fi

echo ""
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Check if vLLM is installed
if ! python3 -c "import vllm" &> /dev/null; then
    echo "ERROR: vLLM not installed."
    echo "Install with: pip install vllm"
    exit 1
fi

# Kill existing vLLM processes on this port
if lsof -i :$VLLM_PORT &> /dev/null; then
    echo "WARNING: Port $VLLM_PORT is already in use. Killing existing process..."
    lsof -ti :$VLLM_PORT | xargs kill -9 || true
    sleep 2
fi

# Launch vLLM server
echo ""
echo "Launching vLLM server..."
echo "This may take 2-5 minutes to load the model..."
echo ""

CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_PATH" \
    --port $VLLM_PORT \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN \
    --quantization $QUANTIZATION \
    --trust-remote-code \
    --disable-log-requests \
    2>&1 | tee "$LOG_FILE" &

VLLM_PID=$!

echo ""
echo "vLLM server started with PID: $VLLM_PID"
echo "Waiting for server to be ready..."

# Wait for server to be ready
MAX_WAIT=300  # 5 minutes
WAIT_TIME=0
while [ $WAIT_TIME -lt $MAX_WAIT ]; do
    if curl -s http://localhost:$VLLM_PORT/health &> /dev/null; then
        echo ""
        echo "✓ vLLM server is ready!"
        echo ""
        echo "Endpoint: http://localhost:$VLLM_PORT/v1/completions"
        echo "Health check: http://localhost:$VLLM_PORT/health"
        echo "PID: $VLLM_PID"
        echo ""
        echo "To stop the server:"
        echo "  kill $VLLM_PID"
        echo "  or"
        echo "  ./scripts/stop_deepseek.sh"
        echo ""
        echo "Test the server:"
        echo "  curl http://localhost:$VLLM_PORT/v1/completions \\"
        echo "    -H 'Content-Type: application/json' \\"
        echo "    -d '{"
        echo "      \"model\": \"$MODEL_NAME\","
        echo "      \"prompt\": \"Hello, world!\","
        echo "      \"max_tokens\": 50"
        echo "    }'"
        echo ""
        exit 0
    fi

    # Check if process is still running
    if ! ps -p $VLLM_PID > /dev/null; then
        echo ""
        echo "ERROR: vLLM process died during startup"
        echo "Check log file: $LOG_FILE"
        exit 1
    fi

    sleep 5
    WAIT_TIME=$((WAIT_TIME + 5))
    echo -n "."
done

echo ""
echo "ERROR: Server failed to start within $MAX_WAIT seconds"
echo "Check log file: $LOG_FILE"
echo "Process PID: $VLLM_PID (you may need to kill it manually)"
exit 1
