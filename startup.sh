#!/bin/bash
# EcoSentinel — Full Stack Startup
# DGX Spark (GB10 Blackwell, ARM64, sm_121, CUDA 13.0)
# Follows SKILL.md conventions for this machine.

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="/home/nvidia/models"
DATA_DIR="/home/nvidia/data"
LLAMA_DIR="/home/nvidia/llama.cpp"

echo "======================================"
echo "  EcoSentinel — DGX Spark Startup"
echo "======================================"

# ── Pre-flight (from SKILL.md) ────────────────────────────────────────────
echo ""
echo "[1/6] Pre-flight checks..."
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
docker run --rm --gpus all ubuntu:22.04 nvidia-smi -L 2>/dev/null | head -1

# ── Step 1: Nemotron via llama-server ─────────────────────────────────────
echo ""
echo "[2/6] Starting Nemotron (llama-server on port 8080)..."
if curl -s http://localhost:8080/health > /dev/null 2>&1; then
  echo "  ✓ llama-server already running"
else
  NEMOTRON_GGUF="$MODELS_DIR/gguf/ggml-org--Nemotron-Nano-3-30B-A3B-GGUF/Nemotron-Nano-3-30B-A3B-Q4_K_M.gguf"
  cd "$LLAMA_DIR"
  LD_LIBRARY_PATH="$LLAMA_DIR/build/lib" \
  nohup "$LLAMA_DIR/build/bin/llama-server" \
    --model "$NEMOTRON_GGUF" \
    --host 0.0.0.0 \
    --port 8080 \
    --n-gpu-layers 99 \
    --ctx-size 40000 \
    --parallel 2 \
    --alias nemotron \
    > /tmp/llama-server.log 2>&1 &
  echo "  Waiting for Nemotron to load..."
  for i in $(seq 1 30); do
    sleep 2
    if curl -s http://localhost:8080/health > /dev/null 2>&1; then
      echo "  ✓ Nemotron ready (took ~${i}0s)"
      break
    fi
  done
fi

# ── Step 2: RAPIDS container (GPU data processing) ────────────────────────
# As per SKILL.md: RAPIDS 25.10 Notebooks from NVIDIA NGC
# Note: using rapidsai/notebooks:26.04a-cuda13-py3.11 (latest available tag for CUDA 13)
echo ""
echo "[3/6] Starting RAPIDS (JupyterLab on port 8888)..."
if docker ps --format "{{.Names}}" | grep -q ecosentinel-rapids; then
  echo "  ✓ RAPIDS container already running"
else
  docker run -d \
    --name ecosentinel-rapids \
    --restart unless-stopped \
    --gpus all \
    --shm-size 8g \
    -p 8888:8888 \
    -v "$DATA_DIR":/data:ro \
    -v "$SCRIPT_DIR":/workspace/ecosentinel \
    rapidsai/notebooks:26.04a-cuda13-py3.11 \
    jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' \
    2>&1 | tail -2
  echo "  ✓ RAPIDS JupyterLab starting on :8888"
fi

# ── Step 3: Anthropic→Nemotron Proxy ──────────────────────────────────────
# Routes OpenClaw embedded agent calls to Nemotron via fake Anthropic API
echo ""
echo "[4/6] Starting Anthropic→Nemotron proxy (port 8090)..."
if curl -s http://localhost:8090/health > /dev/null 2>&1; then
  echo "  ✓ Proxy already running"
else
  cd "$SCRIPT_DIR"
  source .venv/bin/activate
  nohup python3 app/anthropic_proxy.py > /tmp/proxy.log 2>&1 &
  sleep 2
  curl -s http://localhost:8090/health && echo "  ✓ Proxy ready"
fi

# ── Step 4: OpenClaw gateway ──────────────────────────────────────────────
echo ""
echo "[5/6] Starting OpenClaw gateway (Telegram bot: @EcoSentinel_dgx_bot)..."
export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && source "$NVM_DIR/nvm.sh"

if pgrep -f "openclaw-gateway" > /dev/null 2>&1; then
  echo "  ✓ OpenClaw gateway already running"
else
  # ANTHROPIC_API_KEY routes OpenClaw's embedded agent through our local proxy
  # openclaw.json configures baseUrl=http://localhost:8090 for anthropic provider
  ANTHROPIC_API_KEY=nemotron-local-dgx-spark \
  nohup openclaw gateway --port 18789 --allow-unconfigured > /tmp/openclaw.log 2>&1 &
  sleep 5
  echo "  ✓ OpenClaw gateway started (Telegram: @EcoSentinel_dgx_bot)"
fi

# ── Step 5: EcoSentinel app ───────────────────────────────────────────────
echo ""
echo "[6/6] Starting EcoSentinel app (Gradio :7860 + REST API :7861)..."
if curl -s http://localhost:7861/api/health > /dev/null 2>&1; then
  echo "  ✓ EcoSentinel already running"
else
  cd "$SCRIPT_DIR"
  source .venv/bin/activate
  DATA_DIR="$DATA_DIR" nohup python3 app/main.py > /tmp/ecosentinel.log 2>&1 &
  echo "  Waiting for app..."
  for i in $(seq 1 15); do
    sleep 2
    if curl -s http://localhost:7861/api/health > /dev/null 2>&1; then
      echo "  ✓ EcoSentinel ready"
      break
    fi
  done
fi

# ── Summary ────────────────────────────────────────────────────────────────
echo ""
echo "======================================"
echo "  EcoSentinel is LIVE"
echo "======================================"
echo "  Telegram Bot:  @EcoSentinel_dgx_bot (all queries → Nemotron-30B)"
echo "  Web UI:        http://$(hostname).local:7860"
echo "  REST API:      http://$(hostname).local:7861/api/health"
echo "  JupyterLab:    http://$(hostname).local:8888"
echo "  Nemotron:      http://localhost:8080/health"
echo "  Proxy:         http://localhost:8090/health"
echo "  OpenClaw:      ws://localhost:18789"
echo ""
echo "  Sample Telegram queries:"
echo "  'Which restaurants in San Jose are highest risk?'"
echo "  'Show me food inspection violations in Sunnyvale'"
echo "  'What is PM2.5 and why does it matter?'"
echo "======================================"
