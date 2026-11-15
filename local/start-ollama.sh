#!/usr/bin/env bash
# Start Ollama on the GPU for OpenShorts' local moment picker.
#
# It listens on the Docker bridge address only (not 0.0.0.0), so the backend
# container can reach it through host.docker.internal while it stays off your
# LAN. The 16k context matters: Ollama's 4096 default silently truncates the
# transcript windows the picker sends.
#
# Use the same address for the CLI:  OLLAMA_HOST=172.17.0.1:11434 ollama list
set -euo pipefail
BRIDGE_IP="$(ip -4 -o addr show docker0 2>/dev/null | awk '{print $4}' | cut -d/ -f1)"
BRIDGE_IP="${BRIDGE_IP:-172.17.0.1}"
export OLLAMA_HOST="${OLLAMA_HOST:-$BRIDGE_IP:11434}"
export OLLAMA_CONTEXT_LENGTH="${OLLAMA_CONTEXT_LENGTH:-16384}"
# Unload the model after 5 min idle so Whisper/rendering get the VRAM back.
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-5m}"
echo "Ollama on $OLLAMA_HOST (context $OLLAMA_CONTEXT_LENGTH)"
exec ollama serve
