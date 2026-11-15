#!/usr/bin/env bash
# Start the local OpenShorts stack. Uses the GPU image when Docker has the
# nvidia runtime, the CPU image otherwise. Extra args go to `docker compose up`
# (e.g. local/up.sh -d --build).
#
# Ollama must be running first: local/start-ollama.sh
set -euo pipefail
cd "$(dirname "$0")/.."
files=(-f docker-compose.yml -f docker-compose.local.yml)
if docker info 2>/dev/null | grep -q 'Runtimes:.*nvidia'; then
  files+=(-f docker-compose.local-gpu.yml)
  echo "GPU runtime found: backend uses the GPU image."
else
  echo "No nvidia runtime in Docker: backend runs on CPU."
fi
exec docker compose "${files[@]}" up "$@"
