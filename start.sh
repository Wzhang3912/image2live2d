#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
case "${1:-}" in
  ""|--gpu) ;;
  *) echo 'Usage: ./start.sh [--gpu]'; exit 2 ;;
esac
if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  echo 'Install Docker with Compose: https://docs.docker.com/get-started/get-docker/'
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo 'Start Docker and run this command again.'; exit 1
fi
docker compose up --build --wait --wait-timeout 180 app
url="http://localhost:${PORT:-8000}"
echo "Open $url — the app will guide model setup."
if [[ "${IMAGE2LIVE2D_NO_BROWSER:-0}" != 1 ]]; then
  if command -v open >/dev/null; then open "$url" || true
  elif command -v xdg-open >/dev/null; then xdg-open "$url" >/dev/null 2>&1 || true
  fi
fi
if [[ "${1:-}" == --gpu ]] || command -v nvidia-smi >/dev/null 2>&1; then
  echo 'Preparing the local NVIDIA inference container. Models download only when requested in the app.'
  docker compose --profile gpu up --build -d seethrough || {
    echo 'GPU startup failed. The web app remains available. Check the NVIDIA driver and Container Toolkit.'
    exit 1
  }
fi
