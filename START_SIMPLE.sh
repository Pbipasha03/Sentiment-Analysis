#!/bin/bash

# Local full-stack launcher for the Sentiment Analyzer.
# Usage:
#   ./START_SIMPLE.sh
#   ./START_SIMPLE.sh --detach

set -eo pipefail

PROJECT_DIR="/Users/bipashapatra/Downloads/Microtext-Sentiment-Analyzer"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-4173}"
BACKEND_LOG="${BACKEND_LOG:-/tmp/sentiment-analyzer-backend.log}"
FRONTEND_LOG="${FRONTEND_LOG:-/tmp/sentiment-analyzer-frontend.log}"
PID_FILE="${PID_FILE:-/tmp/sentiment-analyzer.pids}"
DETACH=false

if [[ "${1:-}" == "--detach" ]]; then
  DETACH=true
fi

cd "$PROJECT_DIR"

if command -v pnpm >/dev/null 2>&1; then
  PNPM_CMD=(pnpm)
else
  PNPM_CMD=(npx --yes pnpm)
fi

stop_port() {
  local port="$1"
  local pids

  pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"

  if [[ -n "$pids" ]]; then
    echo "Stopping existing service on port $port..."
    kill $pids 2>/dev/null || true
    sleep 1

    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null || true)"
    if [[ -n "$pids" ]]; then
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local tries="${3:-30}"

  for ((i = 1; i <= tries; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      echo "$name is ready: $url"
      return 0
    fi
    sleep 1
  done

  echo "$name did not start at $url"
  return 1
}

cleanup() {
  if [[ "$DETACH" == "false" ]]; then
    echo ""
    echo "Stopping services..."
    if [[ -n "${BACKEND_PID:-}" ]]; then
      kill "$BACKEND_PID" 2>/dev/null || true
    fi
    if [[ -n "${FRONTEND_PID:-}" ]]; then
      kill "$FRONTEND_PID" 2>/dev/null || true
    fi
  fi
}

start_backend() {
  if [[ "$DETACH" == "true" ]]; then
    nohup env PORT="$BACKEND_PORT" node --enable-source-maps ./artifacts/api-server/dist/index.mjs >"$BACKEND_LOG" 2>&1 </dev/null &
  else
    PORT="$BACKEND_PORT" node --enable-source-maps ./artifacts/api-server/dist/index.mjs >"$BACKEND_LOG" 2>&1 &
  fi

  BACKEND_PID=$!
}

start_frontend() {
  if [[ "$DETACH" == "true" ]]; then
    nohup env PORT="$FRONTEND_PORT" BASE_PATH=/ "${PNPM_CMD[@]}" --filter @workspace/sentiment-analysis run dev:local >"$FRONTEND_LOG" 2>&1 </dev/null &
  else
    PORT="$FRONTEND_PORT" BASE_PATH=/ "${PNPM_CMD[@]}" --filter @workspace/sentiment-analysis run dev:local >"$FRONTEND_LOG" 2>&1 &
  fi

  FRONTEND_PID=$!
}

if [[ "$DETACH" == "false" ]]; then
  trap cleanup EXIT INT TERM
fi

echo ""
echo "Sentiment Analyzer local startup"
echo "Project:  $PROJECT_DIR"
echo "Frontend: http://localhost:$FRONTEND_PORT"
echo "Backend:  http://localhost:$BACKEND_PORT"
echo ""

echo "Cleaning old local servers..."
stop_port "$BACKEND_PORT"
stop_port "$FRONTEND_PORT"

echo "Building backend..."
"${PNPM_CMD[@]}" --filter @workspace/api-server run build

echo "Starting backend..."
start_backend

if ! wait_for_url "Backend" "http://localhost:$BACKEND_PORT/api/healthz" 30; then
  echo ""
  echo "Backend log:"
  tail -40 "$BACKEND_LOG" || true
  exit 1
fi

echo "Starting frontend..."
start_frontend

if ! wait_for_url "Frontend" "http://localhost:$FRONTEND_PORT" 30; then
  echo ""
  echo "Frontend log:"
  tail -60 "$FRONTEND_LOG" || true
  exit 1
fi

printf "BACKEND_PID=%s\nFRONTEND_PID=%s\n" "$BACKEND_PID" "$FRONTEND_PID" >"$PID_FILE"

echo ""
echo "All services are running."
echo "Open: http://localhost:$FRONTEND_PORT"
echo ""
echo "Logs:"
echo "  Backend:  $BACKEND_LOG"
echo "  Frontend: $FRONTEND_LOG"
echo ""

if [[ "$DETACH" == "true" ]]; then
  disown "$BACKEND_PID" "$FRONTEND_PID" 2>/dev/null || true
  echo "Detached mode enabled. PID file: $PID_FILE"
  echo "To stop later:"
  echo "  kill $BACKEND_PID $FRONTEND_PID"
  exit 0
fi

echo "Press Ctrl+C to stop both services."
wait "$BACKEND_PID" "$FRONTEND_PID"
