#!/bin/bash

# Full-Stack Sentiment Analyzer - Start Script
# This script starts both the backend API server and frontend dev server

set -e

echo "=========================================="
echo "Sentiment Analyzer Full-Stack Setup"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if pnpm is available
if ! command -v pnpm &> /dev/null && ! npx pnpm --version &> /dev/null; then
    echo -e "${YELLOW}Installing pnpm...${NC}"
    npm install -g pnpm || use_npx=true
fi

# Determine how to run pnpm
if [ "$use_npx" = true ]; then
    PNPM_CMD="npx pnpm"
else
    PNPM_CMD="pnpm"
fi

echo -e "${BLUE}Building backend API server...${NC}"
cd /Users/bipashapatra/Downloads/Microtext-Sentiment-Analyzer
$PNPM_CMD --filter @workspace/api-server run build
echo -e "${GREEN}✓ Backend built successfully${NC}"
echo ""

echo -e "${BLUE}Starting services...${NC}"
echo ""

# Start backend in background
echo -e "${BLUE}[1/2] Starting Backend API Server on http://localhost:8000${NC}"
PORT=8000 node --enable-source-maps ./artifacts/api-server/dist/index.mjs &
BACKEND_PID=$!
echo -e "${GREEN}✓ Backend PID: $BACKEND_PID${NC}"

sleep 2

# Give backend a moment to start, then check if it's ready
for i in {1..5}; do
    if curl -s http://localhost:8000/api/healthz > /dev/null 2>&1; then
        echo -e "${GREEN}✓ Backend is ready${NC}"
        break
    fi
    if [ $i -lt 5 ]; then
        echo "  Waiting for backend to start ($i/5)..."
        sleep 1
    fi
done

echo ""
echo -e "${BLUE}[2/2] Starting Frontend Dev Server on http://localhost:4173${NC}"
$PNPM_CMD --filter "@workspace/sentiment-analysis" run dev:local &
FRONTEND_PID=$!
echo -e "${GREEN}✓ Frontend PID: $FRONTEND_PID${NC}"

echo ""
echo "=========================================="
echo -e "${GREEN}✓ All services started!${NC}"
echo "=========================================="
echo ""
echo "Frontend:   http://localhost:4173"
echo "Backend:    http://localhost:8000"
echo ""
echo "Backend Endpoints:"
echo "  - Health:            /api/healthz"
echo "  - Analyze:           POST /api/sentiment/analyze"
echo "  - Batch Analyze:     POST /api/sentiment/analyze-batch"
echo "  - Train Models:      POST /api/models/train"
echo "  - Model Metrics:     GET  /api/models/metrics"
echo "  - Compare Models:    POST /api/models/compare"
echo "  - Word Cloud:        POST /api/dataset/wordcloud"
echo "  - Generate Report:   POST /api/report/generate"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all services${NC}"
echo ""

# Wait for both processes, exit if either fails
wait
