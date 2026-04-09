#!/bin/bash

# Production Sentiment Analysis System - Setup & Run Script
# Handles dependencies, training, and launching the app

set -e  # Exit on error

BACKEND_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BACKEND_DIR"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🎯 Sentiment Analysis System${NC}"
echo -e "${BLUE}========================================${NC}"

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# ============================================================================
# 1. CHECK PYTHON
# ============================================================================

echo -e "${YELLOW}[1/4] Checking Python...${NC}"

if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    echo -e "${RED}❌ Python not found. Please install Python 3.8+${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# ============================================================================
# 2. INSTALL DEPENDENCIES
# ============================================================================

echo -e "${YELLOW}[2/4] Installing dependencies...${NC}"

if [ ! -f "requirements_production.txt" ]; then
    echo -e "${RED}❌ requirements_production.txt not found${NC}"
    exit 1
fi

if $PYTHON_CMD -m pip install -q -r requirements_production.txt; then
    echo -e "${GREEN}✓ Dependencies installed${NC}"
else
    echo -e "${RED}❌ Failed to install dependencies${NC}"
    exit 1
fi

# ============================================================================
# 3. TRAIN MODELS
# ============================================================================

echo -e "${YELLOW}[3/4] Training models...${NC}"

if [ -f "train_production.py" ]; then
    if $PYTHON_CMD train_production.py; then
        echo -e "${GREEN}✓ Models trained successfully${NC}"
    else
        echo -e "${RED}❌ Training failed${NC}"
        exit 1
    fi
else
    echo -e "${RED}❌ train_production.py not found${NC}"
    exit 1
fi

# ============================================================================
# 4. LAUNCH STREAMLIT APP
# ============================================================================

echo -e "${YELLOW}[4/4] Launching Streamlit app...${NC}"

if [ -f "app_production.py" ]; then
    echo -e "${GREEN}✓ Starting Streamlit...${NC}"
    echo ""
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}🚀 App launching...${NC}"
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}Visit: http://localhost:8501${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
    
    streamlit run app_production.py
else
    echo -e "${RED}❌ app_production.py not found${NC}"
    exit 1
fi
