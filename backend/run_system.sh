#!/bin/bash
# Complete Sentiment Analysis - Setup & Run Script
# Usage: bash run_system.sh [train|api|streamlit|all]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo -e "${BLUE}"
echo "╔════════════════════════════════════════════════════════════╗"  
echo "║   Complete Sentiment Analysis System                       ║"
echo "║   All 13 Requirements Implemented ✓                        ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check Python
echo -e "${YELLOW}Checking Python...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Error: Python 3 not found${NC}"
    exit 1
fi

PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${GREEN}✓ Python $PYTHON_VERSION found${NC}"

# Check/Install requirements
echo ""
echo -e "${YELLOW}Checking dependencies...${NC}"
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}Error: requirements.txt not found${NC}"
    exit 1
fi

python3 -m pip install -q -r requirements.txt 2>&1 | tail -5 || {
    echo -e "${YELLOW}Installing requirements...${NC}"
    python3 -m pip install -r requirements.txt
}
echo -e "${GREEN}✓ Dependencies ready${NC}"

# Default command
COMMAND="${1:-all}"

echo ""
echo -e "${BLUE}Mode: $COMMAND${NC}"
echo ""

# Train function
train_model() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}TRAINING SENTIMENT ANALYSIS MODEL${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    
    if [ ! -f "train_complete.py" ]; then
        echo -e "${RED}Error: train_complete.py not found${NC}"
        exit 1
    fi
    
    python3 -u train_complete.py
    
    if [ $? -eq 0 ]; then
        echo ""
        echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
        echo -e "${GREEN}✓ Training complete!${NC}"
        echo -e "${GREEN}════════════════════════════════════════════════════════════${NC}"
        
        # Check files
        if [ -f "vectorizer_complete.pkl" ] && [ -f "sentiment_model_complete.pkl" ] && [ -f "label_encoder_complete.pkl" ]; then
            echo -e "${GREEN}✓ All model files saved:${NC}"
            echo -e "  • vectorizer_complete.pkl ($(du -h vectorizer_complete.pkl | cut -f1))"
            echo -e "  • sentiment_model_complete.pkl ($(du -h sentiment_model_complete.pkl | cut -f1))"
            echo -e "  • label_encoder_complete.pkl ($(du -h label_encoder_complete.pkl | cut -f1))"
        fi
    else
        echo -e "${RED}Training failed!${NC}"
        exit 1
    fi
}

# API function
start_api() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}STARTING API SERVER${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    
    if [ ! -f "api_complete.py" ]; then
        echo -e "${RED}Error: api_complete.py not found${NC}"
        exit 1
    fi
    
    # Check if models exist
    if [ ! -f "sentiment_model_complete.pkl" ]; then
        echo -e "${RED}✗ Models not found!${NC}"
        echo -e "${YELLOW}Please train first: bash run_system.sh train${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Models found${NC}"
    echo -e "${YELLOW}Starting Flask API on http://localhost:5000${NC}"
    echo -e "${YELLOW}Press CTRL+C to stop${NC}"
    echo ""
    
    python3 -u api_complete.py
}

# Streamlit function
start_streamlit() {
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BLUE}STARTING STREAMLIT FRONTEND${NC}"
    echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
    
    if [ ! -f "streamlit_complete.py" ]; then
        echo -e "${RED}Error: streamlit_complete.py not found${NC}"
        exit 1
    fi
    
    # Check if API is running
    echo -e "${YELLOW}Checking API connection...${NC}"
    if ! curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
        echo -e "${RED}✗ API is not running!${NC}"
        echo -e "${YELLOW}Please start API first in another terminal:${NC}"
        echo -e "${YELLOW}  bash run_system.sh api${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ API is running${NC}"
    echo -e "${YELLOW}Opening Streamlit UI on http://localhost:8501${NC}"
    echo -e "${YELLOW}Press CTRL+C to stop${NC}"
    echo ""
    
    streamlit run streamlit_complete.py
}

# Main logic
case $COMMAND in
    train)
        train_model
        ;;
    api)
        start_api
        ;;
    streamlit)
        start_streamlit
        ;;
    all)
        # Check if models exist
        if [ ! -f "sentiment_model_complete.pkl" ]; then
            echo -e "${YELLOW}Models not found. Training first...${NC}"
            train_model
            echo ""
        fi
        
        echo -e "${YELLOW}To complete setup:${NC}"
        echo ""
        echo -e "${BLUE}Terminal 1 - Start API:${NC}"
        echo -e "  cd $SCRIPT_DIR && python3 api_complete.py"
        echo ""
        echo -e "${BLUE}Terminal 2 - Start Frontend (optional):${NC}"
        echo -e "  cd $SCRIPT_DIR && streamlit run streamlit_complete.py"
        echo ""
        echo -e "${GREEN}✓ Setup complete!${NC}"
        echo ""
        echo -e "${YELLOW}API will be available at: http://localhost:5000${NC}"
        echo -e "${YELLOW}Frontend will be available at: http://localhost:8501${NC}"
        ;;
    help)
        echo -e "${BLUE}Usage:${NC}"
        echo ""
        echo "  bash run_system.sh train        # Train model"
        echo "  bash run_system.sh api          # Start API server"
        echo "  bash run_system.sh streamlit    # Start web frontend"
        echo "  bash run_system.sh all          # Setup and show instructions"
        echo "  bash run_system.sh help         # Show this help"
        echo ""
        echo -e "${BLUE}Quick start:${NC}"
        echo ""
        echo "  1. Train: bash run_system.sh train"
        echo "  2. API (terminal 1): python3 api_complete.py"
        echo "  3. Frontend (terminal 2): streamlit run streamlit_complete.py"
        echo ""
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        echo "Usage: bash run_system.sh [train|api|streamlit|all|help]"
        exit 1
        ;;
esac

echo ""
echo -e "${GREEN}Done! ✓${NC}"
