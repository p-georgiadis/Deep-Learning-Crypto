#!/bin/bash
# setup_dev.sh

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "Setting up development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate || {
    echo "${RED}Failed to activate virtual environment${NC}"
    exit 1
}

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/raw data/processed
mkdir -p models
mkdir -p logs
mkdir -p results
mkdir -p visualizations
mkdir -p notebooks

# Initialize pre-commit hooks
echo "Setting up pre-commit hooks..."
pip install pre-commit
pre-commit install

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cp .env.example .env
fi

# Setup Jupyter kernel
echo "Setting up Jupyter kernel..."
python -m ipykernel install --user --name crypto_pred --display-name "Crypto Prediction"

echo "${GREEN}Development environment setup complete!${NC}"
echo "To activate the environment:"
echo "  source .venv/bin/activate"
echo "To start Jupyter:"
echo "  jupyter lab"