#!/bin/bash

# Weekly Gold Price Prediction Runner
# This script runs the gold price predictor

cd "$(dirname "$0")"

echo "Running Weekly Gold Price Prediction..."
echo "Date: $(date)"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the predictor
python3 gold_predictor.py

echo "Prediction completed at $(date)"
