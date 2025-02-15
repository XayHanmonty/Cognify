#!/bin/bash

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Check if virtual environment exists and create if it doesn't
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if not already installed
pip install langchain langchain-openai pydantic

# Run the test script
python test_agent.py

# Deactivate virtual environment
deactivate
