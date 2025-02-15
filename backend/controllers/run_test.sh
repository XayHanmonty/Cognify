#!/bin/bash

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Check if MongoDB is running
if ! pgrep -x "mongod" > /dev/null; then
    echo "MongoDB is not running. Starting MongoDB..."
    brew services start mongodb-community || {
        echo "Failed to start MongoDB. Please install it with: brew install mongodb-community"
        exit 1
    }
    sleep 2  # Wait for MongoDB to start
fi

# Check if virtual environment exists and create if it doesn't
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install requirements if not already installed
pip install langchain langchain-openai pydantic pymongo

# Run the test script
python test_agent.py

# Deactivate virtual environment
deactivate
