#!/bin/bash

echo "Starting YouTube Shorts Automation..."

# Activate the Python virtual environment
source ./.venv/bin/activate

# Set PYTHONPATH to the project directory for local imports
export PYTHONPATH="$(pwd)"

# Change to the project directory (already here, but for clarity)
cd "$(pwd)"

# Checkout the master branch (only use stable version of the code)
git add .
git commit -m "Automated commit before running script"
git checkout master

# Run the main Python script
python main.py

echo "Script execution completed."