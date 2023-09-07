#!/bin/bash

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install lxml undetected-chromedriver requests tqdm aiohttp

# Run the Python acts.py script
python3 acts.py

# Deactivate the virtual environment
deactivate
