#!/bin/bash

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate

# Install required Python packages
pip install lxml undetected-chromedriver requests tqdm

# Run the Python acts.py script
python acts.py

# Deactivate the virtual environment
deactivate
