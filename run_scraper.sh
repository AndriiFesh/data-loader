#!/bin/bash

# Create and activate a Python virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt

# Install Chromium (Chromium is an open-source version of Chrome)
sudo apt-get update
sudo apt-get install -y chromium-browser

# Run the Python acts.py script
python3 acts.py

python3 indexing_v2.py

git checkout -b data-loader

dvc pull

cp ft_embeddings/index.faiss faiss_index_acts.dvc

git add ft_embeddings/index.faiss ft_embeddings/index.pkl

git commit -m "add new index files index.faiss и index.pkl" -n

git push & wait

deactivate
