#!/bin/bash

pip install lxml undetected-chromedriver requests tqdm aiohttp
# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required Python packages
pip install -r requirements.txt

# Run the Python acts.py script
python3 acts.py

python3 indexing_v2.py

git checkout -b feature/add-new-embeddings

dvc pull

cp ft_embeddings/index.faiss  #TODO add path to repo
cp ft_embeddings/index.pkl  #TODO add path to repo

git add index.faiss index.pkl

git commit -m "add new index files index.faiss и index.pkl"

git push

deactivate

