#!/bin/bash

# Activate the conda environment where edapi is installed
eval "$(conda shell.bash hook)"
conda activate ED

# Set the PYTHONPATH
export PYTHONPATH=$(pwd)

# Run the API server with no sync option and port 8001
python -m app.run --no-sync --port 8001
