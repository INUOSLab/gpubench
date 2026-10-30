#!/bin/bash
sudo apt-get install -y python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126