#!/bin/bash
# Launch script for CIFAR-10 Streamlit App
# Uses the cifar10app conda environment with Python 3.12

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONDA_ENV="/opt/anaconda3/envs/cifar10app/bin"

echo "🧠 Starting CIFAR-10 Image Classifier..."
echo "   Open http://localhost:8501 in your browser"
echo ""

"${CONDA_ENV}/streamlit" run "${SCRIPT_DIR}/app.py" --server.port 8501
