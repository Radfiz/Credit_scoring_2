#!/usr/bin/env bash
set -eu

echo "=== Stage 1: Model Preparation ==="

echo "1. Training PyTorch model"
python scripts/train_nn_model.py

echo "2. Converting to ONNX"
python scripts/convert_to_onnx.py

echo "3. Quantizing model"
python scripts/quantize_model.py

echo ""
echo "Artifacts:"
ls -1 models/onnx/* reports/*.{csv,json} 2>/dev/null | sort