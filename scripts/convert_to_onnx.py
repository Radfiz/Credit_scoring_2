#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import onnx
import onnxruntime as ort
import numpy as np
import json
from pathlib import Path
from models.neural_network import CreditScoringNN  # та же архитектура

def convert_to_onnx():
    print("Converting PyTorch -> ONNX ...")
    with open("models/trained/nn_model_metadata.json") as f:
        meta = json.load(f)
    input_size = meta["input_size"]

    model = CreditScoringNN(input_size=input_size)
    model.load_state_dict(torch.load("models/trained/credit_scoring_nn.pth"))
    model.eval()

    dummy = torch.randn(1, input_size)
    onnx_path = Path("models/onnx/credit_scoring_model.onnx")
    onnx_path.parent.mkdir(exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=18,
        input_names=["input_features"],
        output_names=["default_probability"],
        dynamic_axes={
            "input_features": {0: "batch_size"},
            "default_probability": {0: "batch_size"}
        }
    )

    # validate
    sess = ort.InferenceSession(str(onnx_path))
    inp = {sess.get_inputs()[0].name: dummy.numpy().astype(np.float32)}
    out_onnx = sess.run(None, inp)[0]
    with torch.no_grad():
        out_torch = model(dummy).numpy()
    delta = np.abs(out_torch - out_onnx).max()
    print(f"Max diff torch-onnx: {delta:.8f}")
    assert delta < 1e-5, "Conversion failed validation"

    meta_onnx = {
        "model_path": str(onnx_path),
        "input_shape": [1, input_size],
        "output_shape": [1, 1],
        "opset_version": 18,
        "conversion_date": meta["training_date"],
        "validation_passed": True,
        "pytorch_metrics": meta["metrics"]
    }
    with open("models/onnx/onnx_metadata.json", "w") as f:
        json.dump(meta_onnx, f, indent=2)
    print("ONNX model and metadata saved.")
    return onnx_path

def benchmark():
    print("Benchmark PyTorch vs ONNX ...")
    with open("models/trained/nn_model_metadata.json") as f:
        input_size = json.load(f)["input_size"]

    model_torch = CreditScoringNN(input_size=input_size)
    model_torch.load_state_dict(torch.load("models/trained/credit_scoring_nn.pth"))
    model_torch.eval()

    sess = ort.InferenceSession("models/onnx/credit_scoring_model.onnx")

    batch_sizes = [1, 8, 32, 128]
    results = []
    import time

    for bs in batch_sizes:
        data = torch.randn(bs, input_size)

        # pytorch
        start = time.time()
        with torch.no_grad():
            for _ in range(100):
                _ = model_torch(data)
        torch_time = time.time() - start

        # onnx
        start = time.time()
        inp = {sess.get_inputs()[0].name: data.numpy().astype(np.float32)}
        for _ in range(100):
            _ = sess.run(None, inp)
        onnx_time = time.time() - start

        results.append({
            "batch_size": bs,
            "pytorch_time": torch_time,
            "onnx_time": onnx_time,
            "speedup": torch_time / onnx_time if onnx_time else 0
        })

    import pandas as pd
    pd.DataFrame(results).to_csv("reports/benchmark_results.csv", index=False)
    print("Benchmark saved to reports/benchmark_results.csv")

if __name__ == "__main__":
    convert_to_onnx()
    benchmark()