#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
import onnxruntime as ort
import numpy as np
import json
import pandas as pd
import time
from pathlib import Path
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

def quantize():
    print("Quantizing ONNX model ...")
    original = "models/onnx/credit_scoring_model.onnx"
    quantized = "models/onnx/credit_scoring_model_quantized.onnx"

    quantize_dynamic(
        model_input=original,
        model_output=quantized,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized model saved: {quantized}")
    return quantized

def benchmark_quantized():
    print("Benchmark original vs quantized ...")
    sess_orig = ort.InferenceSession("models/onnx/credit_scoring_model.onnx")
    sess_quant = ort.InferenceSession("models/onnx/credit_scoring_model_quantized.onnx")
    input_size = json.load(open("models/trained/nn_model_metadata.json"))["input_size"]

    batch_sizes = [1, 4, 16, 64, 256]
    res = []

    for bs in batch_sizes:
        data = np.random.randn(bs, input_size).astype(np.float32)
        # original
        t0 = time.time()
        for _ in range(100):
            _ = sess_orig.run(None, {sess_orig.get_inputs()[0].name: data})
        t_orig = time.time() - t0

        # quantized
        t0 = time.time()
        for _ in range(100):
            _ = sess_quant.run(None, {sess_quant.get_inputs()[0].name: data})
        t_quant = time.time() - t0

        res.append({
            "batch_size": bs,
            "original_time": t_orig,
            "quantized_time": t_quant,
            "speedup": t_orig / t_quant if t_quant else 0
        })

    pd.DataFrame(res).to_csv("reports/quantization_benchmark.csv", index=False)
    print("Quantization benchmark saved.")

def validate_accuracy():
    print("Validate accuracy drop ...")
    sess_orig = ort.InferenceSession("models/onnx/credit_scoring_model.onnx")
    sess_quant = ort.InferenceSession("models/onnx/credit_scoring_model_quantized.onnx")

    # load test sample
    df = pd.read_csv("data/UCI_Credit_Card.csv")
    target = "default.payment.next.month"
    id_col = "ID"
    y = df[target].values
    X = df.drop(columns=[target, id_col])

    from USI_CCD import FeatureCreator, get_lists, build_prep
    fc = FeatureCreator()
    X_proc = fc.transform(X)
    num, cat = get_lists(X_proc)
    prep = build_prep(num, cat)
    X_final = prep.fit_transform(X_proc)
    if hasattr(X_final, "toarray"):
        X_final = X_final.toarray()
    X_final = X_final.astype(np.float32)

    _, X_test, _, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y)

    preds_orig = sess_orig.run(None, {sess_orig.get_inputs()[0].name: X_test})[0].ravel()
    preds_quant = sess_quant.run(None, {sess_quant.get_inputs()[0].name: X_test})[0].ravel()

    def metrics(p):
        return {
            "auc": roc_auc_score(y_test, p),
            "f1": f1_score(y_test, p > 0.5),
            "precision": precision_score(y_test, p > 0.5),
            "recall": recall_score(y_test, p > 0.5)
        }

    orig_m = metrics(preds_orig)
    quant_m = metrics(preds_quant)
    print("Original ONNX metrics:", orig_m)
    print("Quantized ONNX metrics:", quant_m)

    with open("reports/quantization_accuracy.json", "w") as f:
        json.dump({"original": orig_m, "quantized": quant_m}, f, indent=2)

if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    quantize()
    benchmark_quantized()
    validate_accuracy()