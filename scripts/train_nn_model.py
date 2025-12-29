#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from USI_CCD import FeatureCreator, get_lists, build_prep

class CreditScoringNN(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super().__init__()
        self.layer1 = nn.Linear(input_size, 256)
        self.bn1   = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(256, 128)
        self.bn2   = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout_rate)
        self.layer3 = nn.Linear(128, 64)
        self.bn3   = nn.BatchNorm1d(64)
        self.drop3 = nn.Dropout(dropout_rate)
        self.layer4 = nn.Linear(64, 32)
        self.bn4   = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.bn1(self.layer1(x)))
        x = self.drop1(x)
        x = torch.relu(self.bn2(self.layer2(x)))
        x = self.drop2(x)
        x = torch.relu(self.bn3(self.layer3(x)))
        x = self.drop3(x)
        x = torch.relu(self.bn4(self.layer4(x)))
        return torch.sigmoid(self.output(x))

def train_and_save():
    data_path = "data/UCI_Credit_Card.csv"
    df = pd.read_csv(data_path)
    target = "default.payment.next.month"
    id_col = "ID"
    y = df[target].values
    X = df.drop(columns=[target, id_col])

    fc = FeatureCreator()
    X_proc = fc.fit_transform(X)
    num, cat = get_lists(X_proc)
    prep = build_prep(num, cat)
    X_final = prep.fit_transform(X_proc)
    if hasattr(X_final, "toarray"):
        X_final = X_final.toarray()

    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, test_size=0.2, random_state=42, stratify=y)

    X_train = X_train.astype(np.float32)
    X_test  = X_test.astype(np.float32)

    model = CreditScoringNN(input_size=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    epochs = 30
    batch = 128

    train_ds = TensorDataset(torch.tensor(X_train), torch.tensor(y_train.reshape(-1, 1)))
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for xb, yb in train_dl:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 5 == 0:
            print(f"epoch {epoch+1:02d}  loss={loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(torch.tensor(X_test)).numpy().ravel()
    auc = roc_auc_score(y_test, preds)
    print(f"Test ROC-AUC: {auc:.4f}")

    os.makedirs("models/trained", exist_ok=True)
    torch.save(model.state_dict(), "models/trained/credit_scoring_nn.pth")

    meta = {
        "input_size": X_train.shape[1],
        "architecture": "256-128-64-32-1",
        "parameters": sum(p.numel() for p in model.parameters()),
        "training_date": datetime.now().isoformat(),
        "metrics": {"auc": float(auc)}
    }
    with open("models/trained/nn_model_metadata.json", "w") as f:
        import json
        json.dump(meta, f, indent=2)
    print("Model and metadata saved.")

if __name__ == "__main__":
    train_and_save()