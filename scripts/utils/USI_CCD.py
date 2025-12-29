#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
UCI Credit-Card Default – полный пайп-лайн с сохранением моделей
python USI_CCD.py
"""
import warnings, pathlib, json, shutil
warnings.filterwarnings("ignore")

import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (f1_score, precision_score, recall_score,
                             roc_auc_score, classification_report,
                             ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib

# ----------- конфиг ----------------------------------
RANDOM_STATE = 42
TEST_SIZE    = 0.2
CV_SPLITS    = 5
DATA_DIR     = pathlib.Path("data")
MODEL_DIR    = pathlib.Path("models")
REPORT_DIR   = pathlib.Path("reports")
for d in (DATA_DIR, MODEL_DIR, REPORT_DIR):
    d.mkdir(exist_ok=True)

TARGET = "default.payment.next.month"
ID_COL = "ID"

# ----------- 1. скачивание ---------------------------
def download() -> pathlib.Path:
    csv_path = DATA_DIR / "UCI_Credit_Card.csv"
    if csv_path.exists():
        return csv_path
    print("⬇️  Скачиваю датасет…")
    folder = kagglehub.dataset_download("uciml/default-of-credit-card-clients-dataset")
    src = pathlib.Path(folder) / "UCI_Credit_Card.csv"
    shutil.copy(src, csv_path)
    return csv_path

# ----------- 2. EDA ----------------------------------
def eda(df: pd.DataFrame, save_dir: pathlib.Path):
    print("📊  EDA-отчёт …")
    with pd.ExcelWriter(save_dir / "eda.xlsx", engine="openpyxl") as xl:
        df.describe(include="all").T.to_excel(xl, sheet_name="describe")
        df.isna().sum().to_frame("miss").to_excel(xl, sheet_name="missing")

# ----------- 3. инженерия + очистка ------------------
class FeatureCreator(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X: pd.DataFrame):
        X = X.copy()
        if "PAY_0" in X.columns:
            X = X.rename(columns={"PAY_0": "PAY_1"})
        X.loc[X["EDUCATION"].isin([0, 5, 6]), "EDUCATION"] = 4
        X.loc[X["MARRIAGE"] == 0, "MARRIAGE"] = 3
        for col in [f"PAY_{i}" for i in range(1, 7)]:
            if col in X.columns:
                X[col] = np.where(X[col] <= 0, 0, 1)
        X["utilization"] = X["BILL_AMT1"] / (X["LIMIT_BAL"] + 1)
        X["pay_ratio"]   = X["PAY_AMT1"] / (X["BILL_AMT1"] + 1)
        X["bill_trend"]  = (X["BILL_AMT1"] - X["BILL_AMT6"]) / (X["BILL_AMT6"] + 1)
        X["age_group"]   = pd.cut(X["AGE"], bins=[20, 30, 40, 50, 60, 100],
                                  labels=[1, 2, 3, 4, 5]).astype(int)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        return X

# ----------- 4. списки признаков ---------------------
def get_lists(df):
    num = (["LIMIT_BAL", "AGE"] +
           [c for c in df.columns if ("BILL_" in c or "PAY_AMT" in c)] +
           ["utilization", "pay_ratio", "bill_trend"])
    cat = (["SEX", "EDUCATION", "MARRIAGE", "age_group"] +
           [f"PAY_{i}" for i in range(1, 7)])
    return num, cat

# ----------- 5. preprocessor -------------------------
def build_prep(num_cols, cat_cols) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
        ])

# ----------- 6. стратегии сэмплинга ------------------
def get_sampler(name: str):
    if name == "smote":
        return SMOTE(random_state=RANDOM_STATE)
    if name == "rus":
        return RandomUnderSampler(random_state=RANDOM_STATE)
    return None

# ----------- 7. модели -------------------------------
def get_models():
    return {
        "rf": RandomForestClassifier(
            n_estimators=400, max_depth=None, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "gb": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE),
        "lgb": LGBMClassifier(
            n_estimators=600, learning_rate=0.05, num_leaves=31,
            class_weight="balanced", random_state=RANDOM_STATE, verbosity=-1)
    }

# ----------- 8. evaluate + save ----------------------
def evaluate_and_save(model, X_tr, X_te, y_tr, y_te, tag: str, save_dir: pathlib.Path):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    probs = model.predict_proba(X_te)[:, 1]

    metrics = {
        "f1":   round(f1_score(y_te, preds), 4),
        "prec": round(precision_score(y_te, preds), 4),
        "rec":  round(recall_score(y_te, preds), 4),
        "auc":  round(roc_auc_score(y_te, probs), 4)
    }
    print(f"{tag}  ->  {metrics}")
    print(classification_report(y_te, preds))

    ConfusionMatrixDisplay.from_predictions(y_te, preds)
    plt.title(f"{tag}  –  CM")
    plt.tight_layout()
    plt.savefig(save_dir / f"{tag}_cm.png", dpi=150)
    plt.close()

    # СОХРАНЕНИЕ модели
    model_path = MODEL_DIR / f"{tag}_model.pkl"
    joblib.dump(model, model_path)
    print(f"✅  Модель сохранена: {model_path.resolve()}")
    return metrics

# ----------- 9. main --------------------------------
def main():
    csv = download()
    df  = pd.read_csv(csv)
    print("📋  Колонки в файле:", df.columns.tolist())
    eda(df, REPORT_DIR)

    y = df[TARGET]
    X = df.drop(columns=[TARGET, ID_COL])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    results = {}
    for strategy in (None, "smote", "rus"):
        tag = "no_sampling" if strategy is None else strategy
        sampler = get_sampler(strategy)
        print(f"\n===== {tag.upper()} =====")

        pipe = ImbPipeline(
            steps=[("fe", FeatureCreator()),
                   ("prep", build_prep(*get_lists(X_train))),
                   ("samp", sampler),
                   ("clf", get_models()["lgb"])]
        )
        results[tag] = evaluate_and_save(pipe, X_train, X_test, y_train, y_test, tag, REPORT_DIR)

    pd.DataFrame(results).T.to_csv(REPORT_DIR / "summary.csv")
    print("\n📈  Итоговые метрики:")
    print(pd.DataFrame(results).T)

if __name__ == "__main__":
    main()