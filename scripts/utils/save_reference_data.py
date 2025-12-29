# scripts/utils/save_reference_data.py

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import sys
import os

# Устанавливаем кодировку UTF-8 для Windows
if sys.platform == "win32":
    os.environ["PYTHONIOENCODING"] = "utf-8"

def load_and_prepare_reference_data():
    """Загружает и готовит эталонные данные для Evidently."""
    print("Загрузка данных...")

    # --- ИСПРАВЛЕНИЕ ПУТИ К ДАННЫМ ---
    possible_data_paths = [
        Path("data/raw/UCI_Credit_Card.csv"),
        Path("data/UCI_Credit_Card.csv"),
        Path("../data/raw/UCI_Credit_Card.csv"),
        Path("../data/UCI_Credit_Card.csv"),
        Path("../../data/raw/UCI_Credit_Card.csv"),
        Path("../../data/UCI_Credit_Card.csv"),
        # Попробуем пути из проекта HW_ML
        Path("../../../HW_ML/data/UCI_Credit_Card.csv"),
        Path("../../../HW_ML/data/raw/UCI_Credit_Card.csv"),
    ]

    data_path = None
    for path in possible_data_paths:
        if path.exists():
            data_path = path
            print(f"Найдены данные: {path}")
            break

    if data_path is None:
        print("Данные не найдены по стандартным путям. Пожалуйста, укажите правильный путь.")
        return None
    # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

    try:
        df = pd.read_csv(data_path, encoding='utf-8')
        print(f"Загружено: {len(df)} записей")

        # Определяем таргет
        TARGET = "default.payment.next.month"
        ID_COL = "ID"
        y = df[TARGET].values
        X = df.drop(columns=[TARGET, ID_COL])

        # --- Feature engineering class (из scripts/data_loader.py) ---
        from sklearn.base import BaseEstimator, TransformerMixin
        class FeatureEngineer(BaseEstimator, TransformerMixin):
            def __init__(self):
                self.age_bins = None
                self.age_labels = None

            def fit(self, X, y=None):
                self.age_bins = [0, 25, 35, 50, 100]
                self.age_labels = ['Young', 'Adult', 'Senior', 'Elder']
                return self

            def transform(self, X):
                X = X.copy()
                X['age_group'] = pd.cut(X['AGE'], bins=self.age_bins, labels=self.age_labels, right=False)
                X['utilization'] = X['BILL_AMT1'] / (X['LIMIT_BAL'] + 1)
                X['pay_ratio'] = X['PAY_AMT1'] / (X['BILL_AMT1'] + 1)
                bill_cols = [f'BILL_AMT{i}' for i in range(1, 7)]
                X['bill_trend'] = X[bill_cols].diff(axis=1).mean(axis=1, skipna=True).fillna(0)
                return X

            def fit_transform(self, X, y=None):
                return self.fit(X, y).transform(X)
        # --- Конец FeatureEngineer ---

        # Применяем Feature Engineering
        fe = FeatureEngineer()
        X_transformed = fe.fit_transform(X)
        print(f"Данные после Feature Engineering: {X_transformed.shape}")

        # --- ИСПРАВЛЕНИЕ ПУТИ К ПРЕПРОЦЕССОРУ ---
        # Теперь ищем preprocessing_pipeline.pkl
        possible_preprocessor_paths = [
            Path("models/preprocessing_pipeline.pkl"),
            Path("models/trained/preprocessing_pipeline.pkl"), # Возможно, в подпапке
            Path("../models/preprocessing_pipeline.pkl"),
            Path("../models/trained/preprocessing_pipeline.pkl"),
            Path("../../models/preprocessing_pipeline.pkl"),
            Path("../../models/trained/preprocessing_pipeline.pkl"),
            # Пути из проекта HW_ML
            Path("../../../HW_ML/models/preprocessing_pipeline.pkl"),
            Path("../../../HW_ML/scripts/model_training/preprocessing_pipeline.pkl"), # Если он был сохранен там
            # Добавьте другие возможные пути, если нужно
        ]

        preprocessor_path = None
        for path in possible_preprocessor_paths:
            if path.exists():
                preprocessor_path = path
                print(f"Найден препроцессор: {path}")
                break

        if preprocessor_path is None:
            print("Препроцессор (preprocessing_pipeline.pkl) не найден по стандартным путям. Пожалуйста, укажите правильный путь.")
            return None
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        preprocessor = joblib.load(preprocessor_path)

        # Применяем препроцессор
        X_processed = preprocessor.fit_transform(X_transformed) # fit_transform, т.к. это эталон
        print(f"Данные после препроцессинга: {X_processed.shape}")

        # Сохраняем эталонные признаки
        ref_data_path = Path("data/processed/reference_features.csv")
        ref_data_path.parent.mkdir(exist_ok=True)

        # Попробуем восстановить имена столбцов из препроцессора
        feature_names = preprocessor.get_feature_names_out(X_transformed.columns)
        ref_df = pd.DataFrame(X_processed, columns=feature_names)

        ref_df.to_csv(ref_data_path, index=False)
        print(f"Эталонные признаки сохранены: {ref_data_path}")

        # Сохраняем также таргет (y), если планируется concept drift
        ref_target_path = Path("data/processed/reference_target.csv")
        pd.DataFrame(y, columns=[TARGET]).to_csv(ref_target_path, index=False)
        print(f"Эталонный таргет сохранен: {ref_target_path}")

        return ref_df, y

    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    ref_features, ref_target = load_and_prepare_reference_data()
    if ref_features is not None:
        print("Эталонные данные успешно подготовлены и сохранены.")
    else:
        print("Ошибка при подготовке эталонных данных.")