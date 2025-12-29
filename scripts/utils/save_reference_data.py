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

        # --- ИСПРАВЛЕНИЕ ПУТИ К ПРЕПРОЦЕССОРУ ---
        # Теперь ищем preprocessing_pipeline.pkl
        possible_preprocessor_paths = [
            Path("models/trained/preprocessing_pipeline.pkl"), # Путь из лога
            Path("models/preprocessing_pipeline.pkl"),
            Path("../models/trained/preprocessing_pipeline.pkl"),
            Path("../models/preprocessing_pipeline.pkl"),
            # Пути из проекта HW_ML
            Path("../../../HW_ML/models/preprocessing_pipeline.pkl"),
            Path("../../../HW_ML/scripts/model_training/preprocessing_pipeline.pkl"),
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

        # --- ИСПРАВЛЕНИЕ ЗАГРУЗКИ И ПРИМЕНЕНИЯ ПРЕПРОЦЕССОРА ---
        # Загружаем словарь
        pipeline_dict = joblib.load(preprocessor_path)

        # Извлекаем компоненты
        feature_creator = pipeline_dict['feature_creator']
        preprocessor = pipeline_dict['preprocessor']
        # num_cols = pipeline_dict['num_cols'] # Не обязательно использовать, если ColumnTransformer сам разберётся
        # cat_cols = pipeline_dict['cat_cols']

        print(f"Feature creator: {type(feature_creator)}")
        print(f"Preprocessor: {type(preprocessor)}")

        # Применяем Feature Engineering (transform, не fit_transform!)
        X_transformed = feature_creator.transform(X)
        print(f"Данные после Feature Engineering: {X_transformed.shape}")

        # Применяем препроцессор (transform, не fit_transform!)
        X_processed = preprocessor.transform(X_transformed)
        print(f"Данные после препроцессинга: {X_processed.shape}")
        # --- КОНЕЦ ИСПРАВЛЕНИЯ ---

        # Сохраняем эталонные признаки
        ref_data_path = Path("data/processed/reference_features.csv")
        ref_data_path.parent.mkdir(exist_ok=True)

        # Попробуем восстановить имена столбцов из препроцессора
        # ColumnTransformer.get_feature_names_out() требует, чтобы на вход подавался DataFrame с правильными именами столбцов
        # X_transformed - это результат feature_creator.transform(X), он должен иметь правильные имена
        try:
            feature_names = preprocessor.get_feature_names_out(X_transformed.columns)
            ref_df = pd.DataFrame(X_processed, columns=feature_names)
        except Exception as e:
            print(f"Не удалось получить имена столбцов из препроцессора: {e}")
            # Если не получилось, используем числовые индексы, но лучше бы получить имена
            # ref_df = pd.DataFrame(X_processed)
            # Попробуем получить имена столбцов из самого препроцессора, если get_feature_names_out не работает
            # Это может быть сложно, поэтому сохраним как есть, но с предупреждением
            print("Предупреждение: имена столбцов не восстановлены, используем индексы. Это может повлиять на Evidently.")
            ref_df = pd.DataFrame(X_processed)

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