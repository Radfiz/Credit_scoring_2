# inspect_preprocessor.py

import joblib
from pathlib import Path

preprocessor_path = Path("models/trained/preprocessing_pipeline.pkl")

if preprocessor_path.exists():
    print(f"Загружаем {preprocessor_path}...")
    try:
        obj = joblib.load(preprocessor_path)
        print(f"Тип объекта: {type(obj)}")
        print(f"Содержимое (если dict): {obj if isinstance(obj, dict) else 'Not a dict'}")
        print(f"Атрибуты/ключи: {dir(obj)}")
        # Попробуем посмотреть на него подробнее
        if hasattr(obj, '__dict__'):
            print(f"__dict__: {obj.__dict__}")
        # Попробуем найти fit_transform
        if hasattr(obj, 'fit_transform'):
            print("Объект имеет метод fit_transform.")
        else:
            print("Объект НЕ имеет метода fit_transform.")
    except Exception as e:
        print(f"Ошибка при загрузке: {e}")
else:
    print(f"Файл {preprocessor_path} не найден.")