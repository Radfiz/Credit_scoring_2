# scripts/test_imports.py
#!/usr/bin/env python
import sys
import os

print("Текущая рабочая директория:", os.getcwd())
print("\nПуть Python:")
for p in sys.path:
    print(f"  {p}")

# Пробуем импортировать
try:
    sys.path.append('.')  # Добавляем текущую директорию
    from models.neural_network import CreditScoringNN
    print("\n✅ Модель импортирована успешно!")
except ImportError as e:
    print(f"\n❌ Ошибка импорта: {e}")
    
    # Проверяем существование файла
    print("\nПроверка файлов:")
    files_to_check = [
        'models/neural_network.py',
        'models/__init__.py',
        'scripts/',
        'USI_CCD.py'
    ]
    
    for file in files_to_check:
        exists = os.path.exists(file)
        print(f"  {file}: {'✅ существует' if exists else '❌ отсутствует'}")