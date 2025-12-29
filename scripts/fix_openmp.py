# scripts/fix_openmp.py
#!/usr/bin/env python
"""
Исправление конфликта OpenMP библиотек
"""
import os
import sys

def fix_openmp_issue():
    """Добавляет переменную окружения для исправления OpenMP"""
    print("Исправление конфликта OpenMP...")
    
    # Устанавливаем переменную окружения
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    # Также можно попробовать другие настройки
    os.environ['OMP_NUM_THREADS'] = '1'  # Ограничиваем потоки
    os.environ['MKL_NUM_THREADS'] = '1'
    
    print("Переменные окружения установлены:")
    print(f"   KMP_DUPLICATE_LIB_OK = {os.environ.get('KMP_DUPLICATE_LIB_OK')}")
    print(f"   OMP_NUM_THREADS = {os.environ.get('OMP_NUM_THREADS')}")
    print(f"   MKL_NUM_THREADS = {os.environ.get('MKL_NUM_THREADS')}")
    
    return True

def check_environment():
    """Проверка окружения"""
    print("\nПроверка окружения:")
    
    import torch
    print(f"   PyTorch версия: {torch.__version__}")
    print(f"   CUDA доступен: {torch.cuda.is_available()}")
    
    import onnxruntime as ort
    print(f"   ONNX Runtime версия: {ort.__version__}")
    
    # Доступные провайдеры
    print(f"   Доступные провайдеры ONNX Runtime:")
    for provider in ort.get_available_providers():
        print(f"     - {provider}")
    
    return True

if __name__ == "__main__":
    fix_openmp_issue()
    check_environment()