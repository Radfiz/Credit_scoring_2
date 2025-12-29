# scripts/load_testing.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Нагрузочное тестирование моделей на CPU/GPU
Определение оптимальной конфигурации ресурсов
"""

import sys, io, os
# перехват stdout в UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# подавить OpenMP-шум
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import time
import numpy as np
import pandas as pd
import torch
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing
import json
import matplotlib.pyplot as plt
from pathlib import Path

def test_cpu_vs_gpu():
    """Сравнение производительности на CPU и GPU"""
    print("="*60)
    print("ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ CPU vs GPU")
    print("="*60)
    
    # Проверка доступности GPU
    has_gpu = torch.cuda.is_available()
    print(f"  Доступен GPU: {'Да' if has_gpu else 'Нет'}")
    if has_gpu:
        print(f"  GPU устройство: {torch.cuda.get_device_name(0)}")
        print(f"  GPU память: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Загрузка метаданных модели
    with open("models/trained/nn_model_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    input_size = metadata['input_size']
    
    # Тестовые данные
    batch_sizes = [1, 4, 16, 64, 256]
    n_iterations = 100
    
    results = []
    
    for batch_size in batch_sizes:
        print(f"\nBatch size: {batch_size}")
        
        test_data_cpu = torch.randn(batch_size, input_size)
        
        # CPU тест
        start = time.time()
        for _ in range(n_iterations):
            # Имитация вычислений
            _ = test_data_cpu @ test_data_cpu.T
        cpu_time = time.time() - start
        
        print(f"  CPU время: {cpu_time:.4f} сек")
        
        if has_gpu:
            # GPU тест
            test_data_gpu = test_data_cpu.cuda()
            torch.cuda.synchronize()  # Синхронизация
            
            start = time.time()
            for _ in range(n_iterations):
                _ = test_data_gpu @ test_data_gpu.T
            torch.cuda.synchronize()
            gpu_time = time.time() - start
            
            speedup = cpu_time / gpu_time if gpu_time > 0 else 0
            print(f"  GPU время: {gpu_time:.4f} сек")
            print(f"  Ускорение GPU: {speedup:.2f}x")
        else:
            gpu_time = None
            speedup = None
        
        results.append({
            'batch_size': batch_size,
            'cpu_time': cpu_time,
            'gpu_time': gpu_time,
            'speedup': speedup
        })
    
    # Визуализация результатов
    df_results = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # График времени выполнения
    axes[0].plot(df_results['batch_size'], df_results['cpu_time'], 
                label='CPU', marker='o', linewidth=2)
    if has_gpu:
        axes[0].plot(df_results['batch_size'], df_results['gpu_time'], 
                    label='GPU', marker='s', linewidth=2)
    axes[0].set_xlabel('Batch Size')
    axes[0].set_ylabel('Время выполнения (сек)')
    axes[0].set_title('Производительность CPU vs GPU')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xscale('log')
    axes[0].set_yscale('log')
    
    # График ускорения
    if has_gpu:
        axes[1].plot(df_results['batch_size'], df_results['speedup'], 
                    marker='o', color='green', linewidth=2)
        axes[1].set_xlabel('Batch Size')
        axes[1].set_ylabel('Ускорение (CPU/GPU)')
        axes[1].set_title('Ускорение GPU относительно CPU')
        axes[1].axhline(y=1, color='r', linestyle='--', alpha=0.5)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('reports/cpu_gpu_performance.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Сохранение результатов
    df_results.to_csv('reports/cpu_gpu_benchmark.csv', index=False)
    
    print(f"\nРезультаты сохранены: reports/cpu_gpu_benchmark.csv")
    print(f"График сохранен: reports/cpu_gpu_performance.png")
    
    return df_results

def stress_test_model(provider='CPUExecutionProvider'):
    """
    Стресс-тестирование модели с разным количеством запросов
    """
    print(f"\n{"="*60}")
    print(f"СТРЕСС-ТЕСТИРОВАНИЕ ({provider})")
    print(f"{"="*60}")
    
    # Загрузка ONNX модели
    onnx_path = "models/onnx/credit_scoring_model.onnx"
    
    # Настройка провайдера
    providers = [provider]
    
    if provider == 'CUDAExecutionProvider' and not torch.cuda.is_available():
        print("GPU недоступен, используется CPU")
        providers = ['CPUExecutionProvider']
    
    ort_session = ort.InferenceSession(onnx_path, providers=providers)
    
    # Параметры тестирования
    with open("models/trained/nn_model_metadata.json", 'r') as f:
        metadata = json.load(f)
    input_size = metadata['input_size']
    
    test_scenarios = [
        {'name': 'Low Load', 'requests_per_second': 10, 'duration': 10},
        {'name': 'Medium Load', 'requests_per_second': 50, 'duration': 10},
        {'name': 'High Load', 'requests_per_second': 200, 'duration': 10},
        {'name': 'Peak Load', 'requests_per_second': 500, 'duration': 5},
    ]
    
    results = []
    
    for scenario in test_scenarios:
        print(f"\nСценарий: {scenario['name']}")
        print(f"  Запросов в секунду: {scenario['requests_per_second']}")
        print(f"  Длительность: {scenario['duration']} сек")
        
        total_requests = scenario['requests_per_second'] * scenario['duration']
        request_interval = 1.0 / scenario['requests_per_second']
        
        latencies = []
        successful_requests = 0
        failed_requests = 0
        
        start_time = time.time()
        request_count = 0
        
        while time.time() - start_time < scenario['duration'] and request_count < total_requests:
            request_start = time.time()
            
            try:
                # Генерация тестовых данных
                test_data = np.random.randn(1, input_size).astype(np.float32)
                
                # Выполнение инференса
                ort_inputs = {ort_session.get_inputs()[0].name: test_data}
                ort_session.run(None, ort_inputs)
                
                successful_requests += 1
                latency = time.time() - request_start
                latencies.append(latency)
                
            except Exception as e:
                failed_requests += 1
                print(f"    Ошибка запроса: {e}")
            
            request_count += 1
            
            # Поддержание заданной частоты запросов
            elapsed = time.time() - request_start
            if elapsed < request_interval:
                time.sleep(request_interval - elapsed)
        
        # Расчет метрик
        if latencies:
            avg_latency = np.mean(latencies) * 1000  # в миллисекундах
            p95_latency = np.percentile(latencies, 95) * 1000
            p99_latency = np.percentile(latencies, 99) * 1000
            throughput = successful_requests / scenario['duration']
        else:
            avg_latency = p95_latency = p99_latency = throughput = 0
        
        success_rate = (successful_requests / total_requests * 100) if total_requests > 0 else 0
        
        print(f"  Успешных запросов: {successful_requests}/{total_requests} ({success_rate:.1f}%)")
        print(f"  Пропускная способность: {throughput:.1f} req/sec")
        print(f"  Средняя задержка: {avg_latency:.2f} ms")
        print(f"  P95 задержка: {p95_latency:.2f} ms")
        print(f"  P99 задержка: {p99_latency:.2f} ms")
        
        results.append({
            'scenario': scenario['name'],
            'rps': scenario['requests_per_second'],
            'duration': scenario['duration'],
            'total_requests': total_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'success_rate': success_rate,
            'throughput': throughput,
            'avg_latency_ms': avg_latency,
            'p95_latency_ms': p95_latency,
            'p99_latency_ms': p99_latency,
            'provider': provider
        })
    
    # Сохранение результатов
    df_results = pd.DataFrame(results)
    
    # Визуализация
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    axes[0, 0].bar(range(len(results)), [r['throughput'] for r in results])
    axes[0, 0].set_xticks(range(len(results)))
    axes[0, 0].set_xticklabels([r['scenario'] for r in results])
    axes[0, 0].set_ylabel('Пропускная способность (req/sec)')
    axes[0, 0].set_title('Пропускная способность')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].bar(range(len(results)), [r['avg_latency_ms'] for r in results])
    axes[0, 1].set_xticks(range(len(results)))
    axes[0, 1].set_xticklabels([r['scenario'] for r in results])
    axes[0, 1].set_ylabel('Задержка (ms)')
    axes[0, 1].set_title('Средняя задержка')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].bar(range(len(results)), [r['success_rate'] for r in results])
    axes[1, 0].set_xticks(range(len(results)))
    axes[1, 0].set_xticklabels([r['scenario'] for r in results])
    axes[1, 0].set_ylabel('Успешность (%)')
    axes[1, 0].set_title('Процент успешных запросов')
    axes[1, 0].axhline(y=99, color='r', linestyle='--', alpha=0.5)
    axes[1, 0].grid(True, alpha=0.3)
    
    # График зависимости задержки от нагрузки
    axes[1, 1].plot([r['rps'] for r in results], [r['avg_latency_ms'] for r in results], 
                    marker='o', linewidth=2)
    axes[1, 1].set_xlabel('Запросов в секунду')
    axes[1, 1].set_ylabel('Задержка (ms)')
    axes[1, 1].set_title('Зависимость задержки от нагрузки')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'reports/stress_test_{provider}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Сохранение в CSV
    df_results.to_csv(f'reports/stress_test_{provider}.csv', index=False)
    
    print(f"\nРезультаты стресс-теста сохранены:")
    print(f"  reports/stress_test_{provider}.csv")
    print(f"  reports/stress_test_{provider}.png")
    
    return df_results

def determine_optimal_configuration():
    """
    Определение оптимальной конфигурации ресурсов для продакшена
    """
    print("\n" + "="*60)
    print("ОПРЕДЕЛЕНИЕ ОПТИМАЛЬНОЙ КОНФИГУРАЦИИ РЕСУРСОВ")
    print("="*60)
    
    # Анализ результатов тестирования
    recommendations = []
    
    # 1. Анализ CPU vs GPU
    cpu_gpu_results = pd.read_csv('reports/cpu_gpu_benchmark.csv')
    
    if 'speedup' in cpu_gpu_results.columns and cpu_gpu_results['speedup'].notna().any():
        avg_speedup = cpu_gpu_results['speedup'].mean()
        
        if avg_speedup > 2.0:
            recommendations.append({
                'component': 'Вычислительные ресурсы',
                'recommendation': 'Использовать GPU инстансы',
                'reason': f'Среднее ускорение GPU: {avg_speedup:.2f}x',
                'priority': 'HIGH'
            })
        else:
            recommendations.append({
                'component': 'Вычислительные ресурсы',
                'recommendation': 'Использовать CPU инстансы',
                'reason': 'GPU не дает значительного ускорения',
                'priority': 'MEDIUM'
            })
    
    # 2. Анализ стресс-тестов
    try:
        cpu_stress = pd.read_csv('reports/stress_test_CPUExecutionProvider.csv')
        
        # Находим точку, где задержка превышает 100ms
        high_latency_scenarios = cpu_stress[cpu_stress['avg_latency_ms'] > 100]
        
        if not high_latency_scenarios.empty:
            max_rps_before_degradation = high_latency_scenarios['rps'].min() - 10
            
            recommendations.append({
                'component': 'Масштабирование',
                'recommendation': f'Ограничить до {max_rps_before_degradation} RPS на инстанс',
                'reason': 'Задержка превышает 100ms при более высоких нагрузках',
                'priority': 'HIGH'
            })
        
        # Рекомендации по репликам
        target_rps = 1000  # Целевая нагрузка
        instance_capacity = cpu_stress[cpu_stress['success_rate'] > 99]['rps'].max()
        
        if instance_capacity > 0:
            required_instances = int(np.ceil(target_rps / instance_capacity))
            
            recommendations.append({
                'component': 'Количество инстансов',
                'recommendation': f'Развернуть {required_instances} инстансов',
                'reason': f'Для обработки {target_rps} RPS при {instance_capacity:.0f} RPS на инстанс',
                'priority': 'MEDIUM'
            })
    
    except FileNotFoundError:
        print("Файлы результатов стресс-теста не найдены")
    
    # 3. Рекомендации по памяти
    with open("models/trained/nn_model_metadata.json", 'r') as f:
        metadata = json.load(f)
    
    model_size_mb = Path("models/onnx/credit_scoring_model.onnx").stat().st_size / (1024 * 1024)
    
    recommendations.append({
        'component': 'Память',
        'recommendation': f'Выделить минимум {int(model_size_mb * 3)} MB RAM на инстанс',
        'reason': f'Размер модели: {model_size_mb:.1f} MB, требуется буфер для данных',
        'priority': 'HIGH'
    })
    
    # 4. Вывод рекомендаций
    print("\nРЕКОМЕНДАЦИИ ПО КОНФИГУРАЦИИ ДЛЯ ПРОДАКШЕНА:\n")
    
    for i, rec in enumerate(recommendations, 1):
        priority_color = {
            'HIGH': '🔴',
            'MEDIUM': '🟡',
            'LOW': '🟢'
        }.get(rec['priority'], '⚪')
        
        print(f"{priority_color} {i}. {rec['component']}")
        print(f"   Рекомендация: {rec['recommendation']}")
        print(f"   Причина: {rec['reason']}")
        print()
    
    # Сохранение отчета
    report = {
        'generated_at': pd.Timestamp.now().isoformat(),
        'model_info': metadata,
        'recommendations': recommendations,
        'test_files': [
            'reports/cpu_gpu_benchmark.csv',
            'reports/stress_test_CPUExecutionProvider.csv',
            'reports/benchmark_results.csv'
        ]
    }
    
    with open('reports/production_configuration_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nПолный отчет сохранен: reports/production_configuration_report.json")
    
    return recommendations

def main():
    print("="*60)
    print("ОПТИМИЗАЦИЯ И ТЕСТИРОВАНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("="*60)
    
    # 1. Сравнение CPU vs GPU
    cpu_gpu_results = test_cpu_vs_gpu()
    
    # 2. Стресс-тестирование на CPU
    cpu_stress_results = stress_test_model('CPUExecutionProvider')
    
    # 3. Стресс-тестирование на GPU (если доступен)
    if torch.cuda.is_available():
        gpu_stress_results = stress_test_model('CUDAExecutionProvider')
    
    # 4. Определение оптимальной конфигурации
    recommendations = determine_optimal_configuration()
    
    print("="*60)
    print("ТЕСТИРОВАНИЕ ЗАВЕРШЕНО!")
    print("="*60)
    
    return {
        'cpu_gpu_results': cpu_gpu_results,
        'cpu_stress_results': cpu_stress_results,
        'recommendations': recommendations
    }

if __name__ == "__main__":
    results = main()