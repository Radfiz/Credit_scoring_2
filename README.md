Credit Scoring MLOps System
Промышленная система MLOps для кредитного скоринга с полным жизненным циклом автоматизации.

Возможности
Обучение модели: Нейронная сеть, обученная на данных UCI Credit Card.
Конвертация в ONNX: Модель экспортирована в формат ONNX для оптимизации инференса.
Оптимизация: Применена динамическая квантизация PyTorch-модели.
Бенчмаркинг: Сравнение производительности PyTorch, ONNX, CPU, GPU.
Контейнеризация: API упаковано в Docker-образ.
API: FastAPI приложение для инференса модели.
Мониторинг дрифта: Интеграция Evidently AI для обнаружения дрейфа данных.
Логирование: Структурированное логирование с structlog.
Метрики: Экспорт метрик для Prometheus.
Структура проекта
(Описывает текущую структуру, отражая реализованные компоненты)

credit-scoring-mlops/
├── api/                    # FastAPI application
│   ├── app.py
│   ├── models.py
│   ├── services/
│   │   ├── prediction.py
│   │   └── drift_detection.py
│   ├── middleware/
│   │   ├── logging.py
│   │   └── metrics.py
│   └── utils/
│       └── feature_engineering.py
├── config/                # Configuration files (pydantic BaseSettings)
│   └── settings.py
├── data/
│   ├── raw/               # Raw dataset (UCI_Credit_Card.csv)
│   └── processed/         # Processed data (reference_features.csv, reference_target.csv)
├── deployment/
│   └── docker/
│       └── Dockerfile.api # Dockerfile for API
├── models/
│   ├── trained/
│   │   ├── onnx/          # ONNX model files
│   │   └── preprocessing_pipeline.pkl # Pipeline used for training
│   └── archived/
├── scripts/
│   ├── model_training/    # Training scripts (from Stage 1)
│   ├── utils/            # Utility scripts (e.g., save_reference_data.py)
│   └── monitoring/       # Monitoring scripts (e.g., drift_detection.py - placeholder if not run separately)
├── monitoring/           # Monitoring configurations and reports
│   └── reports/
├── requirements*.txt     # Dependencies for different components
├── Makefile              # Commands for common tasks
├── Dockerfile.api        # (If separate)
└── README.md            # This file

Быстрый старт (для запущенного API)
Убедитесь, что Docker установлен.
Соберите образ:
docker build -t credit-scoring-api:latest -f deployment/docker/Dockerfile.api .
Запустите контейнер:
docker run -p 8000:8000 --rm credit-scoring-api:latest

API будет доступен по адресу: http://localhost:8000
/health: Проверка состояния.
/docs: Swagger UI.
/predict: Эндпоинт для предсказания (POST-запрос).
Состояние проекта
Этап 1 (Подготовка модели): Завершён.
Этап 2 (Cloud Infrastructure): Не начат.
Этап 3 (Контейнеризация и оркестрация): Частично завершён (Docker).
Этап 4 (CI/CD): Не начат.
Этап 5 (Мониторинг и observability): Частично начат (метрики, логирование).
Этап 6 (Мониторинг дрифта): Частично начат (интеграция Evidently в API).
Этап 7 (Пайплайн переобучения): Не начат.
