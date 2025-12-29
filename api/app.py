# api/app.py
import structlog
from fastapi import FastAPI, HTTPException
from api.models import CreditApplication, PredictionResponse
from api.services.prediction import PredictionService
from api.middleware.metrics import MetricsMiddleware
from api.middleware.logging import LoggingMiddleware
from config.settings import get_settings # Предполагаем, что настройки будут тут
import uvicorn
from prometheus_client import make_asgi_app

settings = get_settings()

# Настройка structlog
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer() # Используем JSON для логов
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

app = FastAPI(title="Credit Scoring API", version="1.0.0")

# Добавляем middleware
app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)

# Добавляем endpoint для метрик Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Инициализируем сервис предсказания
# Важно: путь к модели должен быть доступен из контейнера или текущей среды
prediction_service = PredictionService(settings.model_path)

@app.get("/")
def read_root():
    return {"message": "Credit Scoring API is running"}

@app.get("/health")
def health_check():
    # Проверка, что модель загружена
    if prediction_service.session is None:
        raise HTTPException(status_code=500, detail="Model is not loaded")
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: CreditApplication):
    try:
        probability, prediction = prediction_service.predict(request.features)
        return PredictionResponse(probability=probability, prediction=prediction)
    except ValueError as ve:
        logger.error("Invalid input features", error=str(ve))
        raise HTTPException(status_code=422, detail=str(ve))
    except RuntimeError as re:
        logger.error("Prediction error", error=str(re))
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.error("Unexpected error during prediction", error=str(e))
        raise HTTPException(status_code=500, detail="An unexpected error occurred")

if __name__ == "__main__":
    # Для запуска с помощью uvicorn напрямую
    # Путь к модели будет передан через переменную окружения или аргумент
    # uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True) # reload=True только для разработки