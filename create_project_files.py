# create_project_files.py
# Запуск: python create_project_files.py
# Создаёт все файлы-заглушки ровно по тем путям, что вы указали.

from pathlib import Path

# ---------- 1. Содержимое файлов ----------
FILES = {
"api/models.py": '''\
from pydantic import BaseModel
from typing import List

class CreditApplication(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    probability: float
    prediction: int
''',

"api/utils/feature_engineering.py": '''\
import numpy as np
from typing import List

def validate_and_prepare_features(features: List[float], expected_length: int = 26) -> np.ndarray:
    if len(features) != expected_length:
        raise ValueError(f"Expected {expected_length} features, got {len(features)}")
    return np.array(features).reshape(1, -1)
''',

"api/services/prediction.py": '''\
import onnxruntime as rt
import numpy as np
from pathlib import Path
import structlog
from typing import List
from api.utils.feature_engineering import validate_and_prepare_features

logger = structlog.get_logger()

class PredictionService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.load_model()

    def load_model(self):
        try:
            self.session = rt.InferenceSession(self.model_path)
            logger.info("ONNX model loaded", model_path=self.model_path)
        except Exception as e:
            logger.error("Failed to load ONNX model", error=str(e))
            raise

    def predict(self, features: List[float]) -> tuple[float, int]:
        try:
            input_data = validate_and_prepare_features(features)
            input_name = self.session.get_inputs()[0].name
            result = self.session.run(None, {input_name: input_data.astype(np.float32)})
            logits = result[0]
            import torch
            probability = torch.sigmoid(torch.tensor(logits)).item()
            prediction = 1 if probability > 0.5 else 0
            logger.info("Prediction done", probability=probability, prediction=prediction)
            return probability, prediction
        except ValueError as ve:
            logger.error("Feature validation error", error=str(ve))
            raise
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise RuntimeError(f"Prediction failed: {str(e)}")
''',

"api/middleware/metrics.py": '''\
from prometheus_client import Counter, Histogram
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time

REQUEST_COUNT = Counter(
    "http_requests_total", "Total number of requests",
    ["method", "endpoint", "status_code"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency",
    ["method", "endpoint"]
)

class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response: Response = await call_next(request)
        end_time = time.time()
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status_code=response.status_code
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(end_time - start_time)
        return response
''',

"api/middleware/logging.py": '''\
import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response
import time

logger = structlog.get_logger()

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        logger.info(
            "Request handled",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            process_time=f"{time.time() - start:.4f}s"
        )
        return response
''',

"api/app.py": '''\
import structlog
from fastapi import FastAPI, HTTPException
from api.models import CreditApplication, PredictionResponse
from api.services.prediction import PredictionService
from api.middleware.metrics import MetricsMiddleware
from api.middleware.logging import LoggingMiddleware
from config.settings import get_settings
from prometheus_client import make_asgi_app
import uvicorn

settings = get_settings()
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
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)
logger = structlog.get_logger()

app = FastAPI(title="Credit Scoring API", version="1.0.0")
app.add_middleware(LoggingMiddleware)
app.add_middleware(MetricsMiddleware)
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

prediction_service = PredictionService(settings.model_path)

@app.get("/")
def read_root():
    return {"message": "Credit Scoring API is running"}

@app.get("/health")
def health_check():
    if prediction_service.session is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictionResponse)
def predict(request: CreditApplication):
    try:
        prob, pred = prediction_service.predict(request.features)
        return PredictionResponse(probability=prob, prediction=pred)
    except ValueError as ve:
        logger.error("Invalid input", error=str(ve))
        raise HTTPException(status_code=422, detail=str(ve))
    except RuntimeError as re:
        logger.error("Prediction error", error=str(re))
        raise HTTPException(status_code=500, detail=str(re))
    except Exception as e:
        logger.error("Unexpected error", error=str(e))
        raise HTTPException(status_code=500, detail="Unexpected error")

if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port)
''',

"config/settings.py": '''\
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    model_path: str = os.getenv("MODEL_PATH", "models/credit_scoring_nn.onnx")
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    class Config:
        env_file = ".env"

def get_settings():
    return Settings()
'''
}

# ---------- 2. Создаём файлы ----------
for path, content in FILES.items():
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(content, encoding='utf-8')
    print(f"✔  {path}")

print("\nГотово! Все файлы созданы.")