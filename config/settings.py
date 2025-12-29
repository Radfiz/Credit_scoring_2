# config/settings.py (альтернатива)
from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # Путь к ONNX-модели
    model_path: str = os.getenv("MODEL_PATH", "models/trained/onnx/credit_scoring_model.onnx")
    # Путь к эталонным признакам для Evidently
    reference_data_path: str = os.getenv("REFERENCE_DATA_PATH", "data/processed/reference_features.csv") # Добавлено

    api_host: str = "0.0.0.0"
    api_port: int = 8000
    debug: bool = False

    class Config:
        env_file = ".env" # Если используется .env файл

def get_settings():
    return Settings()