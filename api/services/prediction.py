# api/services/prediction.py
import onnxruntime as rt
import numpy as np
from pathlib import Path
import structlog
from typing import List
import joblib # Если нужно загружать препроцессор
from api.utils.feature_engineering import validate_and_prepare_features

logger = structlog.get_logger()

class PredictionService:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = None
        self.preprocessor = None # Опционально, если нужен препроцессор
        self.load_model()
        # self.load_preprocessor() # Опционально

    def load_model(self):
        """Загружает ONNX-модель в ONNX Runtime сессию."""
        try:
            self.session = rt.InferenceSession(self.model_path)
            logger.info("ONNX model loaded successfully", model_path=self.model_path)
        except Exception as e:
            logger.error("Failed to load ONNX model", error=str(e))
            raise

    def load_preprocessor(self):
        """Загружает препроцессор (StandardScaler, OneHotEncoder и т.д.) если нужен."""
        preprocessor_path = Path(self.model_path).parent / "preprocessor.pkl" # Предполагаем, что рядом
        if preprocessor_path.exists():
            try:
                self.preprocessor = joblib.load(preprocessor_path)
                logger.info("Preprocessor loaded successfully", preprocessor_path=preprocessor_path)
            except Exception as e:
                logger.error("Failed to load preprocessor", error=str(e))
        else:
            logger.warning("Preprocessor not found, assuming features are already processed", preprocessor_path=preprocessor_path)

    def predict(self, features: List[float]) -> tuple[float, int]:
        """
        Выполняет инференс модели.
        Возвращает кортеж (вероятность_дефолта, предсказание_0_или_1).
        """
        try:
            # 1. Валидация и подготовка признаков
            input_data = validate_and_prepare_features(features)

            # 2. (Опционально) Применение препроцессора, если он был загружен
            # if self.preprocessor:
            #     # Это сложно, так как препроцессор ожидает pandas DataFrame или numpy array
            #     # с исходными названиями столбцов.
            #     # Для простоты, мы предполагаем, что признаки уже обработаны.
            #     pass

            # 3. Получение имени входного слоя (обычно 'input')
            input_name = self.session.get_inputs()[0].name

            # 4. Выполнение инференса
            result = self.session.run(None, {input_name: input_data.astype(np.float32)})
            # Результат: [logits], где logits имеет форму (1, 1)
            logits = result[0]
            # Применяем сигмоиду для получения вероятности
            import torch
            probability = torch.sigmoid(torch.tensor(logits)).item()

            # Применяем порог 0.5 для бинарного предсказания
            prediction = 1 if probability > 0.5 else 0

            logger.info("Prediction successful", features_len=len(features), probability=probability, prediction=prediction)
            return probability, prediction

        except ValueError as ve:
            logger.error("Feature validation error", error=str(ve))
            raise ve # Re-raise для обработки в роуте
        except Exception as e:
            logger.error("Prediction failed", error=str(e))
            raise RuntimeError(f"Prediction failed: {str(e)}") # Или другое подходящее исключение

# Глобальный экземпляр сервиса (или можно внедрять зависимость)
# prediction_service = PredictionService("models/credit_scoring_nn.onnx") # Путь будет зависеть от структуры Docker