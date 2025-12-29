# api/services/drift_detection.py

import pandas as pd
import numpy as np
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import DataDriftTable
from pathlib import Path
import structlog
import joblib # Если нужно загружать препроцессор для преобразования новых данных

logger = structlog.get_logger()

class DriftDetectionService:
    def __init__(self, reference_data_path: str):
        self.reference_data_path = reference_data_path
        self.reference_data = None
        self.load_reference_data()

    def load_reference_data(self):
        """Загружает эталонные признаки из CSV."""
        try:
            self.reference_data = pd.read_csv(self.reference_data_path)
            logger.info("Reference data loaded successfully", path=self.reference_data_path)
            logger.info("Reference data shape", shape=self.reference_data.shape)
        except Exception as e:
            logger.error("Failed to load reference data", error=str(e), path=self.reference_data_path)
            raise

    def check_data_drift(self, current_data: pd.DataFrame) -> dict:
        """
        Проверяет дрейф данных.
        current_data: DataFrame с признаками текущего запроса (или батча).
        Возвращает словарь с результатами проверки.
        """
        try:
            if not current_data.columns.equals(self.reference_data.columns):
                logger.warning("Current data columns do not match reference data columns. This might lead to incorrect drift calculation.")

            # Создаем отчет Evidently
            report = Report(metrics=[DataDriftTable()])
            report.run(
                reference_data=self.reference_data,
                current_data=current_data,
                # column_mapping=ColumnMapping() # Можно указать, если данные сложные
            )

            # Получаем результаты
            report_results = report.as_dict()

            # Извлекаем информацию о дрейфе
            drift_table_result = report_results['metrics'][0]['result']['drift_by_columns']
            dataset_drift = report_results['metrics'][0]['result']['dataset_drift']

            # Определяем порог для дрейфа (например, p-value < 0.05)
            drift_detected = dataset_drift # Это True/False для всего датасета
            drift_details = {}
            for col, details in drift_table_result.items():
                drift_details[col] = {
                    'drift_detected': details['drift_detected'],
                    'p_value': details['p_value'],
                    'stattest_name': details['stattest_name']
                }

            logger.info("Drift check completed", dataset_drift=dataset_drift)
            return {
                'dataset_drift_detected': drift_detected,
                'drift_details_by_column': drift_details
            }

        except Exception as e:
            logger.error("Drift check failed", error=str(e))
            raise RuntimeError(f"Drift check failed: {str(e)}")
