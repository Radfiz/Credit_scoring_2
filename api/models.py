from pydantic import BaseModel
from typing import List

class CreditApplication(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    probability: float
    prediction: int
