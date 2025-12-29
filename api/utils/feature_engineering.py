import numpy as np
from typing import List

def validate_and_prepare_features(features: List[float], expected_length: int = 26) -> np.ndarray:
    if len(features) != expected_length:
        raise ValueError(f"Expected {expected_length} features, got {len(features)}")
    return np.array(features).reshape(1, -1)
