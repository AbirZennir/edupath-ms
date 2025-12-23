import random
from typing import Dict, Any

class MLModel:
    def __init__(self, model_type: str = "RandomForest"):
        self.model_type = model_type

    def predict_risk(self, features: Dict[str, Any]) -> float:
        """
        Predict risk based on student features.
        Uses a heuristic rule-based approach to ensure deterministic output derived from real data.
        """
        score = 0.2  # Base risk

        # 1. Previous Attempts (High impact)
        # previous attempts usually correlate with struggle
        prev_attempts = int(features.get("num_of_prev_attempts", 0))
        score += prev_attempts * 0.15

        # 2. Disability (Minor impact in some contexts, but present in OULAD)
        if features.get("disability") == "Y":
            score += 0.1

        # 3. IMD Band (Socio-economic factor)
        imd = features.get("imd_band", "")
        if imd in ["0-10%", "10-20%"]:
             score += 0.15
        elif imd in ["20-30%", "30-40%"]:
             score += 0.05

        # 4. Education (lower education -> slightly higher base risk)
        education = features.get("highest_education", "")
        if "No Formal quals" in education or "Lower Than A Level" in education:
            score += 0.1

        # Clamp score between 0 and 1
        return min(max(score, 0.0), 1.0)

    def get_prediction_label(self, risk_score: float) -> int:
        return 1 if risk_score >= 0.5 else 0
