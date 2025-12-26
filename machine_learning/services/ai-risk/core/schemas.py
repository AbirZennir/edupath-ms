from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class PredictRiskRequest(BaseModel):
    features: List[float] = Field(..., description="Vector of numeric features")


class ExplainItem(BaseModel):
    feature: str
    impact: float


class ExplainRiskResponse(BaseModel):
    model: str = "mlp"
    risk_score: float
    prediction: int
    top_features: List[ExplainItem]


class RecommendRiskResponse(BaseModel):
    model: str = "rules"
    risk_score: Optional[float] = None
    prediction: Optional[int] = None
    recommendations: List[str]
