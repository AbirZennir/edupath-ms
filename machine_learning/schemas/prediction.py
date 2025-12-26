from pydantic import BaseModel
from typing import Dict, Any, List

class StudentInput(BaseModel):
    studentId: int
    codePresentation: str
    features: Dict[str, Any]

class PredictionOutput(BaseModel):
    studentId: int
    codePresentation: str
    riskScore: float
    prediction: int
    modelType: str
    createdAt: str
