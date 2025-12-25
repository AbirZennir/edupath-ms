from typing import List, Optional
from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    riskScore: float
    studentId: Optional[str] = None
    codeModule: Optional[str] = None
    codePresentation: Optional[str] = None

class RecommendationItem(BaseModel):
    id: str
    title: str
    description: str
    url: str
    type: str  # video, article, quiz
    duration: str
    difficulty: str
    priority: int  # 1 = High, 2 = Medium, 3 = Low

class RecommendationCategory(BaseModel):
    category: str
    icon: str  # Video, Article, Exercise
    color: str
    items: List[RecommendationItem]

class RecommendationResponse(BaseModel):
    studentProfile: str
    riskLevel: str
    categories: List[RecommendationCategory]
