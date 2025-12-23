from pydantic import BaseModel
from typing import List, Optional

class Stats(BaseModel):
    totalStudents: int
    studentsAtRisk: int
    successRate: int
    engagementAvg: int
    resourcesConsulted: int

class AtRiskStudent(BaseModel):
    idStudent: int
    name: str
    className: str
    riskScore: int
    lastConnection: str
    status: str

class ProfileDistributionItem(BaseModel):
    name: str
    value: int
    color: str

class EvolutionItem(BaseModel):
    week: str
    success: int
    engagement: int

class DashboardAnalyticsResponse(BaseModel):
    stats: Stats
    atRiskStudents: List[AtRiskStudent]
    profileDistribution: List[ProfileDistributionItem]
    evolution: List[EvolutionItem]
