from pydantic import BaseModel
from typing import List, Optional

class Stats(BaseModel):
    totalStudents: int
    studentsAtRisk: int
    successRate: int
    engagementAvg: int
    resourcesConsulted: int

class AtRiskStudent(BaseModel):
    id: int
    nom: str
    classe: str
    modules: str
    risque: int
    derniereConnexion: str
    niveau: str
    avatar: str = "A" # Default avatar

class ProfileDistributionItem(BaseModel):
    name: str
    value: int
    color: str

class EvolutionItem(BaseModel):
    week: str
    success: int
    engagement: int

class ModuleSuccessItem(BaseModel):
    module: str
    taux: int

class DashboardAnalyticsResponse(BaseModel):
    stats: Stats
    atRiskStudents: List[AtRiskStudent]
    atRiskStudents: List[AtRiskStudent]
    profileDistribution: List[ProfileDistributionItem]
    evolution: List[EvolutionItem]
    moduleSuccess: List[ModuleSuccessItem] = []
