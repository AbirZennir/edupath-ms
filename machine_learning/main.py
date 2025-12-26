from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from schemas.prediction import StudentInput, PredictionOutput
from schemas.stats import DashboardAnalyticsResponse, AtRiskStudent
from services.prediction_service import PredictionService
from services.analysis_service import AnalysisService

from services.recommendation_service import RecommendationService
from schemas.recommendation import RecommendationRequest, RecommendationResponse

# Initialize App
app = FastAPI(
    title="FastAPI ML Service Bridge",
    description="Microservice for ML predictions and dashboard analytics",
    version="1.0.0"
)

# Initialize Services
prediction_service = PredictionService()
analysis_service = AnalysisService()
recommendation_service = RecommendationService()

# Endpoints

@app.get("/ai/at-risk-students", response_model=List[AtRiskStudent])
async def get_at_risk_students():
    """
    Get list of at-risk students for the teacher dashboard.
    """
    try:
        return analysis_service.get_at_risk_students_public()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/at-risk-students", response_model=List[PredictionOutput])
async def predict_at_risk_students(students: List[StudentInput]):
    """
    Predict at-risk students using the ML model.
    """
    try:
        return prediction_service.predict_students(students)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/dashboard/analytics", response_model=DashboardAnalyticsResponse)
async def get_dashboard_analytics():
    """
    Return dashboard analytics for the frontend.
    """
    try:
        return analysis_service.get_dashboard_analytics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ai/recommendations", response_model=RecommendationResponse)
async def generate_recommendations(request: RecommendationRequest):
    """
    Generate personalized recommendations based on student risk profile.
    """
    try:
        return recommendation_service.generate_recommendations(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/ai/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
