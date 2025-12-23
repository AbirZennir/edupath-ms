from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from schemas.prediction import StudentInput, PredictionOutput
from schemas.stats import DashboardAnalyticsResponse
from services.prediction_service import PredictionService
from services.analysis_service import AnalysisService

# Initialize App
app = FastAPI(
    title="FastAPI ML Service Bridge",
    description="Microservice for ML predictions and dashboard analytics",
    version="1.0.0"
)

# CORS Configuration
origins = ["*"]  # Allow all origins as requested

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Services
prediction_service = PredictionService()
analysis_service = AnalysisService()

# Endpoints

@app.post("/at-risk-students", response_model=List[PredictionOutput])
async def predict_at_risk_students(students: List[StudentInput]):
    """
    Predict at-risk students using the ML model.
    """
    try:
        return prediction_service.predict_students(students)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard/analytics", response_model=DashboardAnalyticsResponse)
async def get_dashboard_analytics():
    """
    Return dashboard analytics for the frontend.
    """
    try:
        return analysis_service.get_dashboard_analytics()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """
    Health check endpoint.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
