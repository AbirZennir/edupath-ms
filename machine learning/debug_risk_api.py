import sys
from database import SessionLocal
from services.analysis_service import AnalysisService
import json

def test_at_risk_data():
    service = AnalysisService()
    try:
        results = service.get_dashboard_analytics()
        print("Data fetched successfully.")
        
        print("\nModule Success Rates:")
        for m in results.moduleSuccess:
            print(f"- {m.module}: {m.taux}%")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_at_risk_data()
