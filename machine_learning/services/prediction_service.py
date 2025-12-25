from typing import List
from schemas.prediction import StudentInput, PredictionOutput
from models.ml_model import MLModel
from utils.data_gen import get_current_utc_time

class PredictionService:
    def __init__(self):
        self.model = MLModel(model_type="RandomForest")

    def predict_students(self, students: List[StudentInput]) -> List[PredictionOutput]:
        results = []
        for student in students:
            risk_score = self.model.predict_risk(student.features)
            prediction = self.model.get_prediction_label(risk_score)
            
            results.append(PredictionOutput(
                studentId=student.studentId,
                codePresentation=student.codePresentation,
                riskScore=risk_score,
                prediction=prediction,
                modelType=self.model.model_type,
                createdAt=get_current_utc_time()
            ))
        return results
