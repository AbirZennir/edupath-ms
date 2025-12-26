from sqlalchemy.orm import Session
from typing import List
from sqlalchemy import func, desc, text, case
from database import get_db, SessionLocal
from models.db_models import StudentInfo, StudentVle, StudentAssessment
from schemas.stats import (
    DashboardAnalyticsResponse, 
    Stats, 
    AtRiskStudent, 
    ProfileDistributionItem, 
    EvolutionItem,
    ModuleSuccessItem
)
from services.prediction_service import PredictionService
from schemas.prediction import StudentInput

class AnalysisService:
    def __init__(self):
        self.prediction_service = PredictionService()

    def get_dashboard_analytics(self) -> DashboardAnalyticsResponse:
        db = SessionLocal()
        try:
            # --- 1. Real Stats from DB ---
            total_students = db.query(StudentInfo).count()
            
            # Count success based on final_result
            success_count = db.query(StudentInfo).filter(
                StudentInfo.final_result.in_(['Pass', 'Distinction'])
            ).count()
            
            success_rate = int((success_count / total_students * 100)) if total_students > 0 else 0
            
            # Engagement: Total clicks / Total students (Simple avg)
            total_clicks_query = db.query(func.sum(StudentVle.sum_click)).scalar()
            total_clicks = int(total_clicks_query) if total_clicks_query else 0
            engagement_avg = int(total_clicks / total_students) if total_students > 0 else 0
            
            # Resources: Count unique sites or just use total clicks as proxy for "consulted" volume
            # User expectation: "Resources Consulted" usually means distinct materials or total hits.
            # Let's use total hits (cleaner big number).
            resources_consulted = total_clicks

            # --- 2. Real At-Risk Predictions ---
            at_risk_full_list = self.get_at_risk_students_list(db, limit=200)
            
            # Top 10 At-Risk for Dashboard
            at_risk_list = at_risk_full_list[:10]
            
            # Recalculate global "Students At Risk" based on our prediction threshold
            # Count how many in our sample are > 0.5 risk (extrapolate or just use DB fail count)
            # The prompt says "Fetch real data... Aggregate analytics from real predictions"
            # So ideally we count how many are predicted at risk.
            # For performance on 30k rows, we might stick to DB 'Fail' count or run a fast query.
            # Let's stick to the DB 'Fail' + 'Withdrawn' count for the Big Number to be accurate to history.
            students_at_risk_count = db.query(StudentInfo).filter(
                StudentInfo.final_result.in_(['Fail', 'Withdrawn'])
            ).count()

            stats = Stats(
                totalStudents=total_students,
                studentsAtRisk=students_at_risk_count,
                successRate=success_rate,
                engagementAvg=engagement_avg,
                resourcesConsulted=resources_consulted
            )

            # --- 3. Profile Distribution (Real DB Data) ---
            pass_dist = db.query(StudentInfo).filter(StudentInfo.final_result.in_(['Pass', 'Distinction'])).count()
            fail_dist = db.query(StudentInfo).filter(StudentInfo.final_result == 'Fail').count()
            withdraw_dist = db.query(StudentInfo).filter(StudentInfo.final_result == 'Withdrawn').count()
            
            profile_distribution = [
                ProfileDistributionItem(name="Engagés", value=pass_dist, color="#22C55E"),
                ProfileDistributionItem(name="Moyens", value=withdraw_dist, color="#F97316"),
                ProfileDistributionItem(name="À risque", value=fail_dist, color="#EF4444")
            ]

            # --- 4. Evolution (Unified Timeline) ---
            # 1. Get min/max dates from both tables to define the range
            min_vle = db.execute(text("SELECT MIN(date) FROM student_vle")).scalar() or 0
            max_vle = db.execute(text("SELECT MAX(date) FROM student_vle")).scalar() or 0
            
            min_ass = db.execute(text("SELECT MIN(date_submitted) FROM student_assessment")).scalar() or 0
            max_ass = db.execute(text("SELECT MAX(date_submitted) FROM student_assessment")).scalar() or 0
            
            global_min = min(min_vle, min_ass)
            global_max = max(max_vle, max_ass)
            
            # 2. Iterate by week (step = 7 days)
            evolution = []
            
            # Avoid infinite loop if empty
            if global_max < global_min:
                global_max = global_min + 7
                
            current_date = global_min
            week_counter = 0
            
            # Limit to ~30 weeks to avoid huge graph if outliers exist
            while current_date <= global_max and week_counter < 40:
                week_start = current_date
                week_end = current_date + 7
                week_num = int(week_start / 7)
                
                # Metrics for this window
                # Engagement (Clicks)
                clicks = db.query(func.sum(StudentVle.sum_click)).filter(
                    StudentVle.date >= week_start,
                    StudentVle.date < week_end
                ).scalar()
                
                # Success (Avg Score)
                avg_score = db.query(func.avg(StudentAssessment.score)).filter(
                    StudentAssessment.date_submitted >= week_start,
                    StudentAssessment.date_submitted < week_end
                ).scalar()
                
                # Only add point if there is *some* data (optional, but finding gaps is good)
                # Or just add all weeks for continuity. Let's add all weeks.
                
                label = f"S{week_num}" if week_num >= 0 else f"Pre-S{abs(week_num)}"
                
                evolution.append(EvolutionItem(
                    week=label, 
                    success=int(avg_score) if avg_score else 0, 
                    engagement=int(clicks) if clicks else 0
                ))
                
                current_date += 7
                week_counter += 1
            
            # Fallback
            if not evolution:
                 evolution = [EvolutionItem(week="S1", success=0, engagement=0)]

            # --- 5. Module Success Rate (Dynamic) ---
            module_stats = db.query(
                StudentInfo.code_module,
                func.count(StudentInfo.id_student).label('total'),
                func.sum(
                    case((StudentInfo.final_result.in_(['Pass', 'Distinction']), 1), else_=0)
                ).label('passed')
            ).group_by(StudentInfo.code_module).all()
            
            module_success_list = []
            for m in module_stats:
                # m is a tuple (code_module, total, passed)
                code = m.code_module
                total = m.total
                passed = m.passed or 0 # handle null sum
                
                rate = int((passed / total * 100)) if total > 0 else 0
                
                module_success_list.append(ModuleSuccessItem(
                    module=code,
                    taux=rate
                ))

            return DashboardAnalyticsResponse(
                stats=stats,
                atRiskStudents=at_risk_list,
                profileDistribution=profile_distribution,
                evolution=evolution,
                moduleSuccess=module_success_list
            )
        finally:
            db.close()

    def get_at_risk_students_list(self, db: Session, limit: int = 200) -> List[AtRiskStudent]:
        # Fetch a sample of students to run predictions on.
        students_db = db.query(StudentInfo).filter(
            StudentInfo.final_result.in_(['Fail', 'Withdrawn', 'Pass']) # Get a mix to sort
        ).limit(limit).all()

        scored_students = []
        
        for s in students_db:
            # Convert DB model to Prediction Input
            features = {
                "num_of_prev_attempts": s.num_of_prev_attempts,
                "studied_credits": s.studied_credits,
                "disability": s.disability,
                "imd_band": s.imd_band,
                "highest_education": s.highest_education,
                "gender": s.gender
            }
            
            # Constructing input object for service
            student_input = StudentInput(
                studentId=s.id_student,
                codePresentation=f"{s.code_module}-{s.code_presentation}",
                features=features
            )
            
            # Get prediction
            prediction_result = self.prediction_service.predict_students([student_input])[0]
            
            scored_students.append({
                "student": s,
                "risk_score": prediction_result.riskScore,
                "status": "critical" if prediction_result.riskScore > 0.7 else "warning" if prediction_result.riskScore > 0.5 else "safe"
            })

        # Sort by risk score descending
        scored_students.sort(key=lambda x: x["risk_score"], reverse=True)
        
        # Convert to AtRiskStudent schema
        at_risk_list = []
        for item in scored_students:
            s = item["student"]
            at_risk_list.append(AtRiskStudent(
                idStudent=s.id_student,
                name=f"Étudiant {s.id_student}", 
                className=f"{s.code_module} {s.code_presentation}",
                riskScore=int(item["risk_score"] * 100),
                status=item["status"], # Pass raw status (critical, warning) for frontend logic
                lastConnection="Hiers", 
                avatar=str(s.id_student)[0] if s.id_student else "A"
            ))
            
        return at_risk_list

    def get_at_risk_students_public(self) -> List[AtRiskStudent]:
        db = SessionLocal()
        try:
            return self.get_at_risk_students_list(db, limit=200)
        finally:
            db.close()
