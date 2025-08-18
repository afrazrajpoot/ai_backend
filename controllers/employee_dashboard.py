from services.db_service import DBService
from services.ai_service import AIService
from utils.logger import logger
from datetime import datetime
from fastapi import HTTPException
class DashboardController:
    @staticmethod
    async def get_dashboard(userId: str):
        try:
            assessments = await DBService.get_assessments_by_user(userId)
            if not assessments:
                return {
                    "recentAssessments": [],
                    "assessmentProgress": {"current": 0, "total": 68, "percentage": 0},
                    "completedAssessments": 0,
                    "averageScore": 0,
                    "careerMatches": 0,
                    "recentRecommendations": [],
                    "monthlyStats": [],
                    "aiRecommendation": "No assessment data available yet."
                }

            last_assessment = assessments[0] if assessments else None
            
            return {
                "recentAssessments": DashboardController._get_recent_assessments(assessments),
                "assessmentProgress": DashboardController._get_assessment_progress(assessments),
                "completedAssessments": len(assessments),
                "averageScore": DashboardController._get_average_score(assessments),
                "careerMatches": len(last_assessment['results']['strengths']) if last_assessment else 0,
                "recentRecommendations": DashboardController._get_recent_recommendations(last_assessment),
                "monthlyStats": DashboardController._get_monthly_stats(assessments),
                "aiRecommendation": await AIService.generate_career_recommendation(last_assessment)
            }
        except Exception as e:
            logger.error(f"Error in get_dashboard: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    def _get_recent_assessments(assessments):
        return [
            {
                "id": assess['id'],
                "name": "Career Assessment",
                "date": assess['createdAt'],
                "status": "Completed",
                "score": assess['overallScore'],
                "type": "Primary"
            }
            for assess in assessments[:3]
        ]

    @staticmethod
    def _get_assessment_progress(assessments):
        completed = len(assessments) * 68
        total = 340
        return {
            "current": min(completed, total),
            "total": total,
            "percentage": min((completed / total) * 100, 100) if total > 0 else 0
        }

    @staticmethod
    def _get_average_score(assessments):
        if not assessments:
            return 0
        return sum(assess['overallScore'] for assess in assessments) // len(assessments)

    @staticmethod
    def _get_recent_recommendations(assessment):
        if not assessment or not assessment.get('results'):
            return []
            
        return [
            {
                "title": f"{factor['name']} Role",
                "matchScore": factor['score'],
                "industry": {
                    "Tech Genius": "Technology",
                    "Number Genius": "Finance/Data",
                    "Creative Genius": "Design/Marketing",
                    "People Genius": "Human Resources",
                    "Word Genius": "Communications",
                    "Logic Genius": "Analytics/Consulting"
                }.get(factor['name'], "Various"),
                "link": "/career-pathways",
                "trending": factor['score'] > 80
            }
            for factor in assessment['results'].get('geniusFactors', [])
            if factor['score'] > 70
        ][:3]

    @staticmethod
    def _get_monthly_stats(assessments):
        monthly_stats = {}
        for assess in assessments:
            # Parse ISO format string to datetime if needed
            created_at = datetime.fromisoformat(assess['createdAt']) if isinstance(assess['createdAt'], str) else assess['createdAt']
            month = created_at.strftime("%b")
            monthly_stats[month] = monthly_stats.get(month, 0) + 1
        return [{"month": m, "completed": c} for m, c in sorted(monthly_stats.items())][-6:]