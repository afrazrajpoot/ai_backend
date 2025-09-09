from fastapi import HTTPException
from typing import List
from utils.models import DepartmentInput, AnalysisResponse
from services.hr_dashboard_services.analysis_service import AnalysisService
class AnalysisController:
    def __init__(self):
        self.analysis_service = AnalysisService()

    async def analyze_retention_risk(self, departments: List[DepartmentInput]) -> AnalysisResponse:
        """Controller method for retention analysis"""
        try:
            return await self.analysis_service.analyze_departments(departments)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")