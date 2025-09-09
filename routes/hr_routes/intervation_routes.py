from fastapi import APIRouter, HTTPException
from typing import List
from utils.models import DepartmentInput, AnalysisResponse
from controllers.hr_dashboard_controllers.intervation_controller import AnalysisController

router = APIRouter(prefix="/api/analysis", tags=["analysis"])
controller = AnalysisController()

@router.post("/retention-risk", response_model=AnalysisResponse)
async def analyze_retention_risk(departments: List[DepartmentInput]):
    """
    Analyze retention risk and generate recommendations for departments
    """
    return await controller.analyze_retention_risk(departments)

@router.get("/health")
async def health_check():
    return {"status": "healthy", "service": "retention-analysis"}