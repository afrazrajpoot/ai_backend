from fastapi import APIRouter
from controllers.assessment_analyze import AssessmentController
from schemas.assessment import AssessmentData

router = APIRouter(prefix="/analyze", tags=["Analyze"])

@router.post("/assessment")
async def analyze_route(assessment_data: AssessmentData):
   
    return await AssessmentController.analyze_assessment(assessment_data)
