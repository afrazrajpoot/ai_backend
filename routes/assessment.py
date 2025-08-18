# routes/assessment.py
from fastapi import APIRouter, HTTPException, Query
from controllers.assessment import AssessmentController
from schemas.assessment import AssessInput

router = APIRouter(prefix="/assessments", tags=["assessments"])

@router.post("/employee-assessment")
async def assess_employee(input_data: AssessInput):
    return await AssessmentController.assess_employee(input_data)

@router.get("/")
async def get_assessments(userId: str = Query(..., description="User ID")):
    return await AssessmentController.get_assessments(userId)