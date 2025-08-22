from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel
from controllers.employee_dashboard import DashboardController

router = APIRouter(prefix="/employee_dashboard", tags=["employee_dashboard"])

class RecommendationRequest(BaseModel):
    employee: dict
    companies: list[dict]

@router.get("/")
async def get_dashboard(userId: str = Query(..., description="User ID")):
    return await DashboardController.get_dashboard(userId)

@router.post("/recommend-companies")
async def recommend_companies(request: RecommendationRequest):
    try:
        return await DashboardController.recommend(request)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))