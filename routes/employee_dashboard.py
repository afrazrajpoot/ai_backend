from fastapi import APIRouter, HTTPException, Query
from controllers.employee_dashboard import DashboardController

router = APIRouter(prefix="/employee_dashboard", tags=["employee_dashboard"])

@router.get("/")
async def get_dashboard(userId: str = Query(..., description="User ID")):
    return await DashboardController.get_dashboard(userId)