# app/routes/department_routes.py
from fastapi import APIRouter, Query, HTTPException
from typing import Optional, Dict, Any
from controllers.hr_dashboard_controllers.department_summary_controller import UserController

router = APIRouter(prefix="/departments", tags=["Departments"])

@router.get("/aggregate")
async def aggregate_departments(
    department: Optional[str] = Query(
        None, 
        description="Filter by specific department name (last item in department array)"
    ),
    hr_id: Optional[str] = Query(
        None,
        description="Filter by HR ID to get departments under specific HR"
    )
) -> Dict[str, Any]:
    """
    Aggregate users by department.
    
    - **department**: Filter by specific department name (optional)
    - **hr_id**: Filter by HR ID (optional)
    
    Returns aggregation of employees in each department.
    """
    try:
        controller = UserController()
        result = await controller.get_department_aggregation(department, hr_id)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )
