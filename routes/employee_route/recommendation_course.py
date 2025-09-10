# routes/employee_route/recommendation_course.py
from fastapi import APIRouter, HTTPException, Query
from controllers.employee_controllers.recommendation_course import EmployeeController
from utils.models import EmployeeLearningResponse
from prisma import Prisma

router = APIRouter(prefix="/employees", tags=["employees"])

@router.get("/learning-dashboard", response_model=EmployeeLearningResponse)
async def get_employee_learning_dashboard(user_id: str = Query(..., description="Employee user ID")):
    try:
        prisma = Prisma()  # Initialize Prisma client
        await prisma.connect()  # Connect for this request
        try:
            controller = EmployeeController(prisma)
            return await controller.get_employee_learning_dashboard(user_id)
        finally:
            await prisma.disconnect()  # Disconnect after request
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
