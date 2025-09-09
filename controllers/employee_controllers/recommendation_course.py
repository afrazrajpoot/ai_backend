# controllers/employee_controllers/recommendation_course.py
from services.employee_services.recommendation_course import EmployeeService
from utils.models import EmployeeLearningResponse, ProgressTracking, RecommendedCourse, Skill
from prisma import Prisma
from typing import List

class EmployeeController:
    def __init__(self, prisma: Prisma):
        self.service = EmployeeService(prisma)  # Pass Prisma client

    async def get_employee_learning_dashboard(self, user_id: str) -> EmployeeLearningResponse:
        try:
            employee_data = await self.service.get_employee_data(user_id)
            if not employee_data:
                raise Exception("Employee not found")
            
            employee = employee_data["employee"]
            skills = employee.skills or []
            current_skills = [
                Skill(name=skill["name"], proficiency=skill["proficiency"])
                for skill in skills
                if isinstance(skill, dict) and "name" in skill and "proficiency" in skill
            ]
            
            return EmployeeLearningResponse(
                employee_id=employee.id,
                employee_name=f"{employee.firstName} {employee.lastName}",
                current_skills=current_skills,
                recommended_courses=await self.service.get_course_recommendations(employee_data),
                progress_tracking=self.service.get_progress_tracking(employee_data)
            )
        except Exception as e:
            raise Exception(f"Error processing request: {str(e)}")