# employee_service.py
import os
from prisma.models import Employee, User
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
from typing import List
from utils.models import ProgressTracking, RecommendedCourse
from prisma import Prisma

class EmployeeService:
    def __init__(self, prisma: Prisma):
        self.prisma = prisma  # Use provided Prisma client
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.system_prompt = """You are a career development AI assistant. Provide 3-5 relevant online courses with real URLs from platforms like Coursera, Udemy, edX, or LinkedIn Learning."""

    async def get_employee_data(self, user_id: str) -> dict:
        try:
            user = await User.prisma(self.prisma).find_unique(
                where={"id": user_id},
                include={"employee": True}
            )
            if not user or not user.employee:
                raise Exception("User or employee not found")
            employee = await Employee.prisma(self.prisma).find_unique(
                where={"id": user.employeeId}
            )
            return {"user": user, "employee": employee}
        except Exception as e:
            raise Exception(f"Database error: {str(e)}")

    def get_progress_tracking(self, employee_data: dict) -> ProgressTracking:
        user = employee_data["user"]
        positions = user.position or []
        current_position = positions[-1] if positions else "Not specified"
        previous_position = positions[-2] if len(positions) > 1 else None
        departments = user.department or []
        current_department = departments[-1] if departments else "Not specified"
        previous_department = departments[-2] if len(departments) > 1 else None
        return ProgressTracking(
            current_position=current_position,
            previous_position=previous_position,
            current_department=current_department,
            previous_department=previous_department
        )

    async def get_course_recommendations(self, employee_data: dict) -> List[RecommendedCourse]:
        employee = employee_data["employee"]
        user = employee_data["user"]
        skills = employee.skills or []
        skill_names = [skill["name"] for skill in skills if isinstance(skill, dict) and "name" in skill]
        employee_profile = {
            "name": f"{employee.firstName} {employee.lastName}",
            "current_skills": skill_names,
            "current_position": user.position[-1] if user.position else "Not specified",
            "current_department": user.department[-1] if user.department else "Not specified",
        }
        return await self._get_course_recommendations(employee_profile)

    async def _get_course_recommendations(self, profile: dict) -> List[RecommendedCourse]:
        prompt = f"""
        Recommend 3-5 online courses for this employee based on their profile:
        {json.dumps(profile, indent=2)}
        Provide: title, provider (Coursera, Udemy, edX, etc.), actual URL, and reason for relevance.
        Return JSON: {{"courses": [{{"title": "Course Title", "provider": "Provider Name", "url": "https://course-url.com", "reason": "Why relevant"}}]}}
        """
        try:
            response = await self.llm.agenerate([[
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]])
            result = json.loads(response.generations[0][0].text)
            return [RecommendedCourse(**course) for course in result["courses"]]
        except Exception as e:
            print(f"AI recommendation failed: {e}")
            return [
                RecommendedCourse(
                    title="Python Programming",
                    provider="Coursera",
                    url="https://www.coursera.org/specializations/python",
                    reason="Foundation programming course"
                ),
                RecommendedCourse(
                    title="Data Structures and Algorithms",
                    provider="Udemy",
                    url="https://www.udemy.com/course/data-structures-and-algorithms-deep-dive-using-java/",
                    reason="Essential for developers"
                ),
                RecommendedCourse(
                    title="Communication Skills",
                    provider="LinkedIn Learning",
                    url="https://www.linkedin.com/learning/communication-skills-for-professionals",
                    reason="Improves career skills"
                )
            ]