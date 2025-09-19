import os
from prisma.models import Employee, User
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
from typing import List
from utils.models import ProgressTracking, RecommendedCourse
from prisma import Prisma
from langchain_community.tools.tavily_search import TavilySearchResults
from config import settings

class EmployeeService:
    def __init__(self, prisma: Prisma):
        self.prisma = prisma  # Use provided Prisma client
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.search_tool = TavilySearchResults(api_key=settings.TAVILY_API_KEY, max_results=10)  # Increased max_results
        self.system_prompt = """You are a career development AI assistant. Provide relevant online courses with real URLs from platforms like Coursera, Udemy, edX, or LinkedIn Learning."""

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
        try:
            # Compose a detailed search query using employee profile
            skills = profile.get("current_skills", [])
            position = profile.get("current_position", "")
            department = profile.get("current_department", "")
            name = profile.get("name", "")

            # Build a single comprehensive search query for Tavily to get all relevant courses
            search_query = (
                f"best online courses for {position} professionals in {department} department "
                f"focusing on skills: {', '.join(skills)} "
                f"from Coursera, Udemy, edX, or LinkedIn Learning with direct URLs"
            )
        

            # Search for courses using Tavily with one query
            search_results = await self._search_courses(search_query)

            # Remove duplicates based on URL
            seen_urls = set()
            unique_results = []
            for result in search_results:
                url = result.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    unique_results.append(result)

            courses = []
            for result in unique_results:
                url = result.get("url", "")
                provider = (
                    "Udemy" if "udemy.com" in url else
                    "Coursera" if "coursera.org" in url else
                    "edX" if "edx.org" in url else
                    "LinkedIn Learning" if "linkedin.com/learning" in url else "Unknown"
                )
                course = RecommendedCourse(
                    title=result.get("title", ""),
                    provider=provider,
                    url=url,
                    reason=f"Recommended for {position} in {department} with skills {', '.join(skills)} based on web search"
                )
                courses.append(course)

      

            return courses[:5]  # Return up to 5 courses from the single query

        except Exception as e:
        
            # If search fails, return empty list; no static fallback
            return []

    async def _search_courses(self, query: str) -> List[dict]:
        try:
            # Debug: Verify API key
        

            results = await self.search_tool.arun(query)
        

            if isinstance(results, str):
             
                try:
                    parsed_results = json.loads(results)
                except json.JSONDecodeError as json_err:
                
                    return []
            else:
                parsed_results = results
           

            search_results = []
            for r in parsed_results:
                if isinstance(r, dict) and "url" in r:
                    url = r["url"]
                    if any(platform in url for platform in ["udemy.com", "coursera.org", "edx.org", "linkedin.com/learning"]):
                        search_results.append({
                            "title": r.get("title", ""),
                            "url": url
                        })
                else:
                    print(f"Skipping invalid result: {r}")

        
            return search_results
        except Exception as e:
         
            return []