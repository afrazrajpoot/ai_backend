import os
from prisma.models import Employee, User
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
from typing import List
from utils.models import ProgressTracking, RecommendedCourse
from prisma import Prisma
from langchain_community.tools.tavily_search import TavilySearchResults

class EmployeeService:
    def __init__(self, prisma: Prisma):
        self.prisma = prisma  # Use provided Prisma client
        self.llm = ChatOpenAI(
            temperature=0.7,
            model_name="gpt-3.5-turbo",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        self.search_tool = TavilySearchResults(
            max_results=3,
            api_key=os.getenv("TAVILY_API_KEY")
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
      
        try:
            # Use LLM to generate relevant course topics based on profile
            prompt = f"""
            Based on this employee profile, suggest 3-5 relevant course topics or titles for skill development:
            {json.dumps(profile, indent=2)}
            Focus on courses from Udemy or Coursera only.
            Return JSON: {{"topics": ["Topic 1", "Topic 2", ...]}}
            """
            response = await self.llm.agenerate([[
                SystemMessage(content=self.system_prompt),
                HumanMessage(content=prompt)
            ]])
            result = json.loads(response.generations[0][0].text)
            topics = result["topics"]

            # Search for courses using Tavily for each topic
            courses = []
            for topic in topics:
                search_query = f'"{topic}" course site:udemy.com OR site:coursera.org'
                search_results = await self._search_courses(search_query)
                if search_results:
                    # Select the first relevant result
                    best_result = search_results[0]
                    course = RecommendedCourse(
                        title=best_result.get("title", topic),
                        provider="Udemy" if "udemy.com" in best_result.get("url", "") else "Coursera",
                        url=best_result.get("url", ""),
                        reason=f"Recommended for {topic} based on profile"
                    )
                    courses.append(course)

            # If fewer than 3 courses found, add defaults
            if len(courses) < 3:
                default_courses = [
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
                        provider="Coursera",
                        url="https://www.coursera.org/learn/communication-skills",
                        reason="Improves career skills"
                    )
                ]
                courses.extend(default_courses[:3 - len(courses)])

            return courses[:5]  # Limit to 5

        except Exception as e:
            print(f"Course recommendation failed: {e}")
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
                    provider="Coursera",
                    url="https://www.coursera.org/learn/communication-skills",
                    reason="Improves career skills"
                )
            ]

    async def _search_courses(self, query: str) -> List[dict]:
        try:
            # Use TavilySearchResults tool to get search results
            results = self.search_tool.run(query)
            # Parse the results - assuming it returns a list of dicts with title, url, etc.
            if isinstance(results, str):
                # If returns JSON string, parse it
                import json
                parsed_results = json.loads(results)
            else:
                parsed_results = results
            
            search_results = []
            for r in parsed_results:
                if isinstance(r, dict) and "url" in r:
                    # Filter for relevant domains
                    url = r["url"]
                    if "udemy.com" in url or "coursera.org" in url:
                        search_results.append({
                            "title": r.get("title", ""),
                            "url": url
                        })
            
            return search_results
        except Exception as e:
            print(f"Tavily search failed: {e}")
            return []