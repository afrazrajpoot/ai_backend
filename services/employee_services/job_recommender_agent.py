import os
import json
import asyncio
import re
import traceback
from typing import List, Dict, Optional, Any, Annotated, TypedDict
from operator import add
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser

from prisma import Prisma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langgraph.graph import StateGraph, END

# Import existing tools
from .job_tools import InternalJobFetcher, ExternalJobFetcher

# --- Pydantic Models for Structured Output ---
class JobRankingResult(BaseModel):
    id: str = Field(description="The original job ID")
    professional_score: int = Field(description="Match score from 0-100 based on profile alignment")
    alignment_reason: str = Field(description="Specific professional reason why this fits the user's profile")
    original_title: str = Field(description="The EXACT job title as it appears in the JOBS LIST. DO NOT CHANGE A SINGLE CHARACTER.")
    refined_company: str = Field(description="The exact company name (look carefully in desc if Co is Unknown)")
    refined_salary: Optional[str] = Field(description="The exact salary or range (infer from text if possible)")
    refined_location: str = Field(description="Specified location (be specific, e.g. Sunrise, FL)")
    refined_type: Optional[str] = Field(description="Job type like Full-time, Contract, or Remote")
    refined_description: str = Field(description="A clean, professional 2-3 sentence summary of the role")

class JobRankingList(BaseModel):
    rankings: List[JobRankingResult]

# --- State Definition ---
class AgentState(TypedDict):
    user_id: str
    recruiter_id: str
    # Context
    user_data: Dict[str, Any]
    assessment_data: Dict[str, Any]
    nearby_area: str
    exclude_ids: List[str]
    # Process
    planning_strategy: str
    search_queries: List[str]
    # Results
    internal_jobs: List[Dict[str, Any]]
    external_jobs: List[Dict[str, Any]]
    combined_results: List[Dict[str, Any]]
    final_recommendations: List[Dict[str, Any]]
    # Metadata
    status: str
    error: Optional[str]

class JobRecommenderAgent:
    def __init__(self):
        try:
            self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0) # Use a faster model for planning
            self.embeddings = OpenAIEmbeddings()
            self.internal_fetcher = InternalJobFetcher(self.embeddings)
            self.external_fetcher = ExternalJobFetcher()
            
            # Build the graph
            self.workflow = self._create_workflow()
            self.app = self.workflow.compile()
        except Exception as e:
            print(f"[AGENT ERROR] Initialization failed: {e}")
            traceback.print_exc()

    def _create_workflow(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # Define Nodes
        workflow.add_node("fetch_context", self.fetch_context)
        workflow.add_node("planner", self.planner)
        workflow.add_node("job_search", self.job_search)
        workflow.add_node("ranking_and_parsing", self.ranking_and_parsing)

        # Define Edges
        workflow.set_entry_point("fetch_context")
        workflow.add_edge("fetch_context", "planner")
        workflow.add_edge("planner", "job_search")
        workflow.add_edge("job_search", "ranking_and_parsing")
        workflow.add_edge("ranking_and_parsing", END)

        return workflow

    # --- Nodes Implementation ---

    async def fetch_context(self, state: AgentState) -> Dict[str, Any]:
        """Fetch user profile and assessment data from database."""
        user_id = state["user_id"]
        print(f"\n[AGENT DEBUG] === Node: fetch_context (User: {user_id}) ===")
        
        db = Prisma()
        await db.connect()
        try:
            # 1. Fetch User and Employee Profile
            user = await db.user.find_unique(
                where={"id": user_id},
                include={"employee": True}
            )
            
            if not user or not user.employee:
                print(f"[AGENT DEBUG] User or Employee profile not found for {user_id}")
                return {"error": "User or Employee profile not found", "status": "failed"}

            # 2. Fetch Assessment Data (Genius Factor Report)
            assessment = None
            try:
                assessment = await db.individualemployeereport.find_first(
                    where={"userId": user_id},
                    order={"createdAt": "desc"}
                )
            except Exception as ae:
                print(f"[AGENT DEBUG] Error accessing assessment model: {ae}")

            # Extract clean data
            user_data = {
                "firstName": user.firstName or "",
                "lastName": user.lastName or "",
                "position": user.position if user.position else [],
                "skills": self._get_clean_skills(user.employee.skills),
                "experience": self._parse_json(user.employee.experience),
                "education": self._parse_json(user.employee.education),
                "bio": user.employee.bio or ""
            }
            
            assessment_data = {}
            if assessment:
                assessment_data = {
                    "genius_factor_score": assessment.geniusFactorScore,
                    "genius_factor_profile": self._parse_json(assessment.geniusFactorProfileJson),
                    "role_alignment": self._parse_json(assessment.currentRoleAlignmentAnalysisJson),
                    "executive_summary": assessment.executiveSummary
                }

            print(f"[AGENT DEBUG] Context fetched successfully. Assessment: {'Yes' if assessment else 'No'}")
            print(f"[AGENT DEBUG] Nearby Area: {user.employee.nearbyArea or 'Global'}")
            print(f"[AGENT DEBUG] Cleaned Skills: {user_data['skills']}")

            return {
                "user_data": user_data,
                "assessment_data": assessment_data,
                "nearby_area": user.employee.nearbyArea or "Global",
                "status": "context_fetched"
            }
        except Exception as e:
            print(f"[AGENT DEBUG] Error in fetch_context: {e}")
            traceback.print_exc()
            return {"error": str(e), "status": "failed"}
        finally:
            await db.disconnect()

    async def planner(self, state: AgentState) -> Dict[str, Any]:
        """Analyze data and plan the job search strategy."""
        print(f"\n[AGENT DEBUG] === Node: planner ===")
        user_data = state.get("user_data", {})
        assessment_data = state.get("assessment_data", {})
        nearby_area = state.get("nearby_area", "Global")
        
        if state.get("status") == "failed":
            return {}

        prompt = PromptTemplate(
            input_variables=["user_profile", "location"],
            template="""You are an expert Career Strategist. Analyze the following user profile to act as a Job Search Engine.
            
            USER PROFILE:
            {user_profile}
            
            TARGET LOCATION (Nearby Area):
            {location}
            
            TASK:
            1. ANALYZE: Identify the user's primary professional role (e.g., "Frontend Developer", "Accountant", "DevOps Engineer") based on their skills.
            2. CONSOLIDATE: Instead of searching for individual skills (like 'React', 'HTML', 'JS'), create ONE "Master Query" that covers the role.
            3. CREATE QUERY: Generate 1 (and ONLY 1) highly optimized search query.
               - Format: "[Role Title] jobs in [Location]"
               - Determine the Role Title by looking at the strongest skill cluster.
               - If location is "{location}" and it equals "Global", use "Remote".
               - Otherwise, include the "{location}".
            
            FORMAT:
            Return a JSON object:
            {{
                "strategy": "Brief reasoning for the role title selection",
                "search_queries": ["Master Query"]
            }}
            """
        )
        
        try:
            chain = prompt | self.llm
            response = await chain.ainvoke({
                "user_profile": json.dumps(user_data),
                "location": nearby_area
            })
            
            content = response.content if hasattr(response, 'content') else str(response)
            plan = json.loads(self._extract_json(content))
            
            print(f"[AGENT DEBUG] Plan generated. Strategy: {plan.get('strategy', '')[:100]}...")
            print(f"[AGENT DEBUG] Queries: {plan.get('search_queries', [])}")

            return {
                "planning_strategy": plan.get("strategy", ""),
                "search_queries": plan.get("search_queries", []),
                "status": "planned"
            }
        except Exception as e:
            print(f"[AGENT DEBUG] Error in planner: {e}")
            # Fallback queries
            skills = ", ".join(user_data.get("skills", [])[:3])
            return {
                "planning_strategy": f"Fallback due to: {e}",
                "search_queries": [f"{skills} jobs in {nearby_area}"],
                "status": "planned_fallback"
            }

    async def job_search(self, state: AgentState) -> Dict[str, Any]:
        """Execute searches for jobs across internal and external sources."""
        print(f"\n[AGENT DEBUG] === Node: job_search ===")
        queries = state.get("search_queries", [])
        recruiter_id = state.get("recruiter_id", "")
        user_data = state.get("user_data", {})
        
        if state.get("status") == "failed":
            return {}

        skills = user_data.get("skills", [])
        
        try:
            # Parallel tasks
            tasks = [
                self.internal_fetcher.fetch_jobs(queries[0] if queries else "", recruiter_id),
                self.external_fetcher.fetch_with_queries(queries[:1], skills) # Strictly 1 query for maximum speed
            ]
            
            results = await asyncio.gather(*tasks)
            
            internal_raw = results[0]
            external_raw = results[1]
            
            # Format internal jobs to match external structure for easier ranking
            internal_formatted = []
            for item in internal_raw:
                job = item["db_job"]
                internal_formatted.append({
                    "id": job.id,
                    "title": job.title,
                    "description": job.description,
                    "company": "Internal",
                    "location": job.location,
                    "salary": str(job.salary or ""),
                    "url": f"/jobs/{job.id}",
                    "match_score": item["vector_score"],
                    "is_external": False,
                    "source": "internal"
                })

            print(f"[AGENT DEBUG] Search completed. Internal: {len(internal_formatted)}, External: {len(external_raw)}")

            return {
                "internal_jobs": internal_formatted,
                "external_jobs": external_raw,
                "combined_results": internal_formatted + external_raw,
                "status": "searched"
            }
        except Exception as e:
            print(f"[AGENT DEBUG] Error in job_search: {e}")
            traceback.print_exc()
            return {"error": str(e), "status": "failed"}

    async def ranking_and_parsing(self, state: AgentState) -> Dict[str, Any]:
        """Rank and parse jobs based on genius factor and assessment profile."""
        print(f"\n[AGENT DEBUG] === Node: ranking_and_parsing ===")
        jobs = state.get("combined_results", [])
        assessment = state.get("assessment_data", {})
        strategy = state.get("planning_strategy", "")
        exclude_ids = set(state.get("exclude_ids", []))
        
        # Filter out excluded jobs
        jobs = [j for j in jobs if j.get("id") not in exclude_ids]
        
        if state.get("status") == "failed" or not jobs:
            print(f"[AGENT DEBUG] Skipping ranking: status={state.get('status')}, jobs count={len(jobs)}")
            return {"final_recommendations": jobs, "status": "completed_no_results"}

        try:
            # OPTIMIZATION: Process only top 7 candidates to reduce latency
            candidates = jobs[:7]
            
            # Prepare minimal context for speed
            jobs_preview = "\n".join([
                f"ID: {j['id']} | Title: {j['title']} | Co: {j['company']} | Loc: {j['location']} | Text: {j['description'][:200]}"
                for j in candidates
            ])

            # Use a simpler, faster prompt without strict Pydantic parsing
            prompt = PromptTemplate(
                input_variables=["strategy", "user_profile", "jobs_list"],
                template="""Rank these jobs for the user.
                
                STRATEGY: {strategy}
                PROFILE: {user_profile}
                JOBS:
                {jobs_list}
                
                Task:
                1. Score each job (0-100) on Skill & Experience match.
                2. Return a valid JSON array of objects.
                
                Format:
                [
                    {{
                        "id": "job_id",
                        "match_score": 85,
                        "reason": "Why it fits",
                        "refined_title": "Clean Title",
                        "refined_company": "Company Name",
                        "refined_location": "Location",
                        "refined_salary": "Salary or N/A"
                    }}
                ]
                
                Return ONLY the JSON array.
                """
            )

            chain = prompt | self.llm
            response = await chain.ainvoke({
                "strategy": strategy,
                "user_profile": json.dumps(state.get("user_data", {})),
                "jobs_list": jobs_preview
            })

            content = response.content if hasattr(response, 'content') else str(response)
            clean_json = self._extract_json(content)
            rankings = json.loads(clean_json)
            
            rankings_map = {r.get("id"): r for r in rankings if isinstance(r, dict)}
            
            final_recommendations = []
            for j in candidates:
                if j["id"] in rankings_map:
                    ranking = rankings_map[j["id"]]
                    j["match_score"] = ranking.get("match_score", 0)
                    j["alignment_reason"] = ranking.get("reason", "Good match")
                    j["title"] = ranking.get("refined_title", j["title"])
                    j["company"] = ranking.get("refined_company", j["company"])
                    j["location"] = ranking.get("refined_location", j["location"])
                    if ranking.get("refined_salary"): j["salary"] = ranking.get("refined_salary")
                    
                    final_recommendations.append(j)
            
            # Sort by professional score
            final_recommendations.sort(key=lambda x: x["match_score"], reverse=True)
            
            print(f"[AGENT DEBUG] Ranking completed instantly. Top job: {final_recommendations[0]['title'] if final_recommendations else 'None'}")
            
            return {
                "final_recommendations": final_recommendations,
                "status": "completed"
            }
        except Exception as e:
            print(f"[AGENT DEBUG] Error in ranking: {e}")
            # Fallback
            jobs.sort(key=lambda x: x.get('match_score', 0), reverse=True)
            return {
                "final_recommendations": jobs[:5],
                "status": "completed_fallback"
            }

    # --- Helpers ---
    def _get_clean_skills(self, skills_field) -> List[str]:
        """Extract clean skill names from various formats."""
        raw_skills = self._parse_json(skills_field)
        if not raw_skills: return []
        
        clean = []
        for s in raw_skills:
            if isinstance(s, str):
                clean.append(s)
            elif isinstance(s, dict):
                # Handle {'name': 'skill', 'proficiency': 50}
                val = s.get("name") or s.get("skill") or s.get("value")
                if val: clean.append(str(val))
        return clean

    def _parse_json(self, field):
        if not field: return []
        if isinstance(field, str):
            try: return json.loads(field)
            except: return []
        return field

    def _extract_json(self, text: str) -> str:
        """Extract JSON from potential markdown response."""
        match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)
        if match:
            return match.group(1)
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            return match.group(0)
        return text

    # --- Public Access ---
    async def get_recommendations(self, user_id: str, recruiter_id: str, exclude_ids: List[str] = []) -> List[Dict[str, Any]]:
        print(f"\n[AGENT DEBUG] STARTING AGENT for User: {user_id}")
        initial_state = {
            "user_id": user_id,
            "recruiter_id": recruiter_id,
            "user_data": {},
            "assessment_data": {},
            "assessment_data": {},
            "nearby_area": "",
            "exclude_ids": exclude_ids,
            "planning_strategy": "",
            "search_queries": [],
            "internal_jobs": [],
            "external_jobs": [],
            "combined_results": [],
            "final_recommendations": [],
            "status": "started",
            "error": None
        }
        
        try:
            if not hasattr(self, 'app'):
                print("[AGENT ERROR] Agent 'app' not initialized. Check __init__ logs.")
                return []
            final_state = await self.app.ainvoke(initial_state)
            print(f"[AGENT DEBUG] AGENT FINISHED with status: {final_state.get('status')}")
            return final_state.get("final_recommendations", [])
        except Exception as e:
            print(f"[AGENT DEBUG] AGENT CRASHED: {e}")
            traceback.print_exc()
            raise e
