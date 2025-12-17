import os
import json
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from prisma import Prisma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import our new tools
from .job_tools import InternalJobFetcher, ExternalJobFetcher

class JobRecommendationService:
    """
    Intelligent recommendation service that orchestrates:
    1. Understanding the user (skills, position)
    2. Fetching candidates (Internal & External)
    3. Intelligent Scoring (LLM-based)
    """
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        
        # Initialize our tools
        self.internal_fetcher = InternalJobFetcher(self.embeddings)
        self.external_fetcher = ExternalJobFetcher()
        
        # Prompt for scoring jobs against user profile
        self.scoring_prompt = PromptTemplate(
            input_variables=["user_skills", "user_positions", "jobs_list"],
            template="""Score each job (0-100) based on match with user profile.

USER PROFILE:
Skills: {user_skills}
Positions: {user_positions}

JOBS:
{jobs_list}

CRITERIA:
1. Skill Match (40pts)
2. Role Relevance (30pts)
3. Industry Fit (20pts)
4. Experience Level (10pts)

Return ONLY a JSON object: {{"Job Title": 85, ...}}
"""
        )
        self.scoring_chain = LLMChain(llm=self.llm, prompt=self.scoring_prompt)

    async def _fetch_and_parse_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user data and parse JSON fields."""
        db = Prisma()
        await db.connect()
        try:
            user = await db.user.find_unique(
                where={"id": user_id}, 
                include={"employee": True}
            )
            if not user: return None
            
            return {
                'skills': self._parse_json_field(getattr(user.employee, 'skills', []) if user.employee else []),
                'position': self._parse_json_field(getattr(user, 'position', []))
            }
        finally:
            await db.disconnect()

    def _parse_json_field(self, field):
        """Helper to parse JSON string fields."""
        if isinstance(field, str):
            try: return json.loads(field)
            except: return []
        return field or []

    def _extract_user_skills_and_positions(self, employee_info: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Extract clean lists of skills and positions from user data."""
        raw_skills = employee_info.get('skills', [])
        raw_positions = employee_info.get('position', [])
        
        skills = []
        for s in raw_skills:
            if isinstance(s, dict): skills.append(s.get('name', ''))
            else: skills.append(str(s))
            
        positions = []
        for p in raw_positions:
            if isinstance(p, dict): positions.append(p.get('name', ''))
            else: positions.append(str(p))
            
        # Clean and dedup
        skills = list(set([s.strip() for s in skills if s.strip()]))
        positions = list(set([p.strip() for p in positions if p.strip()]))
        
        # Add inferred terms
        if any("educat" in p.lower() for p in positions):
            positions.extend(["Instructional Designer", "E-Learning Developer"])
        
        return skills, positions

    def _extract_json_from_response(self, response_text: str) -> Dict[str, float]:
        """Parse scoring response."""
        try:
            match = re.search(r'\{.*\}', response_text.strip(), re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                return {k: float(v) for k, v in data.items()}
        except: pass
        return {}

    def _score_jobs_intelligently(self, user_skills: List[str], user_positions: List[str], job_docs: List[Document]) -> List[Dict[str, Any]]:
        """Score a list of job documents using LLM."""
        if not job_docs: return []
        
        # Prepare text for LLM
        jobs_text = "\n\n".join([
            f"Title: {d.metadata.get('title', 'Unknown')}\nDetails: {d.page_content[:300]}"
            for d in job_docs
        ])
        
        try:
            response = self.scoring_chain.run(
                user_skills=", ".join(user_skills[:10]),
                user_positions=", ".join(user_positions[:5]),
                jobs_list=jobs_text
            )
            scores = self._extract_json_from_response(response)
            
            results = []
            for d in job_docs:
                title = d.metadata.get('title', 'Unknown')
                # Use LLM score or fallback to basic overlap
                if title in scores:
                    score = scores[title]
                else:
                    # Fallback scoring
                    content = d.page_content.lower()
                    matches = sum(1 for s in user_skills if s.lower() in content)
                    score = min(matches * 15 + 50, 90)
                
                results.append({
                    'title': title,
                    'match_score': score,
                    'document': d
                })
            return results
            
        except Exception as e:
            print(f"Error scoring jobs: {e}")
            return [{'title': d.metadata.get('title'), 'match_score': 50.0, 'document': d} for d in job_docs]

    async def get_internal_jobs(self, user_id: str, recruiter_id: str) -> List[Dict[str, Any]]:
        """Get and score internal jobs."""
        print(f"JobRecommendationService: Getting INTERNAL jobs for {user_id}")
        
        user_data = await self._fetch_and_parse_user(user_id)
        if not user_data: return []
        
        skills, positions = self._extract_user_skills_and_positions(user_data)
        if not skills and not positions: return []
        
        # 1. Fetch Candidates
        query = f"{' '.join(positions[:2])} {' '.join(skills[:3])}"
        candidates = await self.internal_fetcher.fetch_jobs(query, recruiter_id)
        
        if not candidates: return []
        
        # 2. Score Candidates
        docs = [c['vector_doc'] for c in candidates]
        scored_results = self._score_jobs_intelligently(skills, positions, docs)
        
        # 3. Format Response
        final_jobs = []
        for candidate in candidates:
            db_job = candidate['db_job']
            # Find the score for this job
            score_item = next((s for s in scored_results if s['title'] == db_job.title), None)
            score = score_item['match_score'] if score_item else candidate['vector_score']
            
            final_jobs.append({
                "id": db_job.id,
                "title": db_job.title or "Unknown",
                "description": db_job.description or "",
                "location": db_job.location or "Not specified",
                "type": db_job.type or "Not specified",
                "salary": db_job.salary or "Not specified",
                "match_score": score,
                "recruiter": {
                    "id": db_job.recruiter.id,
                    "firstName": db_job.recruiter.firstName,
                    "lastName": db_job.recruiter.lastName
                } if db_job.recruiter else None,
                "is_external": False,
                "source": "internal"
            })
            
        return final_jobs

    async def get_external_jobs(self, user_id: str) -> List[Dict[str, Any]]:
        """Get and score external jobs."""
        print(f"JobRecommendationService: Getting EXTERNAL jobs for {user_id}")
        
        user_data = await self._fetch_and_parse_user(user_id)
        if not user_data: return []
        
        skills, positions = self._extract_user_skills_and_positions(user_data)
        if not skills: return []
        
        # 1. Fetch Candidates
        external_jobs = await self.external_fetcher.fetch_external_jobs(skills, positions)
        if not external_jobs: return []
        
        # 2. Prepare for Scoring
        docs = []
        for job in external_jobs:
            content = f"Title: {job['title']}\nDesc: {job['description']}\nSkills: {job['required_skills']}"
            docs.append(Document(page_content=content, metadata={"title": job['title']}))
            
        # 3. Score
        scored_results = self._score_jobs_intelligently(skills, positions, docs)
        
        # 4. Update Scores
        for job in external_jobs:
            score_item = next((s for s in scored_results if s['title'] == job['title']), None)
            if score_item:
                job['match_score'] = score_item['match_score']
                
        return external_jobs

    async def recommend_jobs(self, user_id: str, recruiter_id: str, include_external: bool = True) -> List[Dict[str, Any]]:
        """Get recommendations from all sources."""
        print(f"JobRecommendationService: Generating recommendations...")
        
        internal = await self.get_internal_jobs(user_id, recruiter_id)
        
        external = []
        if include_external:
            external = await self.get_external_jobs(user_id)
            
        # Combine and sort by score
        all_jobs = internal + external
        all_jobs.sort(key=lambda x: x.get('match_score', 0), reverse=True)
        
        return all_jobs