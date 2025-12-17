import os
import json
import time
import re
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from pathlib import Path

from prisma import Prisma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Tavily for external searches (site-specific for Indeed, LinkedIn, Glassdoor)
from langchain_community.tools.tavily_search import TavilySearchResults

# FAISS directory path
def get_faiss_index_dir() -> str:
    index_dir = os.getenv("FAISS_INDEX_PATH", os.path.join(os.path.dirname(__file__), "faiss_jobs_index"))
    os.makedirs(index_dir, exist_ok=True)
    return index_dir

INDEX_DIR = get_faiss_index_dir()
TOP_K = int(os.getenv("JOBS_RETRIEVE_TOP_K", "10"))


@dataclass
class JobRow:
    id: str
    title: str
    description: Optional[str]
    recruiterId: str
    location: Optional[str] = None
    type: Optional[str] = None


class JobVectorStore:
    """
    Simplified FAISS store for internal jobs. Singleton for reuse.
    """
    _instance: Optional["JobVectorStore"] = None

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.vs: Optional[FAISS] = None
        self._loaded = False

    @classmethod
    def get(cls, embeddings: OpenAIEmbeddings) -> "JobVectorStore":
        if cls._instance is None:
            cls._instance = JobVectorStore(embeddings)
        return cls._instance

    def _job_to_document(self, job: JobRow) -> Document:
        content = (
            f"Title: {job.title}\n"
            f"Description: {job.description or ''}\n"
            f"Location: {job.location or ''}\n"
            f"Type: {job.type or ''}\n"
            f"RecruiterId: {job.recruiterId}"
        )
        return Document(
            page_content=content,
            metadata={
                "id": job.id,
                "title": job.title,
                "recruiterId": job.recruiterId,
            },
        )

    async def build_or_load(self, db: Prisma) -> None:
        if self._loaded and self.vs:
            return  # Already loaded

        Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

        jobs = await db.job.find_many()
        print(f"Total jobs in database: {len(jobs)}")
        
        docs: List[Document] = [
            self._job_to_document(JobRow(
                id=j.id, title=j.title, description=j.description,
                recruiterId=j.recruiterId, location=getattr(j, "location", None),
                type=getattr(j, "type", None)
            )) for j in jobs
        ]

        if docs:
            self.vs = FAISS.from_documents(docs, self.embeddings)
            self.vs.save_local(INDEX_DIR)
            print(f"FAISS index created with {len(docs)} documents")
        else:
            self.vs = FAISS.from_texts(["NO_JOBS"], self.embeddings)
            print("FAISS index created with placeholder (no jobs in DB)")

        self._loaded = True

    def retrieve_jobs_with_scores(self, query_text: str, recruiter_id: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        if not self.vs:
            print("FAISS vector store not initialized")
            return []
        
        print(f"Searching for jobs with query: '{query_text}' for recruiter: {recruiter_id}")
        
        try:
            results = self.vs.similarity_search_with_score(query_text, k=k)
            print(f"FAISS returned {len(results)} results")
            
            scored_jobs = []
            for doc, score in results:
                doc_recruiter = doc.metadata.get("recruiterId")
                
                if not recruiter_id or doc_recruiter == recruiter_id:
                    similarity_score = float((1 - min(float(score), 1.0)) * 100)
                    scored_jobs.append({
                        'title': doc.metadata.get('title', 'Unknown Title'),
                        'match_score': similarity_score,
                        'document': doc
                    })
            
            print(f"Total matching jobs after recruiter filter: {len(scored_jobs)}")
            return scored_jobs
            
        except Exception as e:
            print(f"Error in FAISS search: {e}")
            return []


class ExternalJobFetcher:
    """
    Improved tool for fetching external jobs with intelligent query generation.
    """
    def __init__(self):
        self.tavily = TavilySearchResults(max_results=8)  # Good balance
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        
        # Query generation prompt - DYNAMIC based on user skills
        self.query_gen_prompt = PromptTemplate(
            input_variables=["skills", "positions", "num_queries"],
            template="""Generate {num_queries} effective job search queries based on the user's skills and positions.

User Skills: {skills}
User Positions/Roles: {positions}

Generate search queries that:
1. Combine skills with relevant job roles
2. Use industry-standard job titles
3. Are specific enough to find relevant jobs
4. Include variations for different experience levels
5. Focus on the most relevant skill combinations

Return ONLY a JSON array of search query strings. No explanations.

Example format:
[
  "Instructional Designer Adobe Creative Suite remote jobs",
  "Educational Technology Specialist e-learning development",
  "Multimedia Learning Developer Photoshop Illustrator"
]

Search queries:"""
        )
        
        # Job extraction prompt
        self.extract_prompt = PromptTemplate(
            input_variables=["jobs_text", "target_skills"],
            template="""Extract job listings from the following text that match these target skills: {target_skills}

Input text:
{jobs_text}

Instructions:
1. Extract ONLY jobs that require at least one of these skills: {target_skills}
2. For each job, extract these fields:
   - title: Job title (be specific, use "Unknown" if unclear)
   - company: Company name (use "Unknown" if not found)
   - description: Job description (emphasize required skills)
   - location: Job location (use "Remote/On-site" if not specified)
   - type: Job type (Full-time, Part-time, Contract, Internship)
   - url: Job URL (use "#" if not found)
   - required_skills: List of skills mentioned in the job

3. Return ONLY a valid JSON array.
4. If no matching jobs, return empty array: []

Format: [{{"title": "...", "company": "...", "description": "...", "location": "...", "type": "...", "url": "...", "required_skills": ["skill1", "skill2"]}}]"""
        )

    def _extract_json_from_response(self, response_text: str) -> Optional[List[Dict]]:
        """Robust JSON extraction from LLM response."""
        if not response_text:
            return None
        
        cleaned = response_text.strip()
        
        # Try to find JSON array
        json_match = re.search(r'\[\s*\{.*\}\s*\]', cleaned, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        # Try direct parse
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            return []

    def _generate_search_queries(self, skills: List[str], positions: List[str]) -> List[str]:
        """Generate intelligent search queries based on user skills."""
        print(f"\nGenerating search queries from:")
        print(f"  Skills: {skills}")
        print(f"  Positions: {positions}")
        
        try:
            # Create query generation prompt
            query_response = self.llm.invoke(self.query_gen_prompt.format(
                skills=", ".join(skills),
                positions=", ".join(positions),
                num_queries=5
            ))
            
            queries = self._extract_json_from_response(query_response.content)
            
            if isinstance(queries, list) and len(queries) > 0:
                print(f"Generated {len(queries)} search queries:")
                for i, q in enumerate(queries[:5]):
                    print(f"  {i+1}. {q}")
                return queries[:5]  # Use top 5 queries
            else:
                # Fallback queries
                fallback_queries = []
                for pos in positions[:2]:
                    for skill in skills[:3]:
                        fallback_queries.append(f"{pos} {skill} jobs")
                
                print(f"Using fallback queries: {fallback_queries}")
                return fallback_queries
                
        except Exception as e:
            print(f"Error generating queries: {e}")
            # Basic fallback
            return [f"{' '.join(positions[:1])} {' '.join(skills[:3])} jobs"]

    def _fetch_with_query(self, query: str, site: str, target_skills: List[str]) -> List[Dict[str, Any]]:
        """Fetch jobs for a specific query and site."""
        site_query = f"{query} site:{site}"
        print(f"\n  Trying: {site_query}")
        
        try:
            raw_results = self.tavily.invoke({"query": site_query})
            
            if not raw_results:
                return []
            
            # Process results into text
            jobs_text_parts = []
            result_count = 0
            
            if isinstance(raw_results, list):
                for result in raw_results[:5]:  # Limit to 5
                    if isinstance(result, dict):
                        title = result.get('title', '')
                        url = result.get('url', '')
                        content = result.get('content', '')
                        
                        # Only include if it looks like a job listing
                        if title and any(keyword in title.lower() for keyword in ['job', 'hire', 'career', 'opening', 'position', 'role']):
                            jobs_text_parts.append(f"Title: {title}\nURL: {url}\nContent: {content[:500]}")
                            result_count += 1
            
            if result_count == 0:
                return []
            
            jobs_text = "\n\n".join(jobs_text_parts)
            
            # Extract relevant jobs using LLM
            extract_response = self.llm.invoke(self.extract_prompt.format(
                jobs_text=jobs_text,
                target_skills=", ".join(target_skills[:5])
            ))
            
            extracted = self._extract_json_from_response(extract_response.content)
            
            if isinstance(extracted, list) and len(extracted) > 0:
                formatted_jobs = []
                for job in extracted:
                    if not isinstance(job, dict):
                        continue
                    
                    # Clean up the job data
                    title = job.get("title", "Unknown Position").strip()
                    if title == "Unknown" or not title:
                        continue  # Skip invalid titles
                    
                    # Check if job requires user's skills
                    required_skills = job.get("required_skills", [])
                    if required_skills and isinstance(required_skills, list):
                        # Check skill overlap
                        user_skill_lower = [s.lower() for s in target_skills]
                        job_skills_lower = [str(s).lower() for s in required_skills]
                        
                        skill_overlap = any(
                            any(user_skill in job_skill or job_skill in user_skill 
                                for user_skill in user_skill_lower)
                            for job_skill in job_skills_lower
                        )
                        
                        if not skill_overlap:
                            continue  # Skip jobs that don't require user's skills
                    
                    job_id = f"{site}_{abs(hash(title + job.get('url', '')))}"
                    
                    formatted_job = {
                        "id": job_id,
                        "title": title,
                        "company": job.get("company", "Unknown Company"),
                        "description": job.get("description", "No description"),
                        "location": job.get("location", "Remote/On-site"),
                        "type": job.get("type", "Full-time"),
                        "url": job.get("url", "#"),
                        "required_skills": required_skills if required_skills else target_skills[:3],
                        "recruiterId": "external",
                        "match_score": 60.0,  # Base score, will be refined
                        "source_url": job.get("url", "#"),
                        "salary": "Not specified"
                    }
                    formatted_jobs.append(formatted_job)
                
                print(f"    Found {len(formatted_jobs)} relevant jobs")
                return formatted_jobs
                
        except Exception as e:
            print(f"    Error: {str(e)[:100]}")
        
        return []

    async def fetch_external_jobs(self, skills: List[str], positions: List[str]) -> List[Dict[str, Any]]:
        """Intelligent external job search based on user skills."""
        print(f"\n{'='*60}")
        print(f"Starting EXTERNAL job search")
        print(f"User skills: {skills}")
        print(f"User positions: {positions}")
        
        # Generate intelligent search queries
        search_queries = self._generate_search_queries(skills, positions)
        
        if not search_queries:
            print("No search queries generated")
            return []
        
        sites = ["indeed.com", "linkedin.com/jobs", "glassdoor.com"]
        all_jobs = []
        seen_titles = set()  # Avoid duplicates
        
        # Normalize skills for matching
        normalized_skills = [s.lower().strip() for s in skills]
        
        for site in sites:
            print(f"\n--- Searching {site} ---")
            site_jobs = []
            
            for query in search_queries[:3]:  # Try top 3 queries
                jobs = self._fetch_with_query(query, site, normalized_skills)
                
                for job in jobs:
                    # Check for duplicates
                    job_title = job.get("title", "").lower()
                    if job_title not in seen_titles:
                        seen_titles.add(job_title)
                        site_jobs.append(job)
                
                if len(site_jobs) >= 3:  # Limit per site
                    break
            
            if site_jobs:
                print(f"Found {len(site_jobs)} unique jobs from {site}")
                all_jobs.extend(site_jobs)
            
            time.sleep(1)  # Rate limiting
        
        print(f"\nTotal external jobs found: {len(all_jobs)}")
        
        # If no jobs found, provide better fallbacks
        if len(all_jobs) == 0:
            print("\nNo external jobs found, creating skill-based suggestions...")
            return self._create_skill_based_fallback(skills, positions)
        
        return all_jobs

    def _create_skill_based_fallback(self, skills: List[str], positions: List[str]) -> List[Dict[str, Any]]:
        """Create relevant fallback job suggestions based on skills."""
        fallback_jobs = []
        
        # Common job titles for educational technology + Adobe skills
        job_templates = [
            {
                "title": "Instructional Designer",
                "description": f"Create engaging learning experiences using {', '.join(skills[:3])}. Design and develop e-learning courses, interactive modules, and educational content.",
                "match_reason": "Combines educational technology with Adobe Creative Suite skills"
            },
            {
                "title": "E-Learning Developer",
                "description": f"Develop online courses and training materials using {skills[0] if skills else 'multimedia tools'}. Create interactive learning experiences for educational institutions or corporate training.",
                "match_reason": "Perfect for video editing and graphic design skills in education"
            },
            {
                "title": "Educational Media Specialist",
                "description": f"Produce educational videos, graphics, and multimedia content using {', '.join(skills[:2])}. Work with educators to create engaging learning materials.",
                "match_reason": "Direct application of Adobe skills in educational context"
            },
            {
                "title": "Learning Experience Designer",
                "description": f"Design user-centered learning experiences incorporating {skills[1] if len(skills) > 1 else 'multimedia elements'}. Focus on creating effective educational content.",
                "match_reason": "Blends educational technology with creative design skills"
            }
        ]
        
        for i, template in enumerate(job_templates):
            fallback_jobs.append({
                "id": f"fallback_{i+1}",
                "title": template["title"],
                "company": "Various Educational Institutions & EdTech Companies",
                "description": template["description"],
                "location": "Remote/Hybrid",
                "type": "Full-time",
                "url": "#",
                "required_skills": skills[:4],
                "recruiterId": "external",
                "match_score": 85.0 - (i * 5),  # Decreasing scores
                "source_url": "#",
                "salary": "Competitive",
                "source": "skill_based_fallback",
                "match_reason": template["match_reason"]
            })
        
        return fallback_jobs


class JobRecommendationService:
    """
    Intelligent recommendation service with dynamic skill-based queries.
    """
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.vstore = JobVectorStore.get(self.embeddings)
        self.external_fetcher = ExternalJobFetcher()
        
        # Improved scoring prompt
        self.scoring_prompt = PromptTemplate(
            input_variables=["user_skills", "user_positions", "jobs_list"],
            template="""Score each job (0-100) based on how well it matches the user's profile.

USER PROFILE:
Skills: {user_skills}
Positions/Roles: {user_positions}

JOBS TO SCORE:
{jobs_list}

SCORING CRITERIA:
1. Skill Match (40 points): How many required skills match user skills?
2. Role Relevance (30 points): How relevant is the job title to user's positions?
3. Industry Fit (20 points): Is this in education/edtech/learning field?
4. Experience Level (10 points): Appropriate for user's likely experience?

SCORING RULES:
- 90-100: Excellent match (most skills + relevant role)
- 70-89: Good match (many skills + somewhat relevant role)
- 50-69: Fair match (some skills, role may be different)
- 30-49: Poor match (few skills, different field)
- 0-29: Very poor match (no relevant skills)

Return ONLY a JSON object: {{"Exact Job Title 1": 85, "Exact Job Title 2": 72, ...}}
If no jobs, return empty object: {{}}"""
        )
        self.scoring_chain = LLMChain(llm=self.llm, prompt=self.scoring_prompt)

    def _extract_user_skills_and_positions(self, employee_info: Dict[str, Any]) -> tuple[List[str], List[str]]:
        """Extract and clean user skills and positions."""
        skills = self._parse_json_field(employee_info.get('skills', []))
        positions = self._parse_json_field(employee_info.get('position', []))
        
        # Extract skill names
        skill_names = []
        if isinstance(skills, list):
            for skill in skills:
                if isinstance(skill, dict):
                    skill_name = skill.get('name', '').strip()
                    if skill_name:
                        # Normalize skill names
                        skill_lower = skill_name.lower()
                        if "premium pro" in skill_lower:
                            skill_name = "Premiere Pro"
                        elif "photoshop" in skill_lower:
                            skill_name = "Adobe Photoshop"
                        elif "illustrator" in skill_lower:
                            skill_name = "Adobe Illustrator"
                        elif "video editor" in skill_lower:
                            skill_name = "Video Editing"
                        skill_names.append(skill_name)
                elif skill:
                    skill_names.append(str(skill).strip())
        
        # Extract position names
        position_names = []
        if isinstance(positions, list):
            for position in positions:
                if isinstance(position, dict):
                    pos_name = position.get('name', '').strip()
                    if pos_name:
                        position_names.append(pos_name)
                elif position:
                    position_names.append(str(position).strip())
        
        # Add related terms based on skills and positions
        if "Educational Technologist" in position_names or any("educat" in pos.lower() for pos in position_names):
            position_names.extend(["Instructional Designer", "E-Learning Developer", "Learning Experience Designer"])
        
        if any("photoshop" in skill.lower() or "illustrator" in skill.lower() for skill in skill_names):
            skill_names.extend(["Adobe Creative Suite", "Graphic Design", "Digital Media"])
        
        # Remove duplicates and empty strings
        skill_names = list(dict.fromkeys([s for s in skill_names if s]))
        position_names = list(dict.fromkeys([p for p in position_names if p]))
        
        print(f"\nExtracted user profile:")
        print(f"  Skills ({len(skill_names)}): {skill_names}")
        print(f"  Positions ({len(position_names)}): {position_names}")
        
        return skill_names, position_names

    def _parse_json_field(self, field):
        """Helper to parse JSON string fields to dict/list."""
        if isinstance(field, str):
            try:
                return json.loads(field)
            except json.JSONDecodeError:
                return []
        return field or []

    def _extract_json_from_response(self, response_text: str) -> Dict[str, float]:
        """Robust JSON extraction for scores."""
        if not response_text:
            return {}
        
        cleaned = response_text.strip()
        
        # Find JSON object
        json_match = re.search(r'\{.*\}', cleaned, re.DOTALL)
        if json_match:
            try:
                scores = json.loads(json_match.group(0))
                if isinstance(scores, dict):
                    result = {}
                    for title, score in scores.items():
                        try:
                            result[title] = float(score)
                        except (ValueError, TypeError):
                            result[title] = 50.0
                    return result
            except json.JSONDecodeError:
                pass
        
        return {}

    def _score_jobs_intelligently(self, user_skills: List[str], user_positions: List[str], job_docs: List[Document]) -> List[Dict[str, Any]]:
        """Intelligent job scoring based on skill and position matching."""
        if not job_docs:
            return []
        
        # Create jobs list for LLM scoring
        jobs_list_parts = []
        for d in job_docs:
            title = d.metadata.get('title', 'Unknown Title')
            content = d.page_content[:300]
            jobs_list_parts.append(f"Title: {title}\nDetails: {content}")
        
        jobs_list = "\n\n".join(jobs_list_parts)
        
        try:
            print(f"\nScoring {len(job_docs)} jobs...")
            response = self.scoring_chain.run(
                user_skills=", ".join(user_skills[:10]),
                user_positions=", ".join(user_positions[:5]),
                jobs_list=jobs_list
            )
            
            scores = self._extract_json_from_response(response)
            
            # Manual scoring fallback for any jobs not scored by LLM
            scored_jobs = []
            for d in job_docs:
                title = d.metadata.get('title', 'Unknown Title')
                
                if title in scores:
                    match_score = scores[title]
                else:
                    # Manual scoring based on skill overlap
                    content_lower = d.page_content.lower()
                    skill_match_count = sum(1 for skill in user_skills[:5] if skill.lower() in content_lower)
                    position_match = any(pos.lower() in content_lower for pos in user_positions[:3])
                    
                    base_score = min(skill_match_count * 15, 60)  # Up to 60 for skills
                    if position_match:
                        base_score += 20  # Bonus for position match
                    
                    match_score = min(base_score, 95)  # Cap at 95
                
                scored_jobs.append({
                    'title': title,
                    'match_score': match_score,
                    'document': d
                })
            
            return scored_jobs
            
        except Exception as e:
            print(f"Error scoring jobs: {e}")
            # Return default scores
            return [{
                'title': d.metadata.get('title', 'Unknown Title'), 
                'match_score': 50.0, 
                'document': d
            } for d in job_docs]

    async def _fetch_and_parse_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Fetch user data."""
        db = Prisma()
        await db.connect()
        try:
            user = await db.user.find_unique(
                where={"id": user_id}, 
                include={"employee": True}
            )
            
            if not user:
                return None
            
            employee_info = {}
            
            if user.employee:
                employee_info['skills'] = self._parse_json_field(getattr(user.employee, 'skills', []))
            else:
                employee_info['skills'] = []
            
            employee_info['position'] = self._parse_json_field(getattr(user, 'position', []))
            
            return employee_info
        finally:
            await db.disconnect()

    async def get_internal_jobs(self, user_id: str, recruiter_id: str) -> List[Dict[str, Any]]:
        """Get internal jobs with intelligent querying."""
        print(f"\n{'='*60}")
        print(f"Getting INTERNAL jobs")
        
        employee_info = await self._fetch_and_parse_user(user_id)
        if not employee_info:
            return []
        
        # Extract skills and positions for querying
        user_skills, user_positions = self._extract_user_skills_and_positions(employee_info)
        
        if not user_skills and not user_positions:
            return []
        
        # Create query from most relevant skills/positions
        query_terms = user_positions[:2] + user_skills[:3]
        search_query = " ".join(query_terms)
        print(f"Internal search query: {search_query}")
        
        db = Prisma()
        await db.connect()
        try:
            await self.vstore.build_or_load(db)
            
            # Search with and without recruiter filter
            embedding_scored = self.vstore.retrieve_jobs_with_scores(search_query, recruiter_id)
            
            if not embedding_scored and recruiter_id:
                print("No jobs for specific recruiter, searching all...")
                embedding_scored = self.vstore.retrieve_jobs_with_scores(search_query, "")
            
            if not embedding_scored:
                return []
            
            # Score jobs intelligently
            docs = [item['document'] for item in embedding_scored]
            scored = self._score_jobs_intelligently(user_skills, user_positions, docs)
            
            # Fetch full details
            ids = [d.metadata['id'] for d in docs if d.metadata.get('id')]
            if not ids:
                return []
            
            jobs = await db.job.find_many(
                where={"id": {"in": ids}}, 
                include={"recruiter": True}
            )
            
            result = []
            for j in jobs:
                job_title = j.title or "Unknown Title"
                job_score = next((s['match_score'] for s in scored if s['title'] == job_title), 50.0)
                
                result.append({
                    "id": j.id, 
                    "title": job_title,
                    "description": j.description or "No description",
                    "location": j.location or "Not specified",
                    "type": j.type or "Not specified",
                    "salary": j.salary or "Not specified",
                    "match_score": job_score,
                    "recruiter": {
                        "id": j.recruiter.id if j.recruiter else None, 
                        "firstName": j.recruiter.firstName if j.recruiter else "",
                        "lastName": j.recruiter.lastName if j.recruiter else ""
                    } if j.recruiter else None,
                    "is_external": False,
                    "source": "internal"
                })
            
            print(f"Found {len(result)} internal jobs")
            return result
            
        except Exception as e:
            print(f"Error getting internal jobs: {e}")
            return []
        finally:
            await db.disconnect()

    async def get_external_jobs(self, user_id: str) -> List[Dict[str, Any]]:
        """Get external jobs with intelligent query generation."""
        print(f"\n{'='*60}")
        print(f"Getting EXTERNAL jobs")
        
        employee_info = await self._fetch_and_parse_user(user_id)
        if not employee_info:
            return []
        
        # Extract skills and positions
        user_skills, user_positions = self._extract_user_skills_and_positions(employee_info)
        
        if not user_skills:
            print("No skills found for external job search")
            return []
        
        try:
            # Use the intelligent external fetcher
            external_jobs = await self.external_fetcher.fetch_external_jobs(user_skills, user_positions)
            
            if not external_jobs:
                return []
            
            print(f"Found {len(external_jobs)} external jobs, scoring...")
            
            # Create documents for scoring
            docs = []
            for job in external_jobs:
                title = job.get('title', 'Unknown Position')
                description = job.get('description', '')
                required_skills = job.get('required_skills', [])
                
                content = f"Title: {title}\nDescription: {description}\nRequired Skills: {', '.join(required_skills) if required_skills else 'Not specified'}"
                doc = Document(
                    page_content=content, 
                    metadata={
                        "title": title,
                        "description": description,
                        "required_skills": required_skills,
                        "id": job.get('id', '')
                    }
                )
                docs.append(doc)
            
            # Score jobs
            scored = self._score_jobs_intelligently(user_skills, user_positions, docs)
            
            # Combine scores with jobs
            for job, score_item in zip(external_jobs, scored):
                job["match_score"] = score_item['match_score']
                job["is_external"] = True
                job["source"] = job.get("source", "external")
            
            return external_jobs
            
        except Exception as e:
            print(f"Error getting external jobs: {e}")
            return []

    async def recommend_jobs(self, user_id: str, recruiter_id: str, include_external: bool = True) -> List[Dict[str, Any]]:
        """Main recommendation method."""
        print(f"\n{'='*60}")
        print(f"JOB RECOMMENDATIONS for user: {user_id}")
        
        try:
            # Get internal jobs
            internal = await self.get_internal_jobs(user_id, recruiter_id)
            
            # Get external jobs
            external = []
            if include_external:
                external = await self.get_external_jobs(user_id)
            
            # Combine all jobs
            all_jobs = internal + external
            
            if not all_jobs:
                print("\n⚠️ No jobs found. Checking user profile...")
                # Try to get user profile to understand why
                employee_info = await self._fetch_and_parse_user(user_id)
                if employee_info:
                    user_skills, user_positions = self._extract_user_skills_and_positions(employee_info)
                    print(f"User profile - Skills: {user_skills}, Positions: {user_positions}")
                
                # Return empty array (frontend should handle "no results" message)
                return []
            
            # Sort by match score
            all_jobs.sort(key=lambda x: x.get("match_score", 0), reverse=True)
            
            # Enhance job descriptions with match reasons
            for job in all_jobs[:TOP_K]:
                score = job.get("match_score", 50)
                if score >= 80:
                    job["match_reason"] = "Excellent skill and role match"
                elif score >= 60:
                    job["match_reason"] = "Good skill alignment with your profile"
                else:
                    job["match_reason"] = "Partial skill match"
            
            result = all_jobs[:TOP_K]
            
            print(f"\n📊 FINAL RESULTS ({len(result)} jobs):")
            for i, job in enumerate(result):
                source = "internal" if not job.get("is_external") else "external"
                print(f"{i+1}. {job['title']} - {job.get('match_score', 0):.0f} pts ({source})")
            
            return result
            
        except Exception as e:
            print(f"Error in recommend_jobs: {e}")
            return []