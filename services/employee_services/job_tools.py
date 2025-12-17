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
from langchain_community.tools.tavily_search import TavilySearchResults

# --- Configuration ---
def get_faiss_index_dir() -> str:
    """Get the directory for the FAISS index."""
    index_dir = os.getenv("FAISS_INDEX_PATH", os.path.join(os.path.dirname(__file__), "faiss_jobs_index"))
    os.makedirs(index_dir, exist_ok=True)
    return index_dir

INDEX_DIR = get_faiss_index_dir()
TOP_K = int(os.getenv("JOBS_RETRIEVE_TOP_K", "10"))


@dataclass
class JobRow:
    """Represents a simplified job row for vector storage."""
    id: str
    title: str
    description: Optional[str]
    recruiterId: str
    location: Optional[str] = None
    type: Optional[str] = None


class JobVectorStore:
    """
    Singleton class to manage the FAISS vector store for internal jobs.
    Handles building, loading, and searching the index.
    """
    _instance: Optional["JobVectorStore"] = None

    def __init__(self, embeddings: OpenAIEmbeddings):
        self.embeddings = embeddings
        self.vs: Optional[FAISS] = None
        self._loaded = False

    @classmethod
    def get(cls, embeddings: OpenAIEmbeddings) -> "JobVectorStore":
        """Get the singleton instance of JobVectorStore."""
        if cls._instance is None:
            cls._instance = JobVectorStore(embeddings)
        return cls._instance

    def _job_to_document(self, job: JobRow) -> Document:
        """Convert a JobRow to a LangChain Document."""
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
        """Build the index from DB or load if already in memory."""
        if self._loaded and self.vs:
            return  # Already loaded

        Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)

        # Fetch all jobs from the database
        jobs = await db.job.find_many()
        print(f"JobVectorStore: Total jobs in database: {len(jobs)}")
        
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
            print(f"JobVectorStore: FAISS index created with {len(docs)} documents")
        else:
            self.vs = FAISS.from_texts(["NO_JOBS"], self.embeddings)
            print("JobVectorStore: FAISS index created with placeholder (no jobs in DB)")

        self._loaded = True

    def retrieve_jobs_with_scores(self, query_text: str, recruiter_id: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """Search the vector store for jobs matching the query."""
        if not self.vs:
            print("JobVectorStore: Store not initialized")
            return []
        
        print(f"JobVectorStore: Searching for '{query_text}' (Recruiter: {recruiter_id or 'Any'})")
        
        try:
            results = self.vs.similarity_search_with_score(query_text, k=k)
            
            scored_jobs = []
            for doc, score in results:
                doc_recruiter = doc.metadata.get("recruiterId")
                
                # Filter by recruiter if specified
                if not recruiter_id or doc_recruiter == recruiter_id:
                    # Convert distance to similarity score (approximate)
                    similarity_score = float((1 - min(float(score), 1.0)) * 100)
                    scored_jobs.append({
                        'title': doc.metadata.get('title', 'Unknown Title'),
                        'match_score': similarity_score,
                        'document': doc
                    })
            
            return scored_jobs
            
        except Exception as e:
            print(f"JobVectorStore: Error in search: {e}")
            return []


class InternalJobFetcher:
    """
    Tool to fetch internal jobs from the database using vector search.
    """
    def __init__(self, embeddings: OpenAIEmbeddings):
        self.vstore = JobVectorStore.get(embeddings)

    async def fetch_jobs(self, query: str, recruiter_id: str) -> List[Dict[str, Any]]:
        """
        Fetch internal jobs matching the query.
        Returns a list of jobs with their vector documents and initial scores.
        """
        db = Prisma()
        await db.connect()
        try:
            # Ensure vector store is ready
            await self.vstore.build_or_load(db)
            
            # 1. Search vector store
            embedding_scored = self.vstore.retrieve_jobs_with_scores(query, recruiter_id)
            
            # If no results for specific recruiter, try global search (fallback)
            if not embedding_scored and recruiter_id:
                print("InternalJobFetcher: No jobs for specific recruiter, searching all...")
                embedding_scored = self.vstore.retrieve_jobs_with_scores(query, "")
            
            if not embedding_scored:
                return []
            
            # 2. Fetch full details from DB for the found jobs
            # We need to map back from vector docs to DB records
            docs_map = {item['document'].metadata['id']: item for item in embedding_scored if item['document'].metadata.get('id')}
            ids = list(docs_map.keys())
            
            if not ids:
                return []
            
            jobs_data = await db.job.find_many(
                where={"id": {"in": ids}}, 
                include={"recruiter": True}
            )
            
            # 3. Combine DB data with vector scores
            results = []
            for job in jobs_data:
                vector_info = docs_map.get(job.id)
                if not vector_info:
                    continue
                    
                results.append({
                    "db_job": job,
                    "vector_doc": vector_info['document'],
                    "vector_score": vector_info['match_score']
                })
                
            print(f"InternalJobFetcher: Found {len(results)} jobs")
            return results
            
        except Exception as e:
            print(f"InternalJobFetcher: Error fetching jobs: {e}")
            return []
        finally:
            await db.disconnect()


class ExternalJobFetcher:
    """
    Tool for fetching external jobs from the web (Indeed, LinkedIn, Glassdoor)
    using Tavily search and LLM-based extraction.
    """
    def __init__(self):
        self.tavily = TavilySearchResults(max_results=8)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.2)
        
        # --- Prompts ---
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

Return ONLY a JSON array of strings. Example: ["Java Developer remote", "Software Engineer Python"]
"""
        )
        
        self.extract_prompt = PromptTemplate(
            input_variables=["jobs_text", "target_skills"],
            template="""Extract job listings from the text below that match these skills: {target_skills}

Input text:
{jobs_text}

Instructions:
1. Extract ONLY jobs requiring at least one target skill.
2. For each job, extract: title, company, description, location, type, url, required_skills.
3. Return ONLY a valid JSON array of objects.
4. If no jobs found, return [].
"""
        )

    def _extract_json_from_response(self, response_text: str) -> Optional[List[Dict]]:
        """Helper to safely parse JSON from LLM response."""
        if not response_text: return None
        cleaned = response_text.strip()
        # Try finding JSON array pattern
        match = re.search(r'\[\s*\{.*\}\s*\]', cleaned, re.DOTALL)
        if match:
            try: return json.loads(match.group(0))
            except: pass
        try: return json.loads(cleaned)
        except: return []

    def _generate_search_queries(self, skills: List[str], positions: List[str]) -> List[str]:
        """Generate search queries using LLM."""
        try:
            response = self.llm.invoke(self.query_gen_prompt.format(
                skills=", ".join(skills),
                positions=", ".join(positions),
                num_queries=5
            ))
            queries = self._extract_json_from_response(response.content)
            if queries and isinstance(queries, list):
                return queries[:5]
        except Exception as e:
            print(f"ExternalJobFetcher: Error generating queries: {e}")
        
        # Fallback
        return [f"{' '.join(positions[:1])} {' '.join(skills[:3])} jobs"]

    def _fetch_with_query(self, query: str, site: str, target_skills: List[str]) -> List[Dict[str, Any]]:
        """Search a specific site with a query."""
        site_query = f"{query} site:{site}"
        print(f"  ExternalJobFetcher: Searching {site} for '{query}'")
        
        try:
            raw_results = self.tavily.invoke({"query": site_query})
            if not raw_results or not isinstance(raw_results, list):
                return []
            
            # Prepare text for LLM extraction
            jobs_text = "\n\n".join([
                f"Title: {r.get('title','')}\nURL: {r.get('url','')}\nContent: {r.get('content','')[:500]}"
                for r in raw_results[:5]
                if isinstance(r, dict) and r.get('title')
            ])
            
            if not jobs_text: return []

            # Extract structured data
            response = self.llm.invoke(self.extract_prompt.format(
                jobs_text=jobs_text,
                target_skills=", ".join(target_skills[:5])
            ))
            
            extracted = self._extract_json_from_response(response.content)
            if not extracted or not isinstance(extracted, list):
                return []

            formatted_jobs = []
            for job in extracted:
                if not isinstance(job, dict): continue
                
                title = job.get("title", "Unknown").strip()
                if not title or title == "Unknown": continue

                formatted_jobs.append({
                    "id": f"{site}_{abs(hash(title + job.get('url', '')))}",
                    "title": title,
                    "company": job.get("company", "Unknown"),
                    "description": job.get("description", ""),
                    "location": job.get("location", "Remote/On-site"),
                    "type": job.get("type", "Full-time"),
                    "url": job.get("url", "#"),
                    "required_skills": job.get("required_skills", target_skills[:3]),
                    "recruiterId": "external",
                    "match_score": 60.0, # Base score
                    "source_url": job.get("url", "#"),
                    "is_external": True,
                    "source": "external"
                })
            
            return formatted_jobs

        except Exception as e:
            print(f"  ExternalJobFetcher: Error searching {site}: {e}")
            return []

    async def fetch_external_jobs(self, skills: List[str], positions: List[str]) -> List[Dict[str, Any]]:
        """Main method to fetch external jobs."""
        print(f"ExternalJobFetcher: Starting search for skills={skills}, positions={positions}")
        
        queries = self._generate_search_queries(skills, positions)
        sites = ["indeed.com", "linkedin.com/jobs", "glassdoor.com"]
        all_jobs = []
        seen_titles = set()
        
        for site in sites:
            site_jobs = []
            for query in queries[:3]:
                jobs = self._fetch_with_query(query, site, skills)
                for job in jobs:
                    if job['title'] not in seen_titles:
                        seen_titles.add(job['title'])
                        site_jobs.append(job)
                if len(site_jobs) >= 2: # Limit per site as requested
                    break
            all_jobs.extend(site_jobs)
            time.sleep(1) # Polite delay
            
        print(f"ExternalJobFetcher: Total found: {len(all_jobs)}")
        
        if not all_jobs:
            return self._create_skill_based_fallback(skills)
            
        return all_jobs

    def _create_skill_based_fallback(self, skills: List[str]) -> List[Dict[str, Any]]:
        """Generate fallback jobs if none found."""
        return [{
            "id": f"fallback_{i}",
            "title": f"{skill} Specialist",
            "company": "Tech Company",
            "description": f"Looking for an expert in {skill}.",
            "location": "Remote",
            "type": "Full-time",
            "url": "#",
            "required_skills": [skill],
            "recruiterId": "external",
            "match_score": 70.0,
            "is_external": True,
            "source": "skill_based_fallback"
        } for i, skill in enumerate(skills[:3])]
