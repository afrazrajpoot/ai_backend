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
        else:
            self.vs = FAISS.from_texts(["NO_JOBS"], self.embeddings)

        self._loaded = True

    def retrieve_jobs_with_scores(self, query_text: str, recruiter_id: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """Search the vector store for jobs matching the query."""
        if not self.vs:
            return []
        
        
        
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
                
            return results
            
        except Exception as e:
            return []
        finally:
            await db.disconnect()


class ExternalJobFetcher:
    """
    Tool for fetching external jobs from the web (Indeed, LinkedIn, Glassdoor)
    using Tavily search and LLM-based extraction.
    """
    def __init__(self):
        self.tavily = TavilySearchResults(max_results=12)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
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
                num_queries=3
            ))
            queries = self._extract_json_from_response(response.content)
            if queries and isinstance(queries, list):
                return queries[:3]
        except Exception as e:
            pass
        
        # Fallback
        return [f"{' '.join(positions[:1])} {' '.join(skills[:3])} jobs"]

    def _fetch_with_query(self, query: str, site: str, target_skills: List[str]) -> List[Dict[str, Any]]:
        """Search a specific site with a query."""
        site_query = f"{query} site:{site}"

        try:
            raw_results = self.tavily.invoke({"query": site_query})

            if not raw_results or not isinstance(raw_results, list):
                return []
            
            # Prepare text for LLM extraction
            jobs_text = "\n\n".join([
                f"Title: {r.get('title','')}\nURL: {r.get('url','')}\nContent: {r.get('content','')[:500]}"
                for r in raw_results[:10]
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
                extracted = []

            # Create jobs using raw data titles instead of LLM extracted titles
            formatted_jobs = []
            for raw_result in raw_results[:10]:  # Use first 10 raw results
                if not isinstance(raw_result, dict): continue

                raw_title = raw_result.get("title", "")
                if not raw_title: continue

                # Try to find matching extracted job for additional details
                matching_extracted = None
                for job in extracted:
                    if isinstance(job, dict) and job.get("title"):
                        # Simple matching based on title similarity
                        if raw_title.lower().replace(" ", "") == job.get("title", "").lower().replace(" ", ""):
                            matching_extracted = job
                            break

                # Use extracted data if available, otherwise use defaults
                company = "Unknown"
                description = raw_result.get("content", "")[:200]  # Use raw content as description
                location = "Remote/On-site"
                job_type = "Full-time"
                url = raw_result.get("url", "#")
                required_skills = target_skills[:3]

                if matching_extracted:
                    company = matching_extracted.get("company", company)
                    description = matching_extracted.get("description", description)
                    location = matching_extracted.get("location", location)
                    job_type = matching_extracted.get("type", job_type)
                    url = matching_extracted.get("url", url)
                    required_skills = matching_extracted.get("required_skills", required_skills)

                formatted_jobs.append({
                    "id": f"{site}_{abs(hash(raw_title + url))}",
                    "title": raw_title,  # Use raw title from web search
                    "company": company,
                    "description": description,
                    "location": location,
                    "type": job_type,
                    "url": url,
                    "required_skills": required_skills,
                    "recruiterId": "external",
                    "match_score": 60.0, # Base score
                    "source_url": url,
                    "is_external": True,
                    "source": "external"
                })
            
            return formatted_jobs

        except Exception as e:
            return []

    async def fetch_external_jobs(self, skills: List[str], positions: List[str]) -> List[Dict[str, Any]]:
        """Main method to fetch external jobs (generates queries first)."""
        queries = self._generate_search_queries(skills, positions)
        return await self.fetch_with_queries(queries, skills)

    async def fetch_with_queries(self, queries: List[str], target_skills: List[str]) -> List[Dict[str, Any]]:
        """Fetch external jobs using pre-generated queries."""
        import asyncio

        sites = ["indeed.com", "linkedin.com/jobs", "glassdoor.com"]
        
        # Create tasks for parallel execution
        tasks = []
        
        # If only 1 query, search top 2 sites to guarantee results
        if len(queries) == 1:
            query = queries[0]
            for site in sites[:2]: # Search Indeed and LinkedIn
                 tasks.append(asyncio.to_thread(self._fetch_with_query, query, site, target_skills))
        else:
            for i, query in enumerate(queries):
                # Rotate through sites for each query
                site = sites[i % len(sites)]
                # We need to run the synchronous _fetch_with_query in a thread to avoid blocking
                tasks.append(asyncio.to_thread(self._fetch_with_query, query, site, target_skills))
        
        # Run all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        all_jobs = []
        seen_titles = set()
        
        for jobs in results:
            if isinstance(jobs, list):
                for job in jobs:
                    if job['title'] not in seen_titles:
                        seen_titles.add(job['title'])
                        all_jobs.append(job)
            elif isinstance(jobs, Exception):
                pass


        if not all_jobs:
            return []

        return all_jobs
