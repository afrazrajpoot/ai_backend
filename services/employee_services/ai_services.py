# services/rag_job_recommendation_service.py
import os
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from prisma import Prisma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

logger = logging.getLogger(__name__)

INDEX_DIR = os.getenv("JOBS_FAISS_DIR", "./faiss_jobs_index")
TOP_K = int(os.getenv("JOBS_RETRIEVE_TOP_K", "25"))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


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
    FAISS store for all jobs. Built only once and reused.
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
            return  # Already built

        # Try loading persisted index
        if os.path.exists(INDEX_DIR):
            try:
                self.vs = FAISS.load_local(INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded FAISS index from disk")
                self._loaded = True
                return
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")

        # Build from DB
        jobs = await db.job.find_many()
        docs: List[Document] = [self._job_to_document(JobRow(
            id=j.id,
            title=j.title,
            description=j.description,
            recruiterId=j.recruiterId,
            location=getattr(j, "location", None),
            type=getattr(j, "type", None)
        )) for j in jobs]

        if not docs:
            self.vs = FAISS.from_texts(["NO_JOBS"], self.embeddings, metadatas=[{"placeholder": True}])
        else:
            self.vs = FAISS.from_documents(docs, self.embeddings)

        os.makedirs(INDEX_DIR, exist_ok=True)
        self.vs.save_local(INDEX_DIR)
        self._loaded = True
        logger.info("FAISS index built and saved")

    def retrieve_jobs(self, query_text: str, recruiter_id: str, k: int = TOP_K) -> List[Document]:
        if not self.vs:
            return []
        # Retrieve top-K relevant jobs from FAISS
        docs = self.vs.similarity_search(query_text, k=k)
        # Filter by recruiterId
        return [d for d in docs if d.metadata.get("recruiterId") == recruiter_id]


class JobRecommendationService:
    """
    RAG + recruiter-aware job recommendation.
    """
    def __init__(self):
        self.llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0.3)
        self.embeddings = OpenAIEmbeddings()
        self.vstore = JobVectorStore.get(self.embeddings)

    async def recommend_jobs_for_employee(self, user_id: str, recruiter_id: str) -> List[Dict[str, Any]]:
        db = Prisma()
        await db.connect()
        try:
            # Build/load FAISS store
            await self.vstore.build_or_load(db)

            # Fetch user and expand related employee
            user = await db.user.find_unique(
                where={"id": user_id},
                include={"employee": True}
            )
            if not user or not user.employee:
                raise ValueError("User or related Employee not found")

            employee = user.employee

            # Build query text for FAISS search using rich employee info
            employee_info = {
                "firstName": employee.firstName,
                "lastName": employee.lastName,
                "bio": employee.bio,
                "skills": employee.skills,
                "education": employee.education,
                "experience": employee.experience,
                "position": getattr(user, "position", None),
                "department": getattr(user, "department", None),
                "salary": getattr(user, "salary", None)
            }

            query_text = json.dumps(employee_info)

            # Retrieve top jobs filtered by recruiter
            retrieved_docs = self.vstore.retrieve_jobs(query_text, recruiter_id)

            if not retrieved_docs:
                logger.info("No jobs retrieved for this recruiter")
                return []

            # Prepare job details for LLM
            available_jobs = [{
                "title": d.metadata.get("title"),
                "description": d.page_content
            } for d in retrieved_docs]

            # Ask LLM to score jobs and return match scores
            system_msg = (
                "You are a job recommendation AI. "
                "Given the employee profile and available jobs, "
                "return a JSON array of objects with `title` and `match_score` (0-100). "
                "Use all available employee data to evaluate fit."
            )
            user_msg = f"Employee profile:\n{json.dumps(employee_info)}\n\nAvailable jobs:\n{json.dumps(available_jobs)}"

            llm_resp = self.llm.invoke([
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ])

            raw = (llm_resp.content or "").strip()
            if raw.startswith("```"):
                raw = raw.replace("```json", "").replace("```", "").strip()

            try:
                recommended_jobs = json.loads(raw)
                # Ensure all match_scores are numbers
                recommended_jobs = [
                    {"title": j.get("title"), "match_score": float(j.get("match_score", 0))}
                    for j in recommended_jobs if j.get("title")
                ]
            except Exception as e:
                logger.warning(f"Failed to parse LLM output: {e}")
                recommended_jobs = []

            if not recommended_jobs:
                logger.info("LLM returned no recommended jobs")
                return []

            # Filter final DB jobs by recruiter and recommended titles
            titles = [j["title"] for j in recommended_jobs]
            final_jobs = await db.job.find_many(
                where={"recruiterId": recruiter_id, "title": {"in": titles}},
                include={"recruiter": True}
            )

            # Merge match scores
            final_jobs_with_scores = []
            for j in final_jobs:
                score = next((r["match_score"] for r in recommended_jobs if r["title"] == j.title), 0)
                final_jobs_with_scores.append({
                    "id": j.id,
                    "title": j.title,
                    "description": j.description,
                    "recruiterId": j.recruiterId,
                    "location": j.location,
                    "type": j.type,
                    "match_score": score,
                    "recruiter": {
                        "id": j.recruiter.id if getattr(j, "recruiter", None) else None,
                        "firstName": j.recruiter.firstName if getattr(j, "recruiter", None) else None,
                        "lastName": j.recruiter.lastName if getattr(j, "recruiter", None) else None,
                    }
                })

            # Sort by match_score descending
            final_jobs_with_scores.sort(key=lambda x: x["match_score"], reverse=True)
            return final_jobs_with_scores

        finally:
            await db.disconnect()
