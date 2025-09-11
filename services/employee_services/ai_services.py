import os
import json
import logging
import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import re

from prisma import Prisma
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

INDEX_DIR = os.getenv("JOBS_FAISS_DIR", "./faiss_jobs_index")
TOP_K = int(os.getenv("JOBS_RETRIEVE_TOP_K", "25"))


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

        # Ensure directory exists
        os.makedirs(INDEX_DIR, exist_ok=True)

        # Try loading persisted index - check if index files actually exist
        index_file = os.path.join(INDEX_DIR, "index.faiss")
        pkl_file = os.path.join(INDEX_DIR, "index.pkl")
        
        if os.path.exists(index_file) and os.path.exists(pkl_file):
            try:
                self.vs = FAISS.load_local(INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True)
                logger.info("Loaded FAISS index from disk")
                self._loaded = True
                return
            except Exception as e:
                logger.warning(f"Failed to load FAISS index: {e}")
                # Clean up corrupted files
                try:
                    if os.path.exists(index_file):
                        os.remove(index_file)
                    if os.path.exists(pkl_file):
                        os.remove(pkl_file)
                    logger.info("Removed corrupted FAISS index files")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup corrupted files: {cleanup_error}")

        # Build from DB
        logger.info("Building FAISS index from database...")
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
            logger.info("Created placeholder FAISS index (no jobs found)")
        else:
            self.vs = FAISS.from_documents(docs, self.embeddings)
            logger.info(f"Created FAISS index with {len(docs)} documents")

        # Save the index
        try:
            self.vs.save_local(INDEX_DIR)
            logger.info("FAISS index built and saved to disk")
        except Exception as e:
            logger.error(f"Failed to save FAISS index: {e}")
            # Continue anyway since we have the index in memory
        
        self._loaded = True
    
    def retrieve_jobs(self, query_text: str, recruiter_id: str, k: int = TOP_K) -> List[Document]:
        if not self.vs:
            return []
        # Retrieve top-K relevant jobs from FAISS
        docs = self.vs.similarity_search(query_text, k=k)
        # Filter by recruiterId
        return [d for d in docs if d.metadata.get("recruiterId") == recruiter_id]

    def retrieve_jobs_with_scores(self, query_text: str, recruiter_id: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """Retrieve jobs with similarity scores using FAISS embeddings"""
        if not self.vs:
            return []
        
        # Search with scores
        results = self.vs.similarity_search_with_score(query_text, k=k)
        
        # Filter by recruiter and format
        scored_jobs = []
        for doc, score in results:
            if doc.metadata.get("recruiterId") == recruiter_id:
                # Convert distance to similarity score (0-100)
                similarity_score = (1 - min(score, 1.0)) * 100
                scored_jobs.append({
                    'title': doc.metadata.get('title'),
                    'match_score': similarity_score,
                    'document': doc
                })
        
        return scored_jobs


class JobRecommendationService:
    """
    Job recommendation using TF-IDF + cosine similarity + feature engineering
    """
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.vstore = JobVectorStore.get(self.embeddings)
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=5000,
            min_df=2,
            max_df=0.8
        )
    
    def _extract_features(self, employee_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and preprocess features from employee data"""
        features = {}
        
        # Text features
        bio = employee_info.get('bio', '')
        skills = ' '.join([str(skill).lower() for skill in employee_info.get('skills', [])])
        
        # Handle arrays for education and experience
        education_list = employee_info.get('education', [])
        education = ' '.join([str(edu).lower() for edu in education_list if edu])
        
        experience_list = employee_info.get('experience', [])
        experience = ' '.join([str(exp).lower() for exp in experience_list if exp])
        
        features['text_data'] = f"{bio} {skills} {education} {experience}"
        features['text_data'] = re.sub(r'[^\w\s]', '', features['text_data'])
        features['text_data'] = re.sub(r'\s+', ' ', features['text_data']).strip().lower()
        
        # Handle array fields - convert to strings
        department_list = employee_info.get('department', [])
        features['department'] = ' '.join([str(dept).lower() for dept in department_list if dept])
        
        position_list = employee_info.get('position', [])
        features['position'] = ' '.join([str(pos).lower() for pos in position_list if pos])
        
        features['skills_list'] = [str(skill).lower() for skill in employee_info.get('skills', [])]
        
        return features
    
    def _calculate_similarity(self, employee_features: Dict[str, Any], job_docs: List[Document]) -> List[Dict[str, Any]]:
        """Calculate similarity scores between employee and jobs using multiple techniques"""
        if not job_docs:
            return []
        
        # Method 1: FAISS embedding similarity
        query_text = employee_features['text_data']
        recruiter_id = job_docs[0].metadata.get("recruiterId") if job_docs else ""
        embedding_scores = self.vstore.retrieve_jobs_with_scores(query_text, recruiter_id, k=len(job_docs))
        
        # Method 2: TF-IDF similarity
        job_texts = [doc.page_content.lower() for doc in job_docs]
        all_texts = [employee_features['text_data']] + job_texts
        
        try:
            tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
            employee_tfidf = tfidf_matrix[0:1]
            job_tfidfs = tfidf_matrix[1:]
            tfidf_similarities = cosine_similarity(employee_tfidf, job_tfidfs).flatten() * 100
        except Exception as e:
            logger.warning(f"TF-IDF failed: {e}")
            tfidf_similarities = np.zeros(len(job_docs))
        
        enhanced_scores = []
        
        for i, doc in enumerate(job_docs):
            job_content = doc.page_content.lower()
            
            # Get embedding score if available
            embedding_score = 0
            for emb_score in embedding_scores:
                if emb_score['title'] == doc.metadata.get('title'):
                    embedding_score = emb_score['match_score']
                    break
            
            # Keyword matching for skills
            skills_match = 0
            if employee_features['skills_list']:
                skills_found = sum(1 for skill in employee_features['skills_list'] 
                                 if skill and skill in job_content)
                skills_match = (skills_found / len(employee_features['skills_list'])) * 100
            
            # Department matching (check if any department appears in job content)
            dept_match = 30  # Default score
            if employee_features['department']:
                dept_words = employee_features['department'].split()
                dept_found = any(dept_word in job_content for dept_word in dept_words if dept_word)
                dept_match = 100 if dept_found else 30
            
            # Position matching (check if any position appears in job content)
            position_match = 30  # Default score
            if employee_features['position']:
                position_words = employee_features['position'].split()
                position_found = any(pos_word in job_content for pos_word in position_words if pos_word)
                position_match = 100 if position_found else 30
            
            # Combine scores (weighted average)
            final_score = (
                0.4 * embedding_score +
                0.3 * tfidf_similarities[i] +
                0.15 * skills_match +
                0.075 * dept_match +
                0.075 * position_match
            )
            
            enhanced_scores.append({
                'title': doc.metadata.get('title'),
                'match_score': min(100, max(0, round(final_score, 2))),
                'document': doc
            })
        
        return enhanced_scores
    
    def _simple_keyword_matching(self, employee_features: Dict[str, Any], job_docs: List[Document]) -> List[Dict[str, Any]]:
        """Fallback method using simple keyword matching"""
        scored_jobs = []
        
        for doc in job_docs:
            job_content = doc.page_content.lower()
            score = 0
            
            # Count matching skills
            skills = employee_features.get('skills_list', [])
            if skills:
                skills_found = sum(1 for skill in skills if skill and skill in job_content)
                score += (skills_found / len(skills)) * 40
            
            # Check department match (any department word)
            dept_text = employee_features.get('department', '')
            if dept_text:
                dept_words = dept_text.split()
                dept_found = any(dept_word in job_content for dept_word in dept_words if dept_word)
                if dept_found:
                    score += 30
            
            # Check position match (any position word)
            position_text = employee_features.get('position', '')
            if position_text:
                position_words = position_text.split()
                position_found = any(pos_word in job_content for pos_word in position_words if pos_word)
                if position_found:
                    score += 30
            
            scored_jobs.append({
                'title': doc.metadata.get('title'),
                'match_score': min(100, score),
                'document': doc
            })
        
        return scored_jobs

    async def recommend_jobs_for_employee(self, user_id: str, recruiter_id: str) -> List[Dict[str, Any]]:
        db = Prisma()
        await db.connect()
        
        try:
            # Build/load FAISS store
            await self.vstore.build_or_load(db)
            
            # Fetch user data with proper handling of array fields
            user = await db.user.find_unique(
                where={"id": user_id},
                include={"employee": True}
            )
            
            if not user:
                raise ValueError("User not found")

            # Build employee info with proper array handling
            employee_info = {
                "firstName": user.firstName or "",
                "lastName": user.lastName or "",
                "bio": getattr(user.employee, "bio", "") if user.employee else "",
                "skills": getattr(user.employee, "skills", []) if user.employee else [],
                "education": getattr(user.employee, "education", []) if user.employee else [],
                "experience": getattr(user.employee, "experience", []) if user.employee else [],
                "position": user.position or [],  # This is now an array
                "department": user.department or [],  # This is now an array
                "salary": user.salary or ""
            }

            # Extract features
            employee_features = self._extract_features(employee_info)
            
            # Retrieve jobs using the original query text for FAISS
            query_text = json.dumps(employee_info)
            retrieved_docs = self.vstore.retrieve_jobs(query_text, recruiter_id, k=TOP_K)
            
            if not retrieved_docs:
                logger.info("No jobs retrieved for this recruiter")
                return []
            
            # Calculate similarity scores
            try:
                recommended_jobs = self._calculate_similarity(employee_features, retrieved_docs)
            except Exception as e:
                logger.warning(f"Advanced similarity failed, using fallback: {e}")
                recommended_jobs = self._simple_keyword_matching(employee_features, retrieved_docs)
            
            # Get final job details from database
            titles = [j['title'] for j in recommended_jobs]
            if not titles:
                return []
                
            final_jobs = await db.job.find_many(
                where={"recruiterId": recruiter_id, "title": {"in": titles}},
                include={"recruiter": True}
            )
            
            # Merge with scores
            final_jobs_with_scores = []
            for job in final_jobs:
                score_data = next((r for r in recommended_jobs if r['title'] == job.title), None)
                if score_data:
                    final_jobs_with_scores.append({
                        "id": job.id,
                        "title": job.title,
                        "description": job.description,
                        "recruiterId": job.recruiterId,
                        "location": job.location,
                        "type": job.type,
                        "match_score": score_data['match_score'],
                        "salary": job.salary,
                        "recruiter": {
                            "id": job.recruiter.id if job.recruiter else None,
                            "firstName": job.recruiter.firstName if job.recruiter else None,
                            "lastName": job.recruiter.lastName if job.recruiter else None,
                        }
                    })
            
            # Sort by score and return top results
            final_jobs_with_scores.sort(key=lambda x: x["match_score"], reverse=True)
            return final_jobs_with_scores[:TOP_K]
            
        except Exception as e:
            logger.error(f"Error in job recommendation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            await db.disconnect()