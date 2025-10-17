import os
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import re
from collections import defaultdict
from pathlib import Path

from prisma import Prisma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)

# Production-safe directory initialization
def get_faiss_index_dir() -> str:
    """
    Get and ensure FAISS index directory exists with proper error handling
    """
    # First, check if explicit env var is set
    explicit_dir = os.getenv("JOBS_FAISS_DIR")
    if explicit_dir:
        index_dir = explicit_dir
    else:
        # Fallback: calculate from file location
        try:
            employee_services_dir = os.path.dirname(os.path.abspath(__file__))
            services_dir = os.path.dirname(employee_services_dir)
            project_root = os.path.dirname(services_dir)
            index_dir = os.path.join(project_root, "faiss_jobs_index")
        except Exception as e:
            logger.error(f"Failed to calculate PROJECT_ROOT: {e}")
            # Emergency fallback to /tmp or current directory
            index_dir = os.path.join(os.getcwd(), "faiss_jobs_index")
    
    # Ensure directory exists with proper error handling
    try:
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"FAISS index directory ready: {index_dir}")
        
        # Test write permissions
        test_file = os.path.join(index_dir, ".write_test")
        with open(test_file, 'w') as f:
            f.write("test")
        os.remove(test_file)
        logger.info(f"Write permissions verified for: {index_dir}")
        
    except PermissionError as e:
        logger.error(f"Permission denied creating/writing to {index_dir}: {e}")
        # Try to use alternative directory
        alt_dir = os.path.join("/tmp", "faiss_jobs_index")
        logger.warning(f"Attempting fallback directory: {alt_dir}")
        try:
            Path(alt_dir).mkdir(parents=True, exist_ok=True)
            index_dir = alt_dir
            logger.info(f"Using fallback FAISS directory: {alt_dir}")
        except Exception as e2:
            logger.error(f"Fallback directory also failed: {e2}")
            raise
    except Exception as e:
        logger.error(f"Unexpected error initializing FAISS directory: {e}")
        raise
    
    return index_dir


INDEX_DIR = get_faiss_index_dir()
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

    def debug_jobs(self) -> None:
        """Debug method to print all jobs in the vector store"""
        if not self.vs:
            print("No vector store loaded.")
            return
        
        print(f"Total documents in store: {len(self.vs.docstore._dict)}")
        for doc_id, doc in self.vs.docstore._dict.items():
            print(f"Doc ID: {doc_id}")
            print(f"Title: {doc.metadata.get('title', 'N/A')}")
            print(f"Recruiter ID: {doc.metadata.get('recruiterId', 'N/A')}")
            print(f"Content preview: {doc.page_content[:200]}...")
            print("---")

    async def build_or_load(self, db: Prisma) -> None:
        logger.info(f"=== FAISS build_or_load started ===")
        logger.info(f"Current state - _loaded: {self._loaded}, vs exists: {self.vs is not None}")
        
        # Check if already loaded and index exists
        if self._loaded and self.vs:
            logger.info("FAISS index already loaded in memory, force rebuilding to get latest jobs.")
            self._loaded = False
            self.vs = None

        # Ensure directory exists
        try:
            Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ FAISS index directory verified: {INDEX_DIR}")
            
            # List contents if directory exists
            contents = os.listdir(INDEX_DIR)
            logger.info(f"✓ Directory contents: {contents}")
        except Exception as e:
            logger.error(f"✗ Failed to ensure FAISS directory exists: {e}")
            raise

        logger.info("📥 Fetching jobs from database...")
        jobs = await db.job.find_many()
        logger.info(f"✓ Found {len(jobs)} total jobs in database")
        
        if jobs:
            logger.info(f"Sample jobs: {[{'id': j.id, 'title': j.title, 'recruiterId': j.recruiterId} for j in jobs[:3]]}")
        
        docs: List[Document] = [self._job_to_document(JobRow(
            id=j.id,
            title=j.title,
            description=j.description,
            recruiterId=j.recruiterId,
            location=getattr(j, "location", None),
            type=getattr(j, "type", None)
        )) for j in jobs]

        logger.info(f"✓ Created {len(docs)} documents from jobs")

        if not docs:
            logger.warning("⚠️  No jobs found to index. Creating placeholder index.")
            self.vs = FAISS.from_texts(["NO_JOBS"], self.embeddings, metadatas=[{"placeholder": True}])
        else:
            logger.info("🔨 Building FAISS index from documents...")
            self.vs = FAISS.from_documents(docs, self.embeddings)
            logger.info(f"✓ Successfully indexed {len(docs)} job documents")
            
            # Debug output
            logger.info(f"✓ Vector store docstore size: {len(self.vs.docstore._dict)}")
            for i, (doc_id, doc) in enumerate(list(self.vs.docstore._dict.items())[:5]):
                logger.info(f"  Sample doc {i}: ID={doc_id}, title={doc.metadata.get('title')}, recruiter={doc.metadata.get('recruiterId')}")

        # Save the index with proper error handling
        logger.info(f"💾 Attempting to save FAISS index to {INDEX_DIR}...")
        try:
            self.vs.save_local(INDEX_DIR)
            logger.info(f"✓ FAISS index saved successfully to {INDEX_DIR}")
            
            # Verify saved files
            saved_contents = os.listdir(INDEX_DIR)
            logger.info(f"✓ Saved files in directory: {saved_contents}")
        except PermissionError as e:
            logger.error(f"✗ Permission denied saving FAISS index to {INDEX_DIR}: {e}")
            logger.error(f"  Please run: sudo chown -R $(whoami) {INDEX_DIR}")
            logger.warning("  Index exists in memory but could not be persisted.")
        except Exception as e:
            logger.error(f"✗ Failed to save FAISS index: {e}")
            import traceback
            logger.error(traceback.format_exc())
            logger.warning("  Index exists in memory but could not be persisted.")
        
        self._loaded = True
        logger.info(f"=== FAISS build_or_load completed, _loaded={self._loaded} ===")
    
    def retrieve_jobs(self, query_text: str, recruiter_id: str, k: int = TOP_K) -> List[Document]:
        if not self.vs:
            return []
        docs = self.vs.similarity_search(query_text, k=k)
        return [d for d in docs if d.metadata.get("recruiterId") == recruiter_id]

    def retrieve_jobs_with_scores(self, query_text: str, recruiter_id: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """Retrieve jobs with similarity scores using FAISS embeddings"""
        if not self.vs:
            return []
        
        results = self.vs.similarity_search_with_score(query_text, k=k)
        
        scored_jobs = []
        for doc, score in results:
            if doc.metadata.get("recruiterId") == recruiter_id:
                similarity_score = float((1 - min(float(score), 1.0)) * 100)
                scored_jobs.append({
                    'title': doc.metadata.get('title'),
                    'match_score': similarity_score,
                    'document': doc
                })
        
        return scored_jobs


class JobRecommendationService:
    """
    Job recommendation using LangChain LLM for similarity scoring
    """
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.vstore = JobVectorStore.get(self.embeddings)
        
        self.scoring_prompt = PromptTemplate(
            input_variables=["employee_profile", "jobs_list"],
            template="""
You are an expert job matching AI. Analyze how well each of the following jobs matches the employee's profile.

Employee Profile: {employee_profile}

Jobs:
{jobs_list}

Provide match scores for each job in JSON format, using the exact title as key and integer score (0-100) as value:
{{"Title of Job 1": 85, "Title of Job 2": 70, ...}}

100 is perfect match. Consider skills, experience, education, department, position alignment.

Respond ONLY with the JSON object.
"""
        )
        self.scoring_chain = LLMChain(llm=self.llm, prompt=self.scoring_prompt)
    
    def _extract_features(self, employee_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and preprocess features from employee data"""
        features = {}
        
        bio = employee_info.get('bio', '')
        skills_list = employee_info.get('skills', [])
        skills_names = []
        for skill in skills_list:
            if isinstance(skill, dict):
                skills_names.append(skill.get('name', str(skill)))
            else:
                skills_names.append(str(skill))
        skills = ' '.join([s.lower() for s in skills_names])
        
        education_list = employee_info.get('education', [])
        education_names = []
        for edu in education_list:
            if isinstance(edu, dict):
                education_names.append(edu.get('name', str(edu)))
            else:
                education_names.append(str(edu))
        education = ' '.join([e.lower() for e in education_names if e])
        
        experience_list = employee_info.get('experience', [])
        experience_names = []
        for exp in experience_list:
            if isinstance(exp, dict):
                experience_names.append(exp.get('name', str(exp)))
            else:
                experience_names.append(str(exp))
        experience = ' '.join([e.lower() for e in experience_names if e])
        
        features['text_data'] = f"{bio} {skills} {education} {experience}"
        features['text_data'] = re.sub(r'[^\w\s]', '', features['text_data'])
        features['text_data'] = re.sub(r'\s+', ' ', features['text_data']).strip().lower()
        
        department_list = employee_info.get('department', [])
        department_names = []
        for dept in department_list:
            if isinstance(dept, dict):
                department_names.append(dept.get('name', str(dept)))
            else:
                department_names.append(str(dept))
        features['department'] = ' '.join([d.lower() for d in department_names])
        
        position_list = employee_info.get('position', [])
        position_names = []
        for pos in position_list:
            if isinstance(pos, dict):
                position_names.append(pos.get('name', str(pos)))
            else:
                position_names.append(str(pos))
        features['position'] = ' '.join([p.lower() for p in position_names])
        
        features['skills_list'] = [s.lower() for s in skills_names]
        
        features['employee_profile'] = (
            f"Name: {employee_info.get('firstName', '')} {employee_info.get('lastName', '')}\n"
            f"Bio: {employee_info.get('bio', '')}\n"
            f"Skills: {', '.join(skills_names)}\n"
            f"Education: {', '.join(education_names)}\n"
            f"Experience: {', '.join(experience_names)}\n"
            f"Position: {', '.join(position_names)}\n"
            f"Department: {', '.join(department_names)}\n"
            f"Salary Expectation: {employee_info.get('salary', 'Not specified')}"
        )
        
        features['optimized_query'] = (
            f"skills: {skills} department: {features['department']} position: {features['position']} "
            f"experience: {experience} education: {education}"
        ).strip()
        
        return features

    def _calculate_similarity(self, employee_features: Dict[str, Any], job_docs: List[Document], emb_score_dict: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate similarity scores using batch LLM"""
        if not job_docs:
            return []
        
        jobs_list = "\n".join([
            f"{i+1}. Title: {doc.metadata['title']}\nDescription: {doc.page_content}" 
            for i, doc in enumerate(job_docs)
        ])
        
        llm_score_dict = {}
        try:
            llm_response = self.scoring_chain.run(
                employee_profile=employee_features['employee_profile'],
                jobs_list=jobs_list
            )
            parsed = json.loads(llm_response.strip())
            llm_score_dict = {k: float(v) for k, v in parsed.items() if isinstance(v, (int, float))}
            logger.info("Batch LLM scoring successful.")
        except Exception as e:
            logger.warning(f"Batch LLM scoring failed: {e}. Using default scores.")
            llm_score_dict = {doc.metadata['title']: 50.0 for doc in job_docs}
        
        enhanced_scores = []
        for doc in job_docs:
            title = doc.metadata['title']
            embedding_score = float(emb_score_dict.get(title, 0.0))
            llm_score = llm_score_dict.get(title, 50.0)
            final_score = 0.3 * embedding_score + 0.7 * llm_score
            enhanced_scores.append({
                'id': doc.metadata['id'],
                'title': title,
                'match_score': min(100.0, max(0.0, round(final_score, 2))),
                'document': doc
            })
        
        return enhanced_scores

    async def recommend_jobs_for_employee(self, user_id: str, recruiter_id: str) -> List[Dict[str, Any]]:
        logger.info(f"=== Recommendation started for user_id={user_id}, recruiter_id={recruiter_id} ===")
        
        db = Prisma()
        await db.connect()
        
        try:
            logger.info("📊 Building/loading FAISS vector store...")
            await self.vstore.build_or_load(db)
            logger.info(f"✓ FAISS store loaded, _loaded={self.vstore._loaded}")
            
            # Fetch user data
            logger.info(f"👤 Fetching user data for user_id={user_id}...")
            user = await db.user.find_unique(
                where={"id": user_id},
                include={"employee": True}
            )
            
            if not user:
                logger.error(f"✗ User not found: {user_id}")
                raise ValueError("User not found")
            
            logger.info(f"✓ User found: {user.firstName} {user.lastName}")

            employee_info = {
                "firstName": user.firstName or "",
                "lastName": user.lastName or "",
                "bio": getattr(user.employee, "bio", "") if user.employee else "",
                "skills": getattr(user.employee, "skills", []) if user.employee else [],
                "education": getattr(user.employee, "education", []) if user.employee else [],
                "experience": getattr(user.employee, "experience", []) if user.employee else [],
                "position": user.position or [],
                "department": user.department or [],
                "salary": user.salary or ""
            }

            logger.info(f"✓ Employee info extracted: skills={len(employee_info['skills'])}, education={len(employee_info['education'])}")

            employee_features = self._extract_features(employee_info)
            logger.info(f"✓ Features extracted. Optimized query: {employee_features['optimized_query'][:100]}...")
            
            query_text = employee_features['optimized_query']
            logger.info(f"🔍 Retrieving jobs with FAISS for recruiter_id={recruiter_id}...")
            embedding_scored = self.vstore.retrieve_jobs_with_scores(query_text, recruiter_id, k=50)
            logger.info(f"✓ Retrieved {len(embedding_scored)} jobs with embedding scores")
            
            if embedding_scored:
                logger.info(f"  Top 3 matches: {[(j['title'], j['match_score']) for j in embedding_scored[:3]]}")
            
            if not embedding_scored:
                logger.warning(f"⚠️  No jobs retrieved for recruiter_id={recruiter_id}")
                logger.info(f"  Total jobs in store: {len(self.vstore.vs.docstore._dict) if self.vstore.vs else 0}")
                return []
            
            # Deduplicate
            title_groups = defaultdict(list)
            for item in embedding_scored:
                title_groups[item['title']].append(item)
            
            unique_items = []
            emb_score_dict = {}
            for title, group in title_groups.items():
                best_item = max(group, key=lambda x: x['match_score'])
                unique_items.append(best_item)
                emb_score_dict[title] = best_item['match_score']
            
            logger.info(f"✓ Deduplicated to {len(unique_items)} unique jobs")
            
            unique_docs = [item['document'] for item in unique_items]
            
            # Calculate similarity
            logger.info("🤖 Calculating LLM similarity scores...")
            try:
                recommended_jobs = self._calculate_similarity(employee_features, unique_docs, emb_score_dict)
                logger.info(f"✓ LLM scoring completed for {len(recommended_jobs)} jobs")
            except Exception as e:
                logger.warning(f"⚠️  LLM similarity failed: {e}")
                recommended_jobs = [
                    {
                        'id': item['document'].metadata['id'],
                        'title': item['title'],
                        'match_score': item['match_score'],
                        'document': item['document']
                    }
                    for item in unique_items
                ]
                logger.info(f"  Fallback to embedding scores for {len(recommended_jobs)} jobs")
            
            # Fetch final job details
            ids = [j['id'] for j in recommended_jobs]
            logger.info(f"📋 Fetching job details for {len(ids)} jobs...")
            
            if not ids:
                logger.warning("⚠️  No job IDs to fetch")
                return []
                
            final_jobs = await db.job.find_many(
                where={"id": {"in": ids}},
                include={"recruiter": True}
            )
            
            logger.info(f"✓ Fetched details for {len(final_jobs)} jobs")
            
            # Merge with scores
            final_jobs_with_scores = []
            for job in final_jobs:
                score_data = next((r for r in recommended_jobs if r['id'] == job.id), None)
                if score_data:
                    final_jobs_with_scores.append({
                        "id": job.id,
                        "title": job.title,
                        "description": job.description,
                        "recruiterId": job.recruiterId,
                        "location": job.location,
                        "type": job.type,
                        "match_score": float(score_data['match_score']),
                        "salary": job.salary,
                        "recruiter": {
                            "id": job.recruiter.id if job.recruiter else None,
                            "firstName": job.recruiter.firstName if job.recruiter else None,
                            "lastName": job.recruiter.lastName if job.recruiter else None,
                        }
                    })
            
            final_jobs_with_scores.sort(key=lambda x: x["match_score"], reverse=True)
            logger.info(f"✓ Final recommendations: {len(final_jobs_with_scores)} jobs")
            if final_jobs_with_scores:
                logger.info(f"  Top 3: {[(j['title'], j['match_score']) for j in final_jobs_with_scores[:3]]}")
            
            logger.info(f"=== Recommendation completed successfully ===")
            return final_jobs_with_scores
            
        except Exception as e:
            logger.error(f"✗ Error in job recommendation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            await db.disconnect()