import os
import json
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
import re
from collections import defaultdict

from prisma import Prisma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger(__name__)

# Enhanced directory resolution with production fallbacks
EMPLOYEE_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
SERVICES_DIR = os.path.dirname(EMPLOYEE_SERVICES_DIR)
PROJECT_ROOT = os.path.dirname(SERVICES_DIR)

def get_index_dir() -> str:
    """Get appropriate index directory with robust fallbacks for production"""
    # First, try environment variable
    env_dir = os.getenv("JOBS_FAISS_DIR")
    if env_dir:
        return env_dir
    
    # Check if we're in production environment
    is_production = os.getenv("ENVIRONMENT") == "production" or os.getenv("NODE_ENV") == "production"
    
    if is_production:
        # Production-specific directory options with fallbacks
        production_paths = [
            os.path.join(PROJECT_ROOT, "faiss_index"),  # Project directory
            "/tmp/faiss_index",  # System temp (usually writable)
            "/var/tmp/faiss_index",  # Persistent temp
            os.path.expanduser("~/faiss_index"),  # User home directory
        ]
        
        for path in production_paths:
            try:
                os.makedirs(path, exist_ok=True, mode=0o755)
                # Test write permission
                test_file = os.path.join(path, ".write_test")
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                print(f"DEBUG: Using production directory: {path}")
                return path
            except Exception as e:
                print(f"DEBUG: Directory {path} not usable: {e}")
                continue
        
        # Last resort: current working directory
        cwd_path = os.path.join(os.getcwd(), "faiss_index")
        os.makedirs(cwd_path, exist_ok=True, mode=0o755)
        print(f"DEBUG: Using current directory as fallback: {cwd_path}")
        return cwd_path
    
    # Development: use project root
    return os.path.join(PROJECT_ROOT, "faiss_index")

INDEX_DIR = get_index_dir()
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

    def _ensure_directory_access(self) -> bool:
        """Ensure directory exists and is accessible with detailed debugging"""
        print(f"DEBUG: Ensuring FAISS index directory: {INDEX_DIR}")
        print(f"DEBUG: Current working directory: {os.getcwd()}")
        print(f"DEBUG: User: {os.getenv('USER', 'unknown')}")
        
        try:
            # Create directory if it doesn't exist
            if not os.path.exists(INDEX_DIR):
                print(f"DEBUG: Creating directory: {INDEX_DIR}")
                os.makedirs(INDEX_DIR, mode=0o755, exist_ok=True)
            
            # Verify directory exists and is accessible
            if not os.path.exists(INDEX_DIR):
                raise OSError(f"Failed to create directory: {INDEX_DIR}")
            
            if not os.path.isdir(INDEX_DIR):
                raise OSError(f"Path exists but is not a directory: {INDEX_DIR}")
            
            # Check permissions
            if not os.access(INDEX_DIR, os.W_OK):
                raise OSError(f"Directory not writable: {INDEX_DIR}")
            if not os.access(INDEX_DIR, os.X_OK):
                raise OSError(f"Directory not executable: {INDEX_DIR}")
            
            # Test write capability
            test_file = os.path.join(INDEX_DIR, ".perm_test")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
            except Exception as e:
                raise OSError(f"Cannot write to directory: {e}")
            
            print(f"DEBUG: Directory validated successfully: {INDEX_DIR}")
            print(f"DEBUG: Directory permissions: {oct(os.stat(INDEX_DIR).st_mode)[-3:]}")
            return True
            
        except Exception as e:
            print(f"ERROR: Directory validation failed: {e}")
            logger.error(f"Directory validation failed: {e}")
            return False

    async def build_or_load(self, db: Prisma) -> None:
        print(f"DEBUG: INDEX_DIR resolved to: {INDEX_DIR}")
        print(f"DEBUG: PROJECT_ROOT: {PROJECT_ROOT}")
        
        # First, ensure directory access
        if not self._ensure_directory_access():
            # Try fallback to /tmp if primary directory fails
            global INDEX_DIR
            original_dir = INDEX_DIR
            INDEX_DIR = "/tmp/faiss_index"
            print(f"DEBUG: Trying fallback directory: {INDEX_DIR}")
            
            if not self._ensure_directory_access():
                raise OSError(f"Failed to access both {original_dir} and fallback {INDEX_DIR}")
        
        # Optional: Load from disk if exists and not forcing rebuild
        index_file = os.path.join(INDEX_DIR, "index.faiss")
        if os.path.exists(index_file) and self.vs is None:
            try:
                print(f"DEBUG: Loading existing FAISS index from: {index_file}")
                self.vs = FAISS.load_local(INDEX_DIR, self.embeddings, allow_dangerous_deserialization=True)
                self._loaded = True
                logger.info(f"FAISS index loaded from {INDEX_DIR}.")
                print("DEBUG: Successfully loaded existing FAISS index")
                return
            except Exception as load_e:
                print(f"DEBUG: Failed to load existing FAISS index: {load_e}. Rebuilding...")
                logger.warning(f"Failed to load existing FAISS index: {load_e}. Rebuilding...")

        # Check if we should force rebuild
        FORCE_REBUILD = os.getenv("FORCE_FAISS_REBUILD", "false").lower() == "true"
        print(f"DEBUG: FORCE_REBUILD: {FORCE_REBUILD}, _loaded: {self._loaded}, vs exists: {self.vs is not None}")
        
        if self._loaded and self.vs and not FORCE_REBUILD:
            print("DEBUG: Using existing in-memory FAISS index.")
            logger.info("Using existing in-memory FAISS index.")
            return

        print("DEBUG: Building/rebuilding FAISS index from database...")
        logger.info("Building/rebuilding FAISS index from database...")
        
        try:
            jobs = await db.job.find_many()
            print(f"DEBUG: Found {len(jobs)} jobs for indexing.")
            logger.info(f"Found {len(jobs)} jobs for indexing.")
            
            docs: List[Document] = [self._job_to_document(JobRow(
                id=j.id,
                title=j.title,
                description=j.description,
                recruiterId=j.recruiterId,
                location=getattr(j, "location", None),
                type=getattr(j, "type", None)
            )) for j in jobs]

            if not docs:
                print("DEBUG: No jobs found, creating placeholder index.")
                logger.warning("No jobs found to index. Creating placeholder index.")
                self.vs = FAISS.from_texts(["NO_JOBS"], self.embeddings, metadatas=[{"placeholder": True}])
            else:
                self.vs = FAISS.from_documents(docs, self.embeddings)
                print(f"DEBUG: Indexed {len(docs)} job documents.")
                logger.info(f"Indexed {len(docs)} job documents.")
                # Debug: print jobs after building
                print("DEBUG: Jobs indexed from database:")
                self.debug_jobs()

            # Save the index
            print(f"DEBUG: Saving index to {INDEX_DIR}")
            try:
                self.vs.save_local(INDEX_DIR)
                print(f"DEBUG: FAISS index saved successfully to {INDEX_DIR}.")
                logger.info(f"FAISS index saved successfully to {INDEX_DIR}.")
                
                # Verify the files were created
                expected_files = ["index.faiss", "index.pkl"]
                for file in expected_files:
                    file_path = os.path.join(INDEX_DIR, file)
                    if os.path.exists(file_path):
                        print(f"DEBUG: Created file: {file_path} (size: {os.path.getsize(file_path)} bytes)")
                    else:
                        print(f"WARNING: Expected file not found: {file_path}")
                        
            except Exception as save_error:
                print(f"ERROR: Failed to save FAISS index: {save_error}")
                logger.error(f"Failed to save FAISS index: {save_error}")
                # Don't re-raise, keep the in-memory index
       
        except Exception as e:
            print(f"ERROR: Failed to build FAISS index: {e}")
            logger.error(f"Failed to build FAISS index: {e}")
            import traceback
            logger.error(traceback.format_exc())
            # Fallback: Create minimal in-memory index if possible
            try:
                self.vs = FAISS.from_texts(["INDEX_BUILD_FAILED"], self.embeddings, metadatas=[{"error": str(e)}])
                print("DEBUG: Fallback in-memory index created.")
                logger.info("Fallback in-memory index created.")
            except Exception as fallback_e:
                print(f"CRITICAL: Fallback index also failed: {fallback_e}")
                logger.critical(f"Fallback index also failed: {fallback_e}")
                self.vs = None
            raise  # Re-raise to alert in prod
        
        self._loaded = True
        print(f"DEBUG: Build complete, _loaded set to True, vs: {self.vs is not None}")
    
    def retrieve_jobs(self, query_text: str, recruiter_id: str, k: int = TOP_K) -> List[Document]:
        if not self.vs:
            print("DEBUG: No vector store for retrieve_jobs.")
            return []
        # Retrieve top-K relevant jobs from FAISS
        docs = self.vs.similarity_search(query_text, k=k)
        # Filter by recruiterId
        return [d for d in docs if d.metadata.get("recruiterId") == recruiter_id]

    def retrieve_jobs_with_scores(self, query_text: str, recruiter_id: str, k: int = TOP_K) -> List[Dict[str, Any]]:
        """Retrieve jobs with similarity scores using FAISS embeddings"""
        if not self.vs:
            print("DEBUG: No vector store for retrieve_jobs_with_scores.")
            return []
        
        # Search with scores for top k relevant
        results = self.vs.similarity_search_with_score(query_text, k=k)
        
        # Filter by recruiter and format
        scored_jobs = []
        for doc, score in results:
            if doc.metadata.get("recruiterId") == recruiter_id:
                # Convert distance to similarity score (0-100), ensure float
                similarity_score = float((1 - min(float(score), 1.0)) * 100)
                scored_jobs.append({
                    'title': doc.metadata.get('title'),
                    'match_score': similarity_score,
                    'document': doc
                })
        
        print(f"DEBUG: Retrieved {len(scored_jobs)} scored jobs for recruiter {recruiter_id}.")
        return scored_jobs


class JobRecommendationService:
    """
    Job recommendation using LangChain LLM for similarity scoring
    """
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        self.vstore = JobVectorStore.get(self.embeddings)
        
        # Prompt template for batch LLM scoring
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
        
        # Text features
        bio = employee_info.get('bio', '')
        # Robust extraction for skills: handle both dicts and strings
        skills_list = employee_info.get('skills', [])
        skills_names = []
        for skill in skills_list:
            if isinstance(skill, dict):
                skills_names.append(skill.get('name', str(skill)))
            else:
                skills_names.append(str(skill))
        skills = ' '.join([s.lower() for s in skills_names])
        
        # Handle arrays for education (robust)
        education_list = employee_info.get('education', [])
        education_names = []
        for edu in education_list:
            if isinstance(edu, dict):
                education_names.append(edu.get('name', str(edu)))
            else:
                education_names.append(str(edu))
        education = ' '.join([e.lower() for e in education_names if e])
        
        # Handle arrays for experience (robust)
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
        
        # Handle array fields - convert to strings (robust)
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
        
        # Create a formatted employee profile for LLM (use extracted names)
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
        
        # Optimized query for FAISS retrieval focusing on skills, department, position
        features['optimized_query'] = (
            f"skills: {skills} department: {features['department']} position: {features['position']} "
            f"experience: {experience} education: {education}"
        ).strip()
        
        return features

    def _calculate_similarity(self, employee_features: Dict[str, Any], job_docs: List[Document], emb_score_dict: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate similarity scores using batch LLM"""
        if not job_docs:
            return []
        
        # Batch LLM scoring
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
            # Parse JSON response
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
        print(f"DEBUG: Starting recommend_jobs_for_employee for user {user_id}, recruiter {recruiter_id}")
        db = Prisma()
        await db.connect()
        
        try:
            # Build/load FAISS store (this will create directory and save first time if needed)
            print("DEBUG: Calling build_or_load...")
            await self.vstore.build_or_load(db)
            print(f"DEBUG: After build_or_load, _loaded: {self.vstore._loaded}, vs: {self.vstore.vs is not None}")
            
            # Debug: print all jobs in store before recommendation
            print("DEBUG: All jobs in vector store before recommendation:")
            self.vstore.debug_jobs()
            
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
            print(f"DEBUG: Optimized query: {employee_features['optimized_query'][:200]}...")  # Truncated for log
            
            # Retrieve top relevant jobs using the optimized query for FAISS
            query_text = employee_features['optimized_query']
            embedding_scored = self.vstore.retrieve_jobs_with_scores(query_text, recruiter_id, k=50)
            
            if not embedding_scored:
                print("DEBUG: No embedding_scored results.")
                return []
            
            # Deduplicate by title: pick the best embedding score per title
            title_groups = defaultdict(list)
            for item in embedding_scored:
                title_groups[item['title']].append(item)
            
            unique_items = []
            emb_score_dict = {}
            for title, group in title_groups.items():
                best_item = max(group, key=lambda x: x['match_score'])
                unique_items.append(best_item)
                emb_score_dict[title] = best_item['match_score']
            
            unique_docs = [item['document'] for item in unique_items]
            print(f"DEBUG: Unique items after dedup: {len(unique_items)}")
            
            # Calculate similarity scores using LLM (batched)
            try:
                recommended_jobs = self._calculate_similarity(employee_features, unique_docs, emb_score_dict)
            except Exception as e:
                logger.warning(f"LLM similarity failed: {e}")
                # Fallback to embedding scores only
                recommended_jobs = [
                    {
                        'id': item['document'].metadata['id'],
                        'title': item['title'],
                        'match_score': item['match_score'],
                        'document': item['document']
                    }
                    for item in unique_items
                ]
            
            # Get final job details from database by ID
            ids = [j['id'] for j in recommended_jobs]
            if not ids:
                print("DEBUG: No IDs for final fetch.")
                return []
                
            final_jobs = await db.job.find_many(
                where={"id": {"in": ids}},
                include={"recruiter": True}
            )
            
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
            
            # Sort by score and return all relevant results
            final_jobs_with_scores.sort(key=lambda x: x["match_score"], reverse=True)
            print(f"DEBUG: Returning {len(final_jobs_with_scores)} recommended jobs.")
            return final_jobs_with_scores
            
        except Exception as e:
            print(f"ERROR: Error in job recommendation: {e}")
            logger.error(f"Error in job recommendation: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return []
        finally:
            await db.disconnect()