import os
import json
import time
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

# Tavily integration for external job search
from langchain_community.tools.tavily_search import TavilySearchResults

# FAISS directory path inside services folder
def get_faiss_index_dir() -> str:
    index_dir = os.getenv("FAISS_INDEX_PATH", os.path.join(os.path.dirname(__file__), "faiss_jobs_index"))
    os.makedirs(index_dir, exist_ok=True)
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
            print("❌ No vector store loaded.")
            return
        
        print(f"📊 Total documents in store: {len(self.vs.docstore._dict)}")
        for doc_id, doc in self.vs.docstore._dict.items():
            print(f"📄 Doc ID: {doc_id}")
            print(f"   Title: {doc.metadata.get('title', 'N/A')}")
            print(f"   Recruiter ID: {doc.metadata.get('recruiterId', 'N/A')}")
            print(f"   Content preview: {doc.page_content[:200]}...")
            print("---")

    async def build_or_load(self, db: Prisma) -> None:
        print(f"🚀 === FAISS build_or_load started ===")
        print(f"📝 Current state - _loaded: {self._loaded}, vs exists: {self.vs is not None}")
        print(f"📁 Using INDEX_DIR: {INDEX_DIR}")
        
        # Check if already loaded and index exists
        if self._loaded and self.vs:
            print("🔄 FAISS index already loaded in memory, force rebuilding to get latest jobs.")
            self._loaded = False
            self.vs = None

        # Create directory if it doesn't exist
        try:
            Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
            print(f"✅ FAISS directory ensured: {INDEX_DIR}")
            
            # Check write permissions
            test_file = os.path.join(INDEX_DIR, ".write_test")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✅ Write permissions verified for: {INDEX_DIR}")
        except PermissionError as e:
            print(f"❌ No write permission to FAISS directory: {INDEX_DIR}")
            print(f"💡 Please fix permissions:")
            print(f"   chmod 755 {INDEX_DIR}")
            raise
        except Exception as e:
            print(f"❌ Failed to create/verify FAISS directory: {e}")
            raise

        print("📥 Fetching jobs from database...")
        jobs = await db.job.find_many()
        print(f"✅ Found {len(jobs)} total jobs in database")
        
        if jobs:
            print(f"📋 Sample jobs: {[{'id': j.id, 'title': j.title, 'recruiterId': j.recruiterId} for j in jobs[:3]]}")
        
        docs: List[Document] = [self._job_to_document(JobRow(
            id=j.id,
            title=j.title,
            description=j.description,
            recruiterId=j.recruiterId,
            location=getattr(j, "location", None),
            type=getattr(j, "type", None)
        )) for j in jobs]

        print(f"✅ Created {len(docs)} documents from jobs")

        if not docs:
            print("⚠️ No jobs found to index. Creating placeholder index.")
            self.vs = FAISS.from_texts(["NO_JOBS"], self.embeddings, metadatas=[{"placeholder": True}])
        else:
            print("🔨 Building FAISS index from documents...")
            self.vs = FAISS.from_documents(docs, self.embeddings)
            print(f"✅ Successfully indexed {len(docs)} job documents")
            
            # Debug output
            print(f"✅ Vector store docstore size: {len(self.vs.docstore._dict)}")
            for i, (doc_id, doc) in enumerate(list(self.vs.docstore._dict.items())[:5]):
                print(f"  📝 Sample doc {i}: ID={doc_id}, title={doc.metadata.get('title')}, recruiter={doc.metadata.get('recruiterId')}")

        # Save the index
        print(f"💾 Saving INTERNAL jobs to FAISS index at {INDEX_DIR}...")
        try:
            self.vs.save_local(INDEX_DIR)
            print(f"✅ FAISS index saved successfully to {INDEX_DIR}")
            
            # Verify saved files
            saved_contents = os.listdir(INDEX_DIR)
            print(f"✅ Saved files in directory: {saved_contents}")
        except Exception as e:
            print(f"❌ Failed to save FAISS index: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            print("  ⚠️ Index exists in memory but could not be persisted.")
            raise
        
        self._loaded = True
        print(f"✅ === FAISS build_or_load completed, _loaded={self._loaded} ===")
    
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
        print("🔄 Initializing JobRecommendationService...")
        self.embeddings = OpenAIEmbeddings()
        print("✅ OpenAIEmbeddings initialized")
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        print("✅ ChatOpenAI initialized")
        self.vstore = JobVectorStore.get(self.embeddings)
        print("✅ JobVectorStore initialized")
        
        # Initialize Tavily for external search with higher max to slice later
        self.tavily = TavilySearchResults(max_results=20)
        print("✅ TavilySearchResults initialized")
        
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
        print("✅ LLMChain initialized")
        
        self.extract_prompt = PromptTemplate(
            input_variables=["jobs_text", "num_jobs"],
            template="""
Extract up to {num_jobs} structured job postings from the following web search results. For each job, infer and fill:

- title: The job title
- company: The name of the hiring company (if not found, use "Unknown Company")
- description: Brief description (up to 300 words)
- location: Location if mentioned, else "Remote/Undisclosed"
- type: Employment type (e.g., Full-time, Part-time, Contract) if mentioned, else "Full-time"
- industry: The industry of the job/company (infer from title/description if not explicit)
- url: The URL of the job posting (MUST be present in the text, do NOT invent one)
- salary: Salary range if mentioned, else "Not specified"

IMPORTANT:
- Extract ONLY jobs that are explicitly listed in the search results.
- If the text says "no results" or contains no job listings, return an empty list [].
- DO NOT invent, hallucinate, or create example jobs.
- DO NOT use "example.com", "examplejobposting.com", or any other placeholder URLs.
- Only return jobs found in the provided text.

Respond ONLY with a JSON array of objects: [{{"title": "...", "company": "...", "description": "...", "location": "...", "type": "...", "industry": "...", "salary": "...", "url": "..."}}, ...]
"""
        )
        print("✅ Extract prompt initialized")
    
    def _extract_features(self, employee_info: Dict[str, Any]) -> Dict[str, Any]:
        """Extract and preprocess features from employee data"""
        print("🔧 Extracting employee features...")
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
                # Use 'degree' or 'name' for education
                edu_name = edu.get('degree', edu.get('name', ''))
                if not edu_name:
                    edu_name = str(edu)
                education_names.append(edu_name)
            else:
                education_names.append(str(edu))
        education = ' '.join([e.lower() for e in education_names if e])
        features['education_str'] = ', '.join(education_names[:2])
        
        experience_list = employee_info.get('experience', [])
        experience_names = []
        for exp in experience_list:
            if isinstance(exp, dict):
                # Use 'position' for experience
                exp_name = exp.get('position', exp.get('title', ''))
                if not exp_name:
                    exp_name = str(exp)
                experience_names.append(exp_name)
            else:
                experience_names.append(str(exp))
        experience = ' '.join([e.lower() for e in experience_names if e])
        features['experience_str'] = ', '.join(experience_names[:3])
        
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
        
        print(f"✅ Features extracted. Query length: {len(features['optimized_query'])}")
        return features

    def _calculate_similarity(self, employee_features: Dict[str, Any], job_docs: List[Document], emb_score_dict: Dict[str, float]) -> List[Dict[str, Any]]:
        """Calculate similarity scores using batch LLM"""
        if not job_docs:
            print("⚠️ No job documents to calculate similarity for")
            return []
        
        print(f"🤖 Calculating LLM similarity for {len(job_docs)} jobs...")
        jobs_list = "\n".join([
            f"{i+1}. Title: {doc.metadata['title']}\nDescription: {doc.page_content}" 
            for i, doc in enumerate(job_docs)
        ])
        
        llm_score_dict = {}
        try:
            print("📤 Sending batch request to LLM...")
            llm_response = self.scoring_chain.run(
                employee_profile=employee_features['employee_profile'],
                jobs_list=jobs_list
            )
            print(f"📥 LLM response received: {llm_response[:200]}...")
            parsed = json.loads(llm_response.strip())
            llm_score_dict = {k: float(v) for k, v in parsed.items() if isinstance(v, (int, float))}
            print(f"✅ Batch LLM scoring successful. Scored {len(llm_score_dict)} jobs")
        except Exception as e:
            print(f"❌ Batch LLM scoring failed: {e}. Using default scores.")
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
        
        print(f"✅ Enhanced scoring completed for {len(enhanced_scores)} jobs")
        return enhanced_scores

    async def fetch_external_jobs(self, employee_features: Dict[str, Any], num_results: int = 5) -> List[Dict[str, Any]]:
        """
        Fetch external jobs using Tavily search based on dynamic user data.
        Uses LLM to extract structured info from search results.
        Returns a list of job dicts with basic info and a default match score.
        """
        print("🌐 === Fetching external jobs via Tavily ===")
        start_time = time.time()
        
        # Construct dynamic query using user data
        parts = ["latest job openings"]
        if pos := employee_features['position'].strip():
            parts.append(f"for {pos} roles")
        if dept := employee_features['department'].strip():
            parts.append(f"in {dept} department")
        if exp_str := employee_features.get('experience_str', '').strip():
            parts.append(f"for professionals with experience in {exp_str}")
        if edu_str := employee_features.get('education_str', '').strip():
            parts.append(f"preferring {edu_str} education")
        skills_str = ', '.join(employee_features['skills_list'][:5])
        if skills_str:
            parts.append(f"needing skills: {skills_str}")
        
        # Add keywords to encourage better results
        parts.append("with salary info hiring now apply")
        
        dynamic_query = ' '.join(parts)
        print(f"🔍 Dynamic Tavily query: {dynamic_query}")
        
        try:
            # Search using Tavily - only pass query
            raw_results = self.tavily.invoke({"query": dynamic_query})
            print(f"✅ Tavily returned {len(raw_results)} raw results")
            
            # Parse results if they are strings (handle potential JSON strings)
            parsed_results = []
            for item in raw_results:
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        # If not JSON, treat as content with default keys
                        item = {"title": "Unknown Job", "url": "", "content": item}
                if isinstance(item, dict):
                    parsed_results.append(item)
                else:
                    parsed_results.append({"title": "Unknown Job", "url": "", "content": str(item)})
            
            # Limit to num_results
            parsed_results = parsed_results[:num_results]
            print(f"✅ Parsed and limited to {len(parsed_results)} results")
            
            if not parsed_results:
                print("⚠️ No parsed results from Tavily")
                return []
            
            # Prepare text for extraction
            jobs_text = "\n---\n".join([
                f"Title: {r.get('title', 'N/A')}\nURL: {r.get('url', 'N/A')}\nContent: {r.get('content', '')[:1000]}"
                for r in parsed_results
            ])
            
            # Use LLM to extract structured jobs
            extract_response = self.llm.invoke(
                self.extract_prompt.format(
                    jobs_text=jobs_text,
                    num_jobs=len(parsed_results)
                )
            )
            
            external_jobs = []
            seen_urls = set()
            
            try:
                extracted = json.loads(extract_response.content.strip())
                if isinstance(extracted, list):
                    for i, job in enumerate(extracted):
                        # Try to get URL from LLM extraction
                        url = job.get('url')
                        
                        # If URL missing, try to match with parsed_results by index if possible
                        if not url and i < len(parsed_results):
                            url = parsed_results[i].get('url', '')
                        
                        # Normalize URL
                        if url:
                            url = url.strip()
                            
                        # Strict URL Validation
                        if url and ("example.com" in url or "examplejobposting.com" in url or "companywebsite.com" in url or "jobposting3.com" in url):
                            print(f"⚠️ Detected hallucinated URL: {url}. Skipping.")
                            url = "" # Clear invalid URL so we might fallback to parsed result
                            
                        # If URL missing or invalid, try to match with parsed_results by index if possible
                        if not url and i < len(parsed_results):
                            url = parsed_results[i].get('url', '')
                            
                        # Skip if we already have this URL (dedup)
                        if url and url in seen_urls:
                            continue
                            
                        external_jobs.append({
                            "id": f"external_{hash(url) if url else i}_{i}",
                            "title": job.get('title', f"Job Opportunity {i+1}"),
                            "company": job.get('company', 'Unknown Company'),
                            "description": job.get('description', 'No description available'),
                            "location": job.get('location', 'Remote/Undisclosed'),
                            "type": job.get('type', 'Full-time'),
                            "industry": job.get('industry', 'Various'),
                            "salary": job.get('salary', 'Not specified'),
                            "source_url": url,
                            "recruiterId": "external",
                            "match_score": 70.0,  # Temporary, will be overridden
                            "recruiter": None
                        })
                        if url:
                            seen_urls.add(url)
                            
                    print(f"✅ LLM extracted {len(external_jobs)} structured jobs")
                else:
                    raise ValueError("Invalid extraction format")
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                print(f"⚠️ LLM extraction failed/incomplete: {e}. Proceeding to fallback.")

            # Fallback: Fill up to num_results with remaining parsed_results
            if len(external_jobs) < num_results:
                print(f"⚠️ Only have {len(external_jobs)} jobs, need {num_results}. Filling from raw results...")
                for i, result in enumerate(parsed_results):
                    if len(external_jobs) >= num_results:
                        break
                        
                    url = result.get('url', '').strip()
                    if url and url in seen_urls:
                        continue
                        
                    title = result.get('title', f"Job Opportunity {i+1}")
                    desc = result.get('content', '')[:400] + "..." if result.get('content') else 'No description available'
                    
                    external_jobs.append({
                        "id": f"external_{hash(url) if url else i}_{i}_fallback",
                        "title": title,
                        "company": "Unknown Company",
                        "description": desc,
                        "location": "Remote/Undisclosed",
                        "type": "Full-time",
                        "industry": "Various",
                        "salary": "Not specified",
                        "source_url": url,
                        "recruiterId": "external",
                        "match_score": 70.0,
                        "recruiter": None
                    })
                    if url:
                        seen_urls.add(url)
                    print(f"  📝 Fallback job added: {title}")
            
            elapsed_time = time.time() - start_time
            print(f"✅ External fetch completed in {elapsed_time:.2f}s with {len(external_jobs)} jobs")
            return external_jobs
            
        except Exception as e:
            print(f"❌ External job fetch failed: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return []

    async def recommend_jobs_for_employee(self, user_id: str, recruiter_id: str, include_external: bool = True) -> List[Dict[str, Any]]:
        print(f"🎯 === Recommendation started for user_id={user_id}, recruiter_id={recruiter_id}, include_external={include_external} ===")
        start_time = time.time()
        
        db = Prisma()
        await db.connect()
        
        try:
            # 1. Fetch user data first (Required for both internal and external)
            print(f"👤 Fetching user data for user_id={user_id}...")
            user = await db.user.find_unique(
                where={"id": user_id},
                include={"employee": True}
            )
            
            if not user:
                print(f"❌ User not found: {user_id}")
                raise ValueError("User not found")
            
            print(f"✅ User found: {user.firstName} {user.lastName}")

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

            print(f"✅ Employee info extracted: skills={len(employee_info['skills'])}, education={len(employee_info['education'])}")

            employee_features = self._extract_features(employee_info)
            print(f"✅ Features extracted. Optimized query: {employee_features['optimized_query'][:100]}...")
            
            # 2. Internal jobs: embedding retrieval and dedup (Best Effort)
            internal_docs = []
            internal_unique_items = []
            emb_score_dict = {}
            
            try:
                print("📊 Building/loading FAISS vector store...")
                await self.vstore.build_or_load(db)
                print(f"✅ FAISS store loaded, _loaded={self.vstore._loaded}")
                
                query_text = employee_features['optimized_query']
                print(f"🔍 Retrieving internal jobs with FAISS for recruiter_id={recruiter_id}...")
                embedding_scored = self.vstore.retrieve_jobs_with_scores(query_text, recruiter_id, k=50)
                print(f"✅ Retrieved {len(embedding_scored)} internal jobs with embedding scores")
                
                if embedding_scored:
                    print(f"  📊 Top 3 internal embedding matches: {[(j['title'], j['match_score']) for j in embedding_scored[:3]]}")
                    
                    # Deduplicate
                    print("🔄 Deduplicating internal jobs...")
                    title_groups = defaultdict(list)
                    for item in embedding_scored:
                        title_groups[item['title']].append(item)
                    
                    for title, group in title_groups.items():
                        best_item = max(group, key=lambda x: x['match_score'])
                        internal_unique_items.append(best_item)
                        emb_score_dict[title] = best_item['match_score']
                    
                    print(f"✅ Deduplicated to {len(internal_unique_items)} unique internal jobs")
                
                internal_docs = [item['document'] for item in internal_unique_items]
                
            except Exception as e:
                print(f"⚠️ Internal job search failed: {e}")
                import traceback
                print(f"⚠️ Traceback: {traceback.format_exc()}")
                # Continue without internal jobs
            
            # 3. External jobs
            external_jobs = []
            external_docs = []
            if include_external:
                try:
                    external_jobs = await self.fetch_external_jobs(employee_features, num_results=10)
                    print(f"✅ Fetched {len(external_jobs)} external jobs")
                    
                    if external_jobs:
                        print(f"  🌐 Sample external: {[(j['title'], j['match_score']) for j in external_jobs[:2]]}")
                        
                        # Create Documents for external jobs for unified scoring
                        # NOTE: These are NOT saved to FAISS, only used for temporary scoring in memory.
                        # The FAISS index is strictly for internal jobs from the database.
                        print("ℹ️ Processing external jobs in-memory (NOT saving to FAISS)...")
                        for j in external_jobs:
                            content = (
                                f"Title: {j['title']}\n"
                                f"Description: {j['description'] or ''}\n"
                                f"Location: {j['location'] or ''}\n"
                                f"Type: {j['type'] or ''}\n"
                                f"RecruiterId: external"
                            )
                            doc = Document(
                                page_content=content,
                                metadata={
                                    **{k: v for k, v in j.items() if k != 'match_score'},
                                    'is_external': True
                                }
                            )
                            external_docs.append(doc)
                            emb_score_dict[j['title']] = 70.0  # Default embedding score for external
                except Exception as e:
                    print(f"⚠️ External job search failed: {e}")
                    # Continue without external jobs
            
            # 4. Unified scoring for all (internal + external)
            all_docs = internal_docs + external_docs
            
            if not all_docs:
                print("⚠️ No jobs found (internal or external). Returning empty list.")
                return []

            print(f"🤖 Calculating unified LLM similarity for {len(all_docs)} total jobs (internal: {len(internal_docs)}, external: {len(external_docs)})...")
            try:
                all_scored = self._calculate_similarity(employee_features, all_docs, emb_score_dict)
                print(f"✅ Unified scoring completed for {len(all_scored)} jobs")
            except Exception as e:
                print(f"⚠️ Unified similarity failed: {e}")
                # Fallback: use embedding/default scores
                all_scored = [
                    {
                        'id': doc.metadata['id'],
                        'title': doc.metadata['title'],
                        'match_score': emb_score_dict.get(doc.metadata['title'], 50.0),
                        'document': doc
                    }
                    for doc in all_docs
                ]
                print(f"  🔄 Fallback scores for {len(all_scored)} jobs")
            
            # Separate internal and external scored
            internal_scored = [s for s in all_scored if not s['document'].metadata.get('is_external', False)]
            external_scored = [s for s in all_scored if s['document'].metadata.get('is_external', False)]
            
            # Process internal: fetch details from DB
            internal_final = []
            if internal_scored:
                try:
                    internal_ids = [s['id'] for s in internal_scored]
                    print(f"📋 Fetching internal job details for {len(internal_ids)} jobs...")
                    final_jobs = await db.job.find_many(
                        where={"id": {"in": internal_ids}},
                        include={"recruiter": True}
                    )
                    print(f"✅ Fetched details for {len(final_jobs)} internal jobs")
                    
                    # Merge with scores
                    for job in final_jobs:
                        score_data = next((r for r in internal_scored if r['id'] == job.id), None)
                        if score_data:
                            internal_final.append({
                                "id": job.id,
                                "title": job.title,
                                "description": job.description,
                                "recruiterId": job.recruiterId,
                                "location": job.location,
                                "type": job.type,
                                "match_score": float(score_data['match_score']),
                                "salary": job.salary,
                                "source_url": None,
                                "recruiter": {
                                    "id": job.recruiter.id if job.recruiter else None,
                                    "firstName": job.recruiter.firstName if job.recruiter else None,
                                    "lastName": job.recruiter.lastName if job.recruiter else None,
                                }
                            })
                except Exception as e:
                    print(f"⚠️ Failed to fetch details for internal jobs: {e}")
            
            # Process external: use extracted data
            external_final = []
            for s in external_scored:
                doc_meta = s['document'].metadata
                external_final.append({
                    "id": doc_meta['id'],
                    "title": doc_meta['title'],
                    "company": doc_meta.get('company', 'Unknown Company'),
                    "description": doc_meta['description'],
                    "recruiterId": "external",
                    "location": doc_meta.get('location', 'Remote/Undisclosed'),
                    "type": doc_meta.get('type', 'Full-time'),
                    "industry": doc_meta.get('industry', 'Various'),
                    "match_score": float(s['match_score']),
                    "salary": doc_meta.get('salary', 'Not specified'),
                    "source_url": doc_meta.get('source_url', ''),
                    "recruiter": None
                })
            
            # Combine and sort
            all_jobs = internal_final + external_final
            all_jobs.sort(key=lambda x: x["match_score"], reverse=True)
            
            print(f"✅ Final recommendations: {len(all_jobs)} jobs (internal: {len(internal_final)}, external: {len(external_final)})")
            if all_jobs:
                print(f"  🏆 Top 3 overall: {[(j['title'], j['match_score']) for j in all_jobs[:3]]}")
            
            elapsed_time = time.time() - start_time
            print(f"✅ === Recommendation completed successfully in {elapsed_time:.2f}s ===")
            return all_jobs
            
        except Exception as e:
            print(f"❌ Error in job recommendation: {e}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return []
        finally:
            await db.disconnect()
            print("✅ Database connection closed")