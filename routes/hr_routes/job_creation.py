# routes/hr_routes/job_creation.py - Complete job creation and description generation routes
from fastapi import APIRouter, HTTPException, status, Body
from prisma import Prisma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from pydantic import BaseModel
from typing import Optional, List
import os
import json

router = APIRouter()

# Initialize LangChain LLM (global)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize embeddings (global)
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Pydantic models for request validation
class JobCreateRequest(BaseModel):
    title: str
    description: str
    location: Optional[str] = None
    salary: Optional[int] = None
    type: str = "FULL_TIME"  # Default
    skills: Optional[List[str]] = None  # JSON array or null
    recruiterId: str  # Required for creation

class DescriptionGenerateRequest(BaseModel):
    title: str
    location: Optional[str] = None
    salary: Optional[int] = None
    type: str = "FULL_TIME"
    skills: Optional[str] = None  # Comma-separated string
    company_name: Optional[str] = None
    company_about: Optional[str] = None

# Updated Prompt template for AI job description generation - professional template with balanced structure
description_prompt = PromptTemplate(
    input_variables=["title", "location", "salary", "type", "skills", "company_name", "company_about"],
    template="""
You are an expert HR professional and job description writer with years of experience crafting compelling, inclusive job postings for top tech companies. Your goal is to create a highly professional, detailed, and engaging job description that attracts top talent in a flowing, narrative style with selective use of bullet points for clarity.

Job Details:
- Title: {title}
- Location: {location}
- Salary: {salary}
- Employment Type: {type}
- Key Skills: {skills}
- Company: {company_name}
- About Company: {company_about}

Generate an extensive job description (400-600 words) using a professional template structured as follows:

**Job Summary**  
Begin with a compelling paragraph (100-150 words) that provides an engaging overview of the role, the company's mission as described in the about section, and the crucial impact the candidate can make. Highlight inclusivity and appeal to diverse backgrounds.

**About the Role**  
Write a concise paragraph (50-75 words) describing the overall role and its place within the team and company.

**Key Responsibilities**  
Use a bulleted list (6-8 items) for core duties. Employ action-oriented language (e.g., "Collaborate with cross-functional teams to..."). Keep bullets specific yet adaptable.

**Qualifications**  
Start with a paragraph (50-75 words) outlining essential requirements, including education, experience, and the provided key skills. Encourage equivalents. Follow with a short bulleted list (4-6 items) for preferred qualifications to broaden appeal.

**What We Offer**  
Conclude the main sections with a narrative paragraph (75-100 words) describing our comprehensive benefits package, including competitive salary, health coverage, professional growth opportunities, work-life balance (remote/hybrid options, flexible hours), collaborative and innovative culture, and unique perks like equity and wellness initiatives.

**Our Commitment to Diversity, Equity, and Inclusion**  
Add a dedicated paragraph (50-75 words) reaffirming our dedication to fostering a diverse, equitable workplace and equal opportunity for all.

**How to Apply**  
End with an enthusiastic call-to-action paragraph (30-50 words) inviting candidates to submit their applications via our careers portal.

Use markdown formatting for headers (e.g., **bold** for subheaders) and ensure the language is professional, bias-free, ATS-optimized with keywords, and appealing to a global audience. Maintain a cohesive, readable flow.
"""
)

@router.post("/jobs/generate-description")
async def generate_description(request: DescriptionGenerateRequest):
    try:
        # Format skills as comma-separated string if provided
        skills_str = request.skills if request.skills else "Relevant technical and soft skills based on the role"
        
        # Create chain
        chain = description_prompt | llm
        
        # Invoke the chain asynchronously
        response = await chain.ainvoke({
            "title": request.title,
            "location": request.location or "Remote/Hybrid (flexible)",
            "salary": f"${request.salary:,}" if request.salary else "Competitive, based on experience",
            "type": request.type,
            "skills": skills_str,
            "company_name": request.company_name or "Innovative Tech Company",
            "company_about": request.company_about or "An innovative tech leader focused on AI and sustainability."
        })
        
        return {
            "description": response.content
        }

    except Exception as e:
        print(f"AI Error generating description: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate description")

@router.post("/jobs/create")
async def create_job(request: JobCreateRequest):
    prisma_client = Prisma()
    try:
        await prisma_client.connect()

        # 🔹 Step 1: Normalize existing skills data (fix old inconsistent records)
        old_jobs = await prisma_client.job.find_many()
        for j in old_jobs:
            if isinstance(j.skills, list):
                await prisma_client.job.update(
                    where={"id": j.id},
                    data={"skills": json.dumps(j.skills)}
                )

        # ✅ Re-fetch jobs after normalization to ensure in-memory data is up-to-date
        jobs = await prisma_client.job.find_many()

        # 🔹 Step 2: Prepare skills JSON for the new job
        skills_json = json.dumps(request.skills) if request.skills else None

        # 🔹 Step 3: Create the new job record
        job = await prisma_client.job.create(
            data={
                "title": request.title,
                "description": request.description,
                "location": request.location,
                "salary": request.salary,
                "type": request.type,
                "skills": skills_json,
                "recruiterId": request.recruiterId,
                "status": "OPEN",  # Default
            }
        )

        # 🔹 Step 4: Setup FAISS vectorstore directory
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        index_path = os.path.join(project_root, "services", "faiss_jobs_index")
        os.makedirs(index_path, exist_ok=True)

        # 🔹 Step 5: Create document for new job
        doc = Document(
            page_content=(
                f"Title: {request.title}\n"
                f"Description: {request.description}\n"
                f"Location: {request.location or ''}\n"
                f"Type: {request.type}\n"
                f"RecruiterId: {request.recruiterId}"
            ),
            metadata={
                "id": str(job.id),
                "title": request.title,
                "recruiterId": request.recruiterId,
            }
        )

        # 🔹 Step 6: Load or create FAISS vectorstore and add new job
        faiss_index = os.path.join(index_path, "index.faiss")
        faiss_meta = os.path.join(index_path, "index.pkl")

        if os.path.exists(faiss_index) and os.path.exists(faiss_meta):
            vectorstore = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            vectorstore.add_documents([doc])
        else:
            vectorstore = FAISS.from_documents([doc], embeddings)

        vectorstore.save_local(index_path)

        # 🔹 Step 7: Debug - Add one previous job to vectorstore
        previous_jobs = [j for j in jobs if j.id != job.id]
        print(f"DEBUG: Total jobs: {len(jobs)}, Previous jobs: {len(previous_jobs)}")

        if previous_jobs:
            first_previous_job = previous_jobs[0]
            print(f"DEBUG: Adding previous job: {first_previous_job.title} (ID: {first_previous_job.id})")

            # ✅ Safe skill parsing
            if first_previous_job.skills:
                if isinstance(first_previous_job.skills, str):
                    try:
                        previous_skills = json.loads(first_previous_job.skills)
                    except json.JSONDecodeError:
                        previous_skills = []
                elif isinstance(first_previous_job.skills, list):
                    previous_skills = first_previous_job.skills
                else:
                    previous_skills = []
            else:
                previous_skills = []

            doc_previous = Document(
                page_content=(
                    f"Title: {first_previous_job.title}\n"
                    f"Description: {first_previous_job.description}\n"
                    f"Location: {first_previous_job.location or ''}\n"
                    f"Type: {first_previous_job.type or ''}\n"
                    f"RecruiterId: {first_previous_job.recruiterId}"
                ),
                metadata={
                    "id": str(first_previous_job.id),
                    "title": first_previous_job.title,
                    "recruiterId": first_previous_job.recruiterId,
                }
            )

            vectorstore_previous = FAISS.load_local(
                index_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            vectorstore_previous.add_documents([doc_previous])
            vectorstore_previous.save_local(index_path)
            print("DEBUG: Previous job added to vector store successfully.")
        else:
            print("DEBUG: No previous jobs found.")

        # 🔹 Step 8: Return success
        return {"message": "Job created successfully", "job": job}

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await prisma_client.disconnect()
