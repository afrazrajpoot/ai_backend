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

# Prompt template for AI job description generation
description_prompt = PromptTemplate(
    input_variables=["title", "location", "salary", "type", "skills"],
    template="""
You are an expert job description writer. Generate a professional, engaging job description for the following role.

Job Details:
- Title: {title}
- Location: {location}
- Salary: {salary}
- Type: {type}
- Skills: {skills}

Write a concise description (150-250 words) that includes:
- Role overview and responsibilities
- Required qualifications and skills
- Company culture/benefits (assume a modern tech company)
- Call to action

Make it inclusive and appealing to diverse candidates.
"""
)

@router.post("/jobs/generate-description")
async def generate_description(request: DescriptionGenerateRequest):
    # Format skills as comma-separated string if provided
    skills_str = request.skills if request.skills else ""
    
    # Create chain
    chain = description_prompt | llm
    
    # Invoke the chain
    response = chain.invoke({
        "title": request.title,
        "location": request.location or "",
        "salary": f"${request.salary:,}" if request.salary else "Competitive",
        "type": request.type,
        "skills": skills_str
    })
    
    return {
        "description": response.content
    }

@router.post("/jobs/create")
async def create_job(request: JobCreateRequest):
    prisma_client = Prisma()
    try:
        await prisma_client.connect()

        # Prepare skills as JSON if provided
        skills_json = json.dumps(request.skills) if request.skills else None

        # Create the job
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

        # Add job to FAISS vector store in services folder
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        index_path = os.path.join(project_root, "services", "faiss_jobs_index")
        os.makedirs(index_path, exist_ok=True)
        
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
        
        if os.path.exists(os.path.join(index_path, "index.faiss")) and os.path.exists(os.path.join(index_path, "index.pkl")):
            vectorstore = FAISS.load_local(
                index_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            vectorstore.add_documents([doc])
        else:
            vectorstore = FAISS.from_documents([doc], embeddings)
        
        vectorstore.save_local(index_path)

        # For debugging: get first previous job and add it new in store
        print("DEBUG: Fetching previous jobs...")
        all_jobs = await prisma_client.job.find_many()
        previous_jobs = [j for j in all_jobs if j.id != job.id]
        print(f"DEBUG: Total jobs: {len(all_jobs)}, Previous jobs: {len(previous_jobs)}")
        if previous_jobs:
            first_previous_job = previous_jobs[0]
            print(f"DEBUG: Adding previous job: {first_previous_job.title} (ID: {first_previous_job.id})")
            # Parse skills if needed
            previous_skills = json.loads(first_previous_job.skills) if first_previous_job.skills else []
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
            # Load the vectorstore again and add the previous job doc
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

        return {
            "message": "Job created successfully",
            "job": job
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating job: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await prisma_client.disconnect()
@router.post("/jobs/generate-description")
async def generate_description(request: DescriptionGenerateRequest):
    try:
        # Generate AI description using LangChain
        chain = description_prompt | llm
        response = await chain.ainvoke({
            "title": request.title,
            "location": request.location or "Not specified",
            "salary": request.salary or "Competitive",
            "type": request.type,
            "skills": request.skills or "Relevant skills"
        })

        description = response.content

        return {
            "description": description
        }

    except Exception as e:
        print(f"AI Error generating description: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate description")