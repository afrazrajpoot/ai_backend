# routes/hr_routes/job_creation.py - Complete job creation and description generation routes
from fastapi import APIRouter, HTTPException, status, Body
from prisma import Prisma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
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