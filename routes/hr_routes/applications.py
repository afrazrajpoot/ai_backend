from fastapi import APIRouter, HTTPException, status, Body
from prisma import Prisma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
import json  # For handling Json fields if needed

router = APIRouter()

# Initialize LangChain LLM (global)
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Prompt template for AI recommendation
recommendation_prompt = PromptTemplate(
    input_variables=["employee_profile", "job_description"],
    template="""
You are an expert HR recommendation assistant. Analyze the candidate's profile against the job requirements and provide a concise recommendation (200-300 words) for the HR team. Include strengths, potential fit, areas for improvement, and suggested next steps (e.g., interview, skills assessment). Be professional and objective.

Employee Profile: {employee_profile}

Job Description: {job_description}

Generate the recommendation:
"""
)

@router.post("/applications")
async def create_application(
    request: dict = Body(...),  # Use Body for request body validation
):
    prisma_client = Prisma()  # Create per-request instance
    try:
        await prisma_client.connect()  # Connect before querying

        # Extract from request (match frontend camelCase keys)
        user_id = request.get("user_id") or request.get("userId")
        job_id = request.get("job_id") or request.get("jobId")
        hr_id = request.get("hr_id") or request.get("hrId")
        score = request.get("score")

        if not job_id:
            raise HTTPException(status_code=400, detail="Job ID is required")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")

        # Check if application already exists
        existing_application = await prisma_client.application.find_first(
            where={
                "userId": user_id,
                "jobId": job_id,
            }
        )

        if existing_application:
            raise HTTPException(status_code=409, detail="Application already submitted")

        # Fetch user and job data
        user = await prisma_client.user.find_unique(
            where={"id": user_id},
            include={"employee": True}
        )

        # Fetch latest IndividualEmployeeReport using find_many + order
        reports = await prisma_client.individualemployeereport.find_many(
            where={"userId": user_id},
            order=[{"createdAt": "desc"}]
        )
        latest_report = reports[0] if reports else None

        job = await prisma_client.job.find_unique(
            where={"id": job_id},
            include={"recruiter": True}
        )

        if not user or not job:
            raise HTTPException(status_code=404, detail="User or Job not found")

        # Construct employee profile (use dot notation for Prisma models)
        employee = user.employee if hasattr(user, 'employee') and user.employee else None
        
        # Handle Json fields from Employee (assuming Prisma deserializes to dict/list)
        skills_data = getattr(employee, 'skills', {}) if employee else {}
        education_data = getattr(employee, 'education', {}) if employee else {}
        experience_data = getattr(employee, 'experience', {}) if employee else {}
        
        # Extract lists from Json if they are dicts (adapt based on actual structure)
        skills_list = skills_data.get('skills', []) if isinstance(skills_data, dict) else skills_data if hasattr(skills_data, '__iter__') else []
        education_list = education_data.get('education', []) if isinstance(education_data, dict) else education_data if hasattr(education_data, '__iter__') else []
        experience_list = experience_data.get('experience', []) if isinstance(experience_data, dict) else experience_data if hasattr(experience_data, '__iter__') else []
        
        # Latest position and department (last in array)
        latest_position = getattr(user, 'position', [])[-1] if getattr(user, 'position', []) else 'Not specified'
        latest_department = getattr(user, 'department', [])[-1] if getattr(user, 'department', []) else 'Not specified'
        
        # Latest report data (if exists)
        report_section = ""
        if latest_report:
            genius_score = getattr(latest_report, 'geniusFactorScore', 'N/A')
            executive_summary = getattr(latest_report, 'executiveSummary', 'No summary available')
            genius_profile = json.dumps(getattr(latest_report, 'geniusFactorProfileJson', {}), indent=2) if getattr(latest_report, 'geniusFactorProfileJson') else '{}'
            current_role = json.dumps(getattr(latest_report, 'currentRoleAlignmentAnalysisJson', {}), indent=2) if getattr(latest_report, 'currentRoleAlignmentAnalysisJson') else '{}'
            opportunities = json.dumps(getattr(latest_report, 'internalCareerOpportunitiesJson', {}), indent=2) if getattr(latest_report, 'internalCareerOpportunitiesJson') else '{}'
            strategies = json.dumps(getattr(latest_report, 'retentionAndMobilityStrategiesJson', {}), indent=2) if getattr(latest_report, 'retentionAndMobilityStrategiesJson') else '{}'
            action_plan = json.dumps(getattr(latest_report, 'developmentActionPlanJson', {}), indent=2) if getattr(latest_report, 'developmentActionPlanJson') else '{}'
            resources = json.dumps(getattr(latest_report, 'personalizedResourcesJson', {}), indent=2) if getattr(latest_report, 'personalizedResourcesJson') else '{}'
            methodology = json.dumps(getattr(latest_report, 'dataSourcesAndMethodologyJson', {}), indent=2) if getattr(latest_report, 'dataSourcesAndMethodologyJson') else '{}'
            risk_analysis = json.dumps(getattr(latest_report, 'risk_analysis', {}), indent=2) if getattr(latest_report, 'risk_analysis') else '{}'
            
            report_section = f"""
Latest Individual Employee Report (Created: {getattr(latest_report, 'createdAt', 'N/A')}):
- Genius Factor Score: {genius_score}
- Executive Summary: {executive_summary}
- Genius Factor Profile: {genius_profile}
- Current Role Alignment Analysis: {current_role}
- Internal Career Opportunities: {opportunities}
- Retention and Mobility Strategies: {strategies}
- Development Action Plan: {action_plan}
- Personalized Resources: {resources}
- Data Sources and Methodology: {methodology}
- Risk Analysis: {risk_analysis}
            """.strip()
        
        employee_profile = f"""
Name: {getattr(user, 'firstName', '') or ''} {getattr(user, 'lastName', '') or ''}
Bio: {getattr(employee, 'bio', 'No bio available') if employee else 'No bio available'}
Skills: {', '.join(skills_list) if skills_list else 'No skills listed'}
Education: {', '.join(education_list) if education_list else 'No education listed'}
Experience: {', '.join(experience_list) if experience_list else 'No experience listed'}
Latest Position: {latest_position}
Latest Department: {latest_department}
Salary Expectation: {getattr(user, 'salary', 'Not specified') or 'Not specified'}
{report_section}
        """.strip()

        # Construct job description (use dot notation)
        recruiter = job.recruiter if hasattr(job, 'recruiter') and job.recruiter else None
        recruiter_name = f"{getattr(recruiter, 'firstName', '') or ''} {getattr(recruiter, 'lastName', '') or ''}".strip()
        job_description = f"""
Title: {getattr(job, 'title', '') or ''}
Description: {getattr(job, 'description', 'No description available') or 'No description available'}
Location: {getattr(job, 'location', 'Not specified') or 'Not specified'}
Type: {getattr(job, 'type', 'Not specified') or 'Not specified'}
Salary: {getattr(job, 'salary', 'Not specified') or 'Not specified'}
Recruiter: {recruiter_name}
        """.strip()

        # Generate AI recommendation using LangChain
        ai_recommendation = "AI recommendation temporarily unavailable."
        try:
            chain = recommendation_prompt | llm
            response = await chain.ainvoke({
                "employee_profile": employee_profile,
                "job_description": job_description
            })
            ai_recommendation = response.content
        except Exception as ai_error:
            print(f"AI Error: {ai_error}")  # Log error
            # Fallback remains

        # Create the application
        application = await prisma_client.application.create(
            data={
                "userId": user_id,
                "jobId": job_id,
                "hrId": hr_id,
                "aiRecommendation": ai_recommendation,
                "scoreMatch": str(score) if score else None,
            },
            include={
                "user": True,
                "job": True,
            }
        )

        # Update user's appliedJobIds
        await prisma_client.user.update(
            where={"id": user_id},
            data={
                "appliedJobIds": {
                    "push": job_id,
                },
            },
        )

        return {
            "message": "Application submitted successfully",
            "application": application
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error creating application: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await prisma_client.disconnect()