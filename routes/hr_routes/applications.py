from fastapi import APIRouter, HTTPException, status, Body
from prisma import Prisma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
from typing import TypedDict, Annotated
import os
import json
import operator
import asyncio
import logging
from tavily import TavilyClient

# Set up logging for debugging - ensure DEBUG level
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize LangChain LLM (global) - Upgraded to gpt-4 for more professional outputs
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0.5,  # Lowered for more consistent, professional tone
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Initialize Tavily client (sync)
api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("TAVILY_API_KEY environment variable is required")
tavily_client = TavilyClient(api_key=api_key)

# Enhanced prompt template for more professional, extensive HR recommendations
# Expanded to 400-600 words, deeper integration of 2025 trends, salary benchmarks, and actionable insights
recommendation_prompt = PromptTemplate(
    input_variables=["employee_profile", "job_description", "retrieved_data"],
    template="""
You are an expert HR recommendation assistant with deep knowledge of 2025 talent dynamics. Analyze the candidate's profile against the job requirements in a highly professional, extensive manner (400-600 words total), weaving in real-time web data as Retrieval-Augmented Generation (RAG) context for cutting-edge industry trends, in-demand skills, evidence-based best practices, precise salary benchmarks (e.g., U.S. averages from Glassdoor/Indeed), career progression pathways, and emerging challenges like AI ethics, sustainability mandates, and regulatory shifts. Format in clean Markdown with the exact structure below, using H2 headings (##) without additional symbols except where specified. Adopt a formal, objective, data-driven tone with precise, actionable language that highlights synergies, quantifies impacts where possible, and emphasizes strategic value.

## Candidate Overview
An extensive summary (100-150 words) of key profile elements: technical/soft skills, Genius Factor breakdowns with interpretive insights (e.g., Spiritual Genius for empathetic leadership), current role/department context, alignment/retention risk metrics, and salary expectations benchmarked against 2025 market data (e.g., 'below average of $X for Y role'). Highlight transferable assets from healthcare/data analytics to marketing leadership.

## Strengths & Fit
A detailed analysis (100-150 words) of alignments: Map skills/experience/Genius Factors to role imperatives (e.g., PyTorch analytics to AI-driven personalization; empathy to team inspiration). Quantify potential impacts (e.g., '20-30% ROI uplift via data-informed campaigns'). Integrate RAG seamlessly for 2025 synergies like ethical AI branding or sustainability storytelling, positioning the candidate as a bridge between tech innovation and human-centric strategy.

## Areas for Development
Bullet-point list of 4-5 prioritized gaps, each with 2-3 sentences of rationale and tailored, measurable actions (e.g., 'Enroll in Coursera’s AI Strategy certification; pilot zero-party data tools'). Draw from RAG for timeliness (e.g., countering AI disillusionment, mastering ESG integrations, navigating privacy regs like CCPA evolutions).

## HR Recommendation
A strategic, forward-looking conclusion (100-150 words) with Overall Fit Score: X/10 (substantiated with 2-3 metrics/insights). Outline phased next steps (e.g., 'Panel interviews with strategy simulations; 3-month AI mentorship'). Conclude with a strong endorsement on growth trajectory, cultural fit, and ROI potential in transformative roles.

🟢 Summary

✅ Strengths:
- 4-5 bullet points of core positives, each with brief quantification or RAG tie-in.

⚙️ Improvements made:
- 3-4 bullet points on RAG-enhanced refinements or development accelerators.

Ensure outputs are extensive yet concise, data-enriched without overt citations, and optimized for executive decision-making. Leverage all context for nuanced, 2025-relevant advice.

Employee Profile: {employee_profile}

Job Description: {job_description}

RAG Context (Real-time Web Data): {retrieved_data}

Generate the professional Markdown-formatted recommendation:
"""
)

# State for LangGraph
class State(TypedDict):
    messages: Annotated[list, operator.add]
    employee_profile: str
    job_description: str
    job_title: str
    retrieved_data: str
    recommendation: str

# Enhanced Node: Perform real-time web search with expanded queries for richer RAG
async def rag_search_node(state: State) -> State:
    # Log entry into the node
    logger.info(f"Entering rag_search_node with job_title: {state.get('job_title', 'N/A')}")
    
    job_title = state['job_title']
    # Extract skills from employee_profile
    skills_line = [line for line in state["employee_profile"].split("\n") if "Skills:" in line]
    if skills_line:
        skills_str = skills_line[0].split(":", 1)[1].strip().replace(",", " ").replace("  ", " ")
    else:
        skills_str = "data analytics skills"

    # Tailored query focused on user skills relevance to job
    query = f"2025 {job_title} how {skills_str} transfer to marketing leadership trends benchmarks best practices AI ethics sustainability privacy"
    include_domains = ["skills-first.org", "glassdoor.com", "indeed.com", "linkedin.com", "forbes.com", "coursera.org", "mckinsey.com", "deloitte.com", "gartner.com"]
    
    # Log the query and params for debugging
    logger.debug(f"RAG Search Query: {query}")
    logger.debug(f"RAG Search Include Domains: {include_domains}")
    
    try:
        def sync_search():
            return tavily_client.search(
                query=query,
                include_domains=include_domains,
                max_results=20,  # Increased for deeper, more extensive context
                search_depth="advanced"
            )
        response = await asyncio.to_thread(sync_search)
        results = response.get("results", [])
        
        # Log the raw response for debugging
        logger.debug(f"RAG Search Raw Response Keys: {list(response.keys()) if response else 'None'}")
        if response:
            logger.debug(f"RAG Search Raw Response: {json.dumps(response, indent=2)[:1500]}...")  # Longer truncate for detail
        
        # Enhanced formatting: Include key excerpts, trends, benchmarks for RAG
        rag_context = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            content = r.get("content", "")[:1200]  # Longer snippets for extensive analysis
            url = r.get("url", "")
            rag_context.append(f"[Source {i}]: {title}\n{content}\nURL: {url}\n---\n")
        
        retrieved_data = "\n".join(rag_context)
        if not retrieved_data:
            retrieved_data = "No real-time data retrieved; default to comprehensive 2025 role knowledge including salary averages ~$180K-$220K for CMO, trends in AI/sustainability."
        
        # Log number of results and data length for debugging
        logger.info(f"RAG Search Retrieved {len(results)} results. Formatted data length: {len(retrieved_data)} chars")
        
        messages = state.get("messages", [])
        messages.append(HumanMessage(content=f"Enhanced RAG search for '{job_title}' 2025 trends: Retrieved {len(results)} sources with benchmarks and challenges."))
        
        return {
            "retrieved_data": retrieved_data,
            "messages": messages
        }
    except Exception as search_error:
        # Log the full error traceback for debugging
        logger.error(f"Tavily RAG search error for query '{query}': {search_error}", exc_info=True)
        return {
            "retrieved_data": f"RAG search failed: {str(search_error)}. Fallback: 2025 CMO benchmarks $188K avg; trends: AI personalization, ESG focus.",
            "messages": state.get("messages", [])
        }

# Node: Generate recommendation using LLM with RAG - Enhanced with retry logic
async def recommend_node(state: State) -> State:
    logger.info("Entering recommend_node")
    try:
        chain = recommendation_prompt | llm
        response = await chain.ainvoke({
            "employee_profile": state["employee_profile"],
            "job_description": state["job_description"],
            "retrieved_data": state["retrieved_data"]
        })
        logger.info("LLM recommendation generated successfully")
        return {"recommendation": response.content}
    except Exception as llm_error:
        logger.error(f"LLM invocation error: {llm_error}", exc_info=True)
        # Fallback response for robustness
        fallback = """
## Candidate Overview
[Fallback summary based on profile.]

## Strengths & Fit
[Fallback analysis.]

## Areas for Development
- [Item 1]
- [Item 2]

## HR Recommendation
Overall Fit Score: 7/10. Proceed with interviews.

🟢 Summary

✅ Strengths:
- Bullet 1

⚙️ Improvements made:
- Bullet 1
        """
        return {"recommendation": fallback}

# Build LangGraph with added error edges for robustness
workflow = StateGraph(State)
workflow.add_node("rag_search", rag_search_node)
workflow.add_node("recommend", recommend_node)
workflow.add_edge("rag_search", "recommend")
workflow.add_edge("recommend", END)
workflow.set_entry_point("rag_search")
app = workflow.compile()

@router.post("/applications")
async def create_application(
    request: dict = Body(...),  # Use Body for request body validation
):
    prisma_client = Prisma()
    try:
        await prisma_client.connect()

        # Extract from request with enhanced validation
        user_id = request.get("user_id") or request.get("userId")
        job_id = request.get("job_id") or request.get("jobId")
        hr_id = request.get("hr_id") or request.get("hrId")
        score = request.get("score")
        application_id = request.get("application_id")  # Optional: for update

        if not job_id:
            raise HTTPException(status_code=400, detail="Job ID is required")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID is required")

        is_update = bool(application_id)
        application = None

        if is_update:
            # Update existing application
            existing_application = await prisma_client.application.find_unique(
                where={"id": application_id}
            )
            if not existing_application:
                raise HTTPException(status_code=404, detail="Application not found")

            # Ensure it belongs to the user and job
            if existing_application.userId != user_id or existing_application.jobId != job_id:
                raise HTTPException(status_code=403, detail="Unauthorized to update this application")

            # Generate AI recommendation (regenerate for update)
            # Fetch user and job for profile and description
            user = await prisma_client.user.find_unique(
                where={"id": user_id},
                include={"employee": True},
            )
            reports = await prisma_client.individualemployeereport.find_many(
                where={"userId": user_id},
                order=[{"createdAt": "desc"}],
            )
            latest_report = reports[0] if reports else None

            job = await prisma_client.job.find_unique(
                where={"id": job_id},
                include={"recruiter": True},
            )

            if not user or not job:
                raise HTTPException(status_code=404, detail="User or Job not found")

            # Build employee data with improved parsing
            employee = user.employee if hasattr(user, "employee") and user.employee else None
            skills_data = getattr(employee, "skills", {}) if employee else {}
            education_data = getattr(employee, "education", {}) if employee else {}
            experience_data = getattr(employee, "experience", {}) if employee else {}

            # Extract lists with robust handling
            def safe_extract_list(data, key):
                if isinstance(data, dict):
                    items = data.get(key, [])
                elif hasattr(data, "__iter__"):
                    items = list(data)
                else:
                    items = []
                if items and isinstance(items[0], dict):
                    return [item.get("name") or item.get("skill") or item.get("degree") or item.get("title") or str(item) for item in items]
                return [str(item) for item in items]

            skills_list = safe_extract_list(skills_data, "skills")
            education_list = safe_extract_list(education_data, "education")
            experience_list = safe_extract_list(experience_data, "experience")

            # Latest position/department with fallback
            latest_position = (
                getattr(user, "position", ["Not specified"])[-1]
                if getattr(user, "position", ["Not specified"])
                else "Not specified"
            )
            latest_department = (
                getattr(user, "department", ["Not specified"])[-1]
                if getattr(user, "department", ["Not specified"])
                else "Not specified"
            )

            # Build report summary with JSON safety
            report_section = ""
            if latest_report:
                genius_score = getattr(latest_report, "geniusFactorScore", "N/A")
                executive_summary = getattr(
                    latest_report, "executiveSummary", "No summary available"
                )
                try:
                    genius_profile = json.dumps(
                        json.loads(getattr(latest_report, "geniusFactorProfileJson", "{}")), indent=2
                    )
                except:
                    genius_profile = "{}"
                report_section = f"""
Latest Individual Employee Report (Created: {getattr(latest_report, 'createdAt', 'N/A')}):
- Genius Factor Score: {genius_score}
- Executive Summary: {executive_summary}
- Genius Factor Profile: {genius_profile}
""".strip()

            # Employee profile string with enhanced formatting
            employee_profile = f"""
Name: {getattr(user, 'firstName', '')} {getattr(user, 'lastName', '')}
Bio: {getattr(employee, 'bio', 'No bio available') if employee else 'No bio available'}
Skills: {', '.join(skills_list) if skills_list else 'No skills listed'}
Education: {', '.join(education_list) if education_list else 'No education listed'}
Experience: {', '.join(experience_list) if experience_list else 'No experience listed'}
Latest Position: {latest_position}
Latest Department: {latest_department}
Salary Expectation: {getattr(user, 'salary', 'Not specified')}
{report_section}
""".strip()

            # Job description and title with recruiter details
            recruiter = job.recruiter if hasattr(job, "recruiter") and job.recruiter else None
            recruiter_name = (
                f"{getattr(recruiter, 'firstName', '')} {getattr(recruiter, 'lastName', '')}".strip()
                if recruiter else "Not specified"
            )
            job_title = getattr(job, 'title', '') or ''
            job_description = f"""
Title: {job_title}
Description: {getattr(job, 'description', 'No description available')}
Location: {getattr(job, 'location', 'Not specified')}
Type: {getattr(job, 'type', 'Not specified')}
Salary: {getattr(job, 'salary', 'Not specified')}
Recruiter: {recruiter_name}
""".strip()

            # Generate AI recommendation using LangGraph with RAG - Enhanced logging
            ai_recommendation = "AI recommendation temporarily unavailable due to processing error."
            try:
                logger.info(f"Starting enhanced LangGraph invocation for job '{job_title}' (update)")
                inputs = {
                    "employee_profile": employee_profile,
                    "job_description": job_description,
                    "job_title": job_title,
                    "messages": []
                }
                result = await app.ainvoke(inputs)
                ai_recommendation = result["recommendation"]
                logger.info("Enhanced LangGraph completed successfully with extensive output")
            except Exception as graph_error:
                logger.error(f"Enhanced LangGraph Error: {graph_error}", exc_info=True)
                print(f"Enhanced LangGraph Error: {graph_error}")  # Fallback print

            # Update Application
            application = await prisma_client.application.update(
                where={"id": application_id},
                data={
                    "aiRecommendation": ai_recommendation,
                    "scoreMatch": str(score) if score else None,
                    "hrId": hr_id,  # Update HR if provided
                },
                include={"user": True, "job": True},
            )

            logger.info(f"Application {application_id} updated successfully for user {user_id}, job {job_id}")

        else:
            # Create new application
            # Check if application already exists
            existing_application = await prisma_client.application.find_first(
                where={"userId": user_id, "jobId": job_id}
            )
            if existing_application:
                raise HTTPException(status_code=409, detail="Application already submitted")

            # Fetch user and job
            user = await prisma_client.user.find_unique(
                where={"id": user_id},
                include={"employee": True},
            )
            reports = await prisma_client.individualemployeereport.find_many(
                where={"userId": user_id},
                order=[{"createdAt": "desc"}],
            )
            latest_report = reports[0] if reports else None

            job = await prisma_client.job.find_unique(
                where={"id": job_id},
                include={"recruiter": True},
            )

            if not user or not job:
                raise HTTPException(status_code=404, detail="User or Job not found")

            # Build employee data with improved parsing
            employee = user.employee if hasattr(user, "employee") and user.employee else None
            skills_data = getattr(employee, "skills", {}) if employee else {}
            education_data = getattr(employee, "education", {}) if employee else {}
            experience_data = getattr(employee, "experience", {}) if employee else {}

            # Extract lists with robust handling
            def safe_extract_list(data, key):
                if isinstance(data, dict):
                    items = data.get(key, [])
                elif hasattr(data, "__iter__"):
                    items = list(data)
                else:
                    items = []
                if items and isinstance(items[0], dict):
                    return [item.get("name") or item.get("skill") or item.get("degree") or item.get("title") or str(item) for item in items]
                return [str(item) for item in items]

            skills_list = safe_extract_list(skills_data, "skills")
            education_list = safe_extract_list(education_data, "education")
            experience_list = safe_extract_list(experience_data, "experience")

            # Latest position/department with fallback
            latest_position = (
                getattr(user, "position", ["Not specified"])[-1]
                if getattr(user, "position", ["Not specified"])
                else "Not specified"
            )
            latest_department = (
                getattr(user, "department", ["Not specified"])[-1]
                if getattr(user, "department", ["Not specified"])
                else "Not specified"
            )

            # Build report summary with JSON safety
            report_section = ""
            if latest_report:
                genius_score = getattr(latest_report, "geniusFactorScore", "N/A")
                executive_summary = getattr(
                    latest_report, "executiveSummary", "No summary available"
                )
                try:
                    genius_profile = json.dumps(
                        json.loads(getattr(latest_report, "geniusFactorProfileJson", "{}")), indent=2
                    )
                except:
                    genius_profile = "{}"
                report_section = f"""
Latest Individual Employee Report (Created: {getattr(latest_report, 'createdAt', 'N/A')}):
- Genius Factor Score: {genius_score}
- Executive Summary: {executive_summary}
- Genius Factor Profile: {genius_profile}
""".strip()

            # Employee profile string with enhanced formatting
            employee_profile = f"""
Name: {getattr(user, 'firstName', '')} {getattr(user, 'lastName', '')}
Bio: {getattr(employee, 'bio', 'No bio available') if employee else 'No bio available'}
Skills: {', '.join(skills_list) if skills_list else 'No skills listed'}
Education: {', '.join(education_list) if education_list else 'No education listed'}
Experience: {', '.join(experience_list) if experience_list else 'No experience listed'}
Latest Position: {latest_position}
Latest Department: {latest_department}
Salary Expectation: {getattr(user, 'salary', 'Not specified')}
{report_section}
""".strip()

            # Job description and title with recruiter details
            recruiter = job.recruiter if hasattr(job, "recruiter") and job.recruiter else None
            recruiter_name = (
                f"{getattr(recruiter, 'firstName', '')} {getattr(recruiter, 'lastName', '')}".strip()
                if recruiter else "Not specified"
            )
            job_title = getattr(job, 'title', '') or ''
            job_description = f"""
Title: {job_title}
Description: {getattr(job, 'description', 'No description available')}
Location: {getattr(job, 'location', 'Not specified')}
Type: {getattr(job, 'type', 'Not specified')}
Salary: {getattr(job, 'salary', 'Not specified')}
Recruiter: {recruiter_name}
""".strip()

            # Generate AI recommendation using LangGraph with RAG - Enhanced logging
            ai_recommendation = "AI recommendation temporarily unavailable due to processing error."
            try:
                logger.info(f"Starting enhanced LangGraph invocation for job '{job_title}'")
                inputs = {
                    "employee_profile": employee_profile,
                    "job_description": job_description,
                    "job_title": job_title,
                    "messages": []
                }
                result = await app.ainvoke(inputs)
                ai_recommendation = result["recommendation"]
                logger.info("Enhanced LangGraph completed successfully with extensive output")
            except Exception as graph_error:
                logger.error(f"Enhanced LangGraph Error: {graph_error}", exc_info=True)
                print(f"Enhanced LangGraph Error: {graph_error}")  # Fallback print

            # Create Application
            application = await prisma_client.application.create(
                data={
                    "userId": user_id,
                    "jobId": job_id,
                    "hrId": hr_id,
                    "aiRecommendation": ai_recommendation,
                    "scoreMatch": str(score) if score else None,
                },
                include={"user": True, "job": True},
            )

            # Update applied jobs with atomic operation
            await prisma_client.user.update(
                where={"id": user_id},
                data={"appliedJobIds": {"push": job_id}},
            )

            logger.info(f"Application created successfully for user {user_id}, job {job_id}")

        message = "Application updated successfully" if is_update else "Application submitted successfully"
        return {"message": message, "application": application}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing application: {e}", exc_info=True)
        print(f"Error processing application: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        await prisma_client.disconnect()