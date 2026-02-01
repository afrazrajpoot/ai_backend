import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime
import asyncpg
import asyncio
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils.logger import logger
from config import settings


# ==================== FIRST LLM OUTPUT MODEL ====================
class StaticContextAnalysis(BaseModel):
    """Output from FIRST LLM: Genius factors and related info from static context"""
    primary_genius_factor: str = Field(description="Primary genius factor from static context")
    secondary_genius_factor: Optional[str] = Field(description="Secondary genius factor from static context")
    scoring_guide_for_factors: str = Field(description="Scoring guide sections for these genius factors")
    industries_for_factors: str = Field(description="Industry mapping for these genius factors")
    primary_industries: List[str] = Field(description="List of primary industries for the identified genius factor")
    secondary_industries: List[str] = Field(description="List of secondary industries for the identified genius factor")
    dominant_pattern: str = Field(description="Dominant response pattern (A, B, C, D)")
    key_insights: str = Field(description="Key insights from matching assessment to static context")


# ==================== THIRD LLM OUTPUT MODELS ====================
class GeniusFactorProfile(BaseModel):
    """Genius factor profile section"""
    primary_genius_factor: str = Field(description="Primary genius factor")
    secondary_genius_factor: Optional[str] = Field(description="Secondary genius factor")
    key_strengths: List[str] = Field(description="List of key strengths")
    energy_sources: List[str] = Field(description="What motivates this profile")
    development_areas: List[str] = Field(description="Areas for growth")
    description: str = Field(description="Detailed description of the genius factor")


class CurrentRoleAlignmentAnalysis(BaseModel):
    """Current role alignment analysis section"""
    alignment_score: int = Field(description="Alignment score (0-100)", ge=0, le=100)
    strengths_utilized: List[str] = Field(description="Strengths currently utilized")
    underutilized_talents: List[str] = Field(description="Underutilized talents")
    retention_risk_factors: List[str] = Field(description="Risk factors for retention")
    immediate_actions: List[str] = Field(description="Immediate improvement actions")


class CareerOpportunity(BaseModel):
    """Individual career opportunity"""
    role_title: str = Field(description="Job title")
    department: str = Field(description="Target department")
    match_score: int = Field(description="Match score (0-100)", ge=0, le=100)
    required_skills: List[str] = Field(description="Skills needed")
    timeline: str = Field(description="Recommended timeline")
    salary_impact: str = Field(description="Expected salary impact")


class InternalCareerOpportunities(BaseModel):
    """Internal career opportunities section"""
    primary_industries: List[str] = Field(description="Primary industries for genius factor")
    secondary_industries: List[str] = Field(description="Secondary industries")
    recommended_departments: List[str] = Field(description="Recommended departments")
    role_suggestions: List[CareerOpportunity] = Field(description="Specific role suggestions")
    transition_strategy: str = Field(description="Overall transition strategy")


class RetentionMobilityStrategies(BaseModel):
    """Retention and mobility strategies section"""
    retention_strategies: List[str] = Field(description="Retention strategies")
    mobility_recommendations: List[str] = Field(description="Mobility recommendations")
    development_support_needed: List[str] = Field(description="Required development support")
    expected_outcomes: List[str] = Field(description="Expected outcomes")


class DevelopmentActionPlan(BaseModel):
    """Development action plan section"""
    thirty_day_goals: List[str] = Field(description="30-day goals")
    ninety_day_goals: List[str] = Field(description="90-day goals")
    six_month_goals: List[str] = Field(description="6-month goals")
    networking_strategy: Dict[str, List[str]] = Field(description="Networking strategy")


class PersonalizedResources(BaseModel):
    """Personalized resources section"""
    affirmations: List[str] = Field(description="Personalized affirmations")
    learning_resources: List[Dict[str, str]] = Field(description="Courses, books, etc.")
    reflection_questions: List[str] = Field(description="Questions for self-reflection")
    mindfulness_practices: List[str] = Field(description="Mindfulness practices")


class DataSourcesMethodology(BaseModel):
    """Data sources and methodology section"""
    assessment_data_used: bool = Field(description="Assessment data utilized")
    user_data_used: bool = Field(description="User data utilized")
    static_context_used: bool = Field(description="Static context references used")
    score_calculation_method: str = Field(description="Method used for score calculations")


class ProfessionalAssessmentReport(BaseModel):
    """Complete professional assessment report from THIRD LLM"""
    executive_summary: str = Field(description="3-4 paragraph executive summary")
    
    # Core Scores (Generated as Integers)
    genius_factor_score: int = Field(description="Genius Factor Score (0-100) based on response consistency and pattern dominance", ge=0, le=100)
    retention_risk_score: int = Field(description="Retention Risk Score (0-100) based on role alignment and market factors", ge=0, le=100)
    mobility_opportunity_score: int = Field(description="Mobility Opportunity Score (0-100) based on internal career paths", ge=0, le=100)
    
    # Detailed Sections
    genius_factor_profile: GeniusFactorProfile = Field(description="Genius factor profile")
    current_role_alignment_analysis: CurrentRoleAlignmentAnalysis = Field(description="Role alignment analysis")
    internal_career_opportunities: InternalCareerOpportunities = Field(description="Career opportunities")
    retention_and_mobility_strategies: RetentionMobilityStrategies = Field(description="Retention strategies")
    development_action_plan: DevelopmentActionPlan = Field(description="Development plan")
    personalized_resources: PersonalizedResources = Field(description="Personalized resources")
    data_sources_and_methodology: DataSourcesMethodology = Field(description="Data sources and methodology")
    
    # Metadata
    generated_at: datetime = Field(default_factory=datetime.now)
    report_version: str = Field(default="1.0")


# ==================== AI SERVICE ====================
class AIService:
    _prompts = None
    _static_context = None

    @classmethod
    def _load_prompts(cls):
        if cls._prompts is None:
            try:
                prompts_path = Path(__file__).parent.parent / "utils" / "prompts.json"
                with open(prompts_path) as f:
                    cls._prompts = json.load(f)
            except Exception as e:
                logger.error(f"Error loading prompts: {str(e)}")
                raise
        return cls._prompts

    @classmethod
    def _load_static_context(cls):
        if cls._static_context is None:
            try:
                context_path = Path(__file__).parent.parent / "utils" / "static_context.json"
                if context_path.exists():
                    with open(context_path) as f:
                        cls._static_context = json.load(f)
                else:
                    logger.warning("Static context file not found. Using empty context.")
                    cls._static_context = {}
            except Exception as e:
                logger.error(f"Error loading static context: {str(e)}")
                cls._static_context = {}
        return cls._static_context

    @staticmethod
    async def _fetch_user_data(user_id: str) -> Dict[str, Any]:
        """Fetch user and employee data from database"""
        if not user_id:
            return {}
            
        conn = None
        try:
            db_params = {
                "user": "postgres",
                "password": "root",
                "database": "genius_factor",
                "host": "localhost",
                "port": 5432
            }
            
            conn = await asyncpg.connect(**db_params)
            
            query = """
                SELECT 
                    u.id, u."firstName", u."lastName", u.email, u.position, u.department, u.salary, u."hrId",
                    e.skills, e.education, e.experience, e.bio
                FROM users u
                LEFT JOIN "Employee" e ON u."employeeId" = e.id
                WHERE u.id = $1
            """
            
            row = await conn.fetchrow(query, user_id)
            
            if row:
                return dict(row)
            return {}
            
        except Exception as e:
            logger.error(f"Error fetching user data: {str(e)}")
            return {}
        finally:
            if conn:
                await conn.close()

    @classmethod
    async def _first_llm_extract_static_info(cls, assessment_data: Dict, analysis_summary: str = "") -> StaticContextAnalysis:
        """FIRST LLM: Extract genius factors and related info from static context using algorithmic findings"""
        try:
            llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o",
                temperature=0.3,
                max_tokens=2000
            )
            
            # Load static context
            static_context = cls._load_static_context()
            
            parser = PydanticOutputParser(pydantic_object=StaticContextAnalysis)
            
            prompt = PromptTemplate(
                template="""You are an expert HR analyst. Analyze the assessment data and extract relevant information FROM THE STATIC CONTEXT.

                ========== STATIC CONTEXT ==========
                **SCORING GUIDE:**
                {scoring_guide}
                
                **INDUSTRY MAPPING:**
                {industry_mapping}
                =====================================

                ========== ASSESSMENT DATA ==========
                {assessment_payload}
                =====================================

                ========== ALGORITHMIC ANALYSIS RESULTS ==========
                {analysis_summary}
                =================================================

                **YOUR TASK: Extract relevant static context for this user:**
                1. **VERIFY** dominant patterns based on THE ALGORITHMIC RESULTS ABOVE.
                2. **IDENTIFY** which genius factors from static context match the PRIMARY and SECONDARY factors found by the algorithm.
                3. **EXTRACT** the scoring guide sections for THOSE specific genius factors
                4. **EXTRACT** the industry mapping for THOSE specific genius factors
                5. **EXTRACT** the list of "Primary Industries" for the identified genius factor
                6. **EXTRACT** the list of "Secondary Industries" for the identified genius factor
                7. **DETERMINE** dominant response pattern (A, B, C, D)
                8. **PROVIDE** key insights about this match

                **IMPORTANT:** Only use genius factors defined in static context. Do not make up new ones.

                {format_instructions}
                """,
                input_variables=["scoring_guide", "industry_mapping", "assessment_payload", "analysis_summary"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            chain = prompt | llm | parser
            
            # Retry logic for LLM call and parsing
            max_retries = 3
            last_error = None
            for attempt in range(max_retries):
                try:
                    analysis = await chain.ainvoke({
                        "scoring_guide": static_context.get("scoring_guide", ""),
                        "industry_mapping": static_context.get("industry_mapping", ""),
                        "assessment_payload": json.dumps(assessment_data, indent=2, default=str),
                        "analysis_summary": analysis_summary
                    })
                    logger.info(f"First LLM extracted: {analysis.primary_genius_factor} with industries")
                    return analysis
                except Exception as e:
                    last_error = e
                    logger.warning(f"First LLM attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep((attempt + 1) * 2)
            
            raise last_error
            
        except Exception as e:
            logger.error(f"Error in first LLM extraction: {str(e)}")
            raise

    @classmethod
    async def _third_llm_generate_professional_report(
        cls, 
        static_analysis: StaticContextAnalysis,
        user_data: Dict,
        assessment_data: Dict,
        analysis_data: str = ""
    ) -> ProfessionalAssessmentReport:
        """THIRD LLM: Generate complete professional report using Pydantic parser"""
        try:
            llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o",
                temperature=0.3,
                max_tokens=4000
            )
            
            # Create output parser for the complete report
            parser = PydanticOutputParser(pydantic_object=ProfessionalAssessmentReport)
            
            # Load system prompt
            prompt_data = cls._load_prompts()
            system_prompt = prompt_data.get('system_prompt', '')
            
            # Helper function to safely extract and format data
            def safe_extract(data, key, default="Not specified"):
                value = data.get(key, default)
                if isinstance(value, dict):
                    try:
                        return json.dumps(value)[:500]
                    except:
                        return str(value)[:500]
                elif isinstance(value, list):
                    return ", ".join(str(item) for item in value[:10])
                return str(value)
            
            # Prepare user data safely
            user_name = f"{user_data.get('firstName', '')} {user_data.get('lastName', '')}".strip() or "Employee"
            user_position = safe_extract(user_data, "position")
            user_department = safe_extract(user_data, "department")
            
            # Safely format salary: handle string/numeric conversion and formatting
            salary_val = user_data.get('salary')
            if salary_val:
                try:
                    num_salary = float(salary_val)
                    if num_salary > 0:
                        user_salary = f"${num_salary:,.0f}"
                    else:
                        user_salary = "Not disclosed"
                except (ValueError, TypeError):
                    user_salary = "Not disclosed"
            else:
                user_salary = "Not disclosed"
            
            user_skills = safe_extract(user_data, "skills")
            user_experience = safe_extract(user_data, "experience")
            user_education = safe_extract(user_data, "education")
            
            # Create the prompt with Pydantic format instructions
            prompt = PromptTemplate(
                template=system_prompt + """
                
                ========== ANALYSIS SUMMARY ==========
                {analysis_data}
                ======================================

                ========== STATIC CONTEXT INFORMATION ==========
                **GENIUS FACTORS IDENTIFIED:**
                - Primary: {primary_factor}
                - Secondary: {secondary_factor}
                
                **SCORING GUIDE FOR THESE FACTORS:**
                {scoring_guide}
                
                **INDUSTRY MAPPING FOR THESE FACTORS:**
                {industries_mapping}
                
                **PRIMARY INDUSTRIES:** {primary_industries}
                **SECONDARY INDUSTRIES:** {secondary_industries}
                
                **DOMINANT PATTERN:** {dominant_pattern}
                **KEY INSIGHTS:** {key_insights}
                ================================================

                ========== USER PROFILE DATA ==========
                **PERSONAL INFO:**
                - Name: {user_name}
                - Position: {user_position}
                - Department: {user_department}
                - Salary: {user_salary}
                
                **SKILLS & EXPERIENCE:**
                - Skills: {user_skills}
                - Experience: {user_experience}
                - Education: {user_education}
                =======================================

                ========== ASSESSMENT DETAILS ==========
                **ASSESSMENT PAYLOAD:**
                {assessment_payload}
                ========================================

                **YOUR TASK: Generate a COMPLETE, PROFESSIONAL career recommendation report**

                **REPORT MUST INCLUDE ALL THESE SECTIONS:**

                1. **EXECUTIVE SUMMARY** (3-4 paragraphs):
                   - Start with a highly professional, executive-level introduction
                   - State identified genius factors clearly and confidently
                   - Explicitly mention the **Primary Industries** and **Secondary Industries** identified
                   - Summarize career implications with strategic depth
                   - End with high-impact strategic recommendations

                2. **GENIUS FACTOR PROFILE** (Detailed):
                   - Primary and secondary genius factors with descriptions
                   - 5-6 specific key strengths based on assessment
                   - 3-4 energy sources that motivate this profile
                   - 3-4 development areas for growth

                3. **CURRENT ROLE ALIGNMENT ANALYSIS** (Comprehensive):
                   - Detailed assessment of current position alignment
                   - Alignment score (0-100) with justification based on data
                   - List of strengths currently utilized
                   - Underutilized talents from assessment
                   - Retention risk factors
                   - Immediate actions for improvement

                4. **INTERNAL CAREER OPPORTUNITIES** (Specific):
                   - **Primary Industries**: Use the extracted primary industries
                   - **Secondary Industries**: Use the extracted secondary industries
                   - Recommended departments based on industry mapping
                   - 5-6 specific role suggestions with reasoning
                   - Required skill development for each role
                   - Transition timeline (short/mid/long term)
                   - Success metrics for transition

                5. **RETENTION & MOBILITY STRATEGIES** (Actionable):
                   - 4-5 retention strategies with implementation steps
                   - Internal mobility recommendations
                   - Development support needed
                   - Expected outcomes

                6. **DEVELOPMENT ACTION PLAN** (Detailed):
                   - 3-4 thirty-day goals with specific actions
                   - 3-4 ninety-day goals with measurable outcomes
                   - 3-4 six-month goals with career progression
                   - Networking strategy with specific steps

                7. **PERSONALIZED RESOURCES** (MANDATORY - DO NOT LEAVE EMPTY):
                   - 4-5 affirmations specific to the identified genius factors
                   - 4-5 learning resources (EXACT titles of actual courses, books, or specific certifications)
                   - 3-4 reflection questions for self-assessment
                   - 3-4 mindfulness practices for career development
                   - **IMPORTANT**: This section MUST contain real data and actionable recommendations. Placeholder text or generic descriptions are UNACCEPTABLE.

                8. **DATA SOURCES & METHODOLOGY** (Transparent):
                   - Document what data was used
                   - Explain methodology of analysis
                   - Describe how scores were calculated

                **SCORES TO CALCULATE (MUST BE DATA-DRIVEN):**
                
                1. **Genius Factor Score (0-100)**: Analyze the **options and selected answers**. Calculate based on response consistency across sections and pattern dominance. High consistency across all 5 sections = higher score. Penalty for contradictory answers.
                2. **Retention Risk Score (0-100)**: Calculate based on alignment between user's **Current Position** and their identified **Genius Factor**. High risk if mismatched, low risk if perfectly aligned.
                3. **Mobility Opportunity Score (0-100)**: Calculate based on internal transfer options and industry trends identified in the analysis.

                **IMPORTANT:** Calculate scores based on ALL provided data - assessment patterns, user profile, static context insights, and industry mapping.

                {format_instructions}
                """,
                input_variables=[
                    "primary_factor",
                    "secondary_factor",
                    "scoring_guide",
                    "industries_mapping",
                    "primary_industries",
                    "secondary_industries",
                    "dominant_pattern",
                    "key_insights",
                    "user_name",
                    "user_position",
                    "user_department",
                    "user_salary",
                    "user_skills",
                    "user_experience",
                    "user_education",
                    "assessment_payload",
                    "analysis_data"
                ],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            chain = prompt | llm | parser
            
            # Prepare all data for the report
            inputs = {
                "primary_factor": static_analysis.primary_genius_factor,
                "secondary_factor": static_analysis.secondary_genius_factor or "None identified",
                "scoring_guide": static_analysis.scoring_guide_for_factors[:1500],
                "industries_mapping": static_analysis.industries_for_factors[:2000],
                "primary_industries": ", ".join(static_analysis.primary_industries),
                "secondary_industries": ", ".join(static_analysis.secondary_industries),
                "dominant_pattern": static_analysis.dominant_pattern,
                "key_insights": static_analysis.key_insights,
                "user_name": user_name,
                "user_position": user_position,
                "user_department": user_department,
                "user_salary": user_salary,
                "user_skills": user_skills,
                "user_experience": user_experience,
                "user_education": user_education,
                "assessment_payload": json.dumps(assessment_data, indent=2, default=str)[:3000],
                "analysis_data": analysis_data
            }
            
            # Retry logic for LLM call and parsing
            max_retries = 3
            last_error = None
            for attempt in range(max_retries):
                try:
                    # Generate the report with Pydantic parser
                    report = await chain.ainvoke(inputs)
                    
                    logger.info(f"✅ Professional report generated with scores: "
                               f"G={report.genius_factor_score}, "
                               f"R={report.retention_risk_score}, "
                               f"M={report.mobility_opportunity_score}")
                    
                    return report
                except Exception as e:
                    last_error = e
                    logger.warning(f"Third LLM attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep((attempt + 1) * 2)
            
            raise last_error
            
        except Exception as e:
            logger.error(f"Error generating professional report: {str(e)}")
            raise

    @classmethod
    async def analyze_majority_answers(cls, basic_results: List[Dict[str, Any]], deep_results: Dict[str, Any]) -> str:
        """Improve RAG summary with detailed metrics from deep analysis"""
        try:
            primary_genius = deep_results.get("primary_genius", [])
            primary_factor = primary_genius[0]["name"] if primary_genius and len(primary_genius) > 0 else "Unknown"
            
            secondary_genius = deep_results.get("secondary_genius", [])
            secondary_factor = secondary_genius[0]["name"] if secondary_genius and len(secondary_genius) > 0 else "None"
            
            overlap = deep_results.get("talent_passion_overlap_pct", 0)
            confidence = deep_results.get("confidence_level", "Unknown")
            hybrid = deep_results.get("hybrid_classification", "None")
            
            summary = (
                f"PRIMARY GENIUS: {primary_factor} (Dominance: {primary_genius[0].get('percentage', 0) if primary_genius else 0}%)\n"
                f"SECONDARY GENIUS: {secondary_factor}\n"
                f"TALENT-PASSION OVERLAP: {overlap:.1f}%\n"
                f"ASSESSMENT CONFIDENCE: {confidence}\n"
                f"HYBRID CLASSIFICATION: {hybrid}\n"
            )
            return summary
        except Exception as e:
            logger.error(f"Error in analyze_majority_answers: {str(e)}")
            return "Analysis completed."

    @classmethod
    async def generate_career_recommendation(
        cls, 
        analysis_result: str, 
        all_answers: Any, 
        user_id: str = None, 
        payload_data: Dict = None
    ) -> Dict[str, Any]:
        """Main method: Two-step professional report generation with Pydantic parsers"""
        try:
            logger.info("Starting two-step professional report generation with Pydantic parsers...")
            
            # Fetch user data
            user_data = {}
            if user_id:
                user_data = await cls._fetch_user_data(user_id)
            
            # --- STEP 1: Extract genius factors and related info from static context ---
            logger.info("Step 1: Extracting static context info with Pydantic parser...")
            static_analysis = await cls._first_llm_extract_static_info(payload_data, analysis_summary=analysis_result)
            
            # --- STEP 2: Generate complete professional report with Pydantic parser ---
            logger.info("Step 2: Generating professional report with Pydantic parser...")
            report = await cls._third_llm_generate_professional_report(
                static_analysis,
                user_data,
                payload_data,
                analysis_data=analysis_result
            )
            
            # Convert report to dict for database compatibility
            report_dict = report.dict()
            
            # Prepare genius factors list
            genius_factors = [static_analysis.primary_genius_factor]
            if static_analysis.secondary_genius_factor:
                genius_factors.append(static_analysis.secondary_genius_factor)
            
            # Prepare risk_analysis JSON for database (matches your schema)
            risk_analysis = {
                "analysis_summary": static_analysis.key_insights,
                "scores": {
                    "genius_factor_score": report.genius_factor_score,
                    "retention_risk_score": report.retention_risk_score,
                    "mobility_opportunity_score": report.mobility_opportunity_score
                },
                "trends": {
                    "retention_trends": "Based on professional analysis of role alignment",
                    "mobility_trends": f"Opportunities identified in {len(static_analysis.primary_industries)} primary industries",
                    "risk_factors": report.current_role_alignment_analysis.retention_risk_factors
                },
                "recommendations": report.retention_and_mobility_strategies.retention_strategies[:3],
                "genius_factors": genius_factors,
                "company": "Fortune 1000 Company"
            }
            
            # Prepare final response
            response = {
                "status": "success",
                "report": report_dict,
                "risk_analysis": risk_analysis,
                "context_analysis": {
                    "primary_genius_factor": static_analysis.primary_genius_factor,
                    "secondary_genius_factor": static_analysis.secondary_genius_factor,
                    "dominant_pattern": static_analysis.dominant_pattern,
                    "key_insights": static_analysis.key_insights,
                    "static_context_used": True,
                    "report_completeness": "Professional extensive report with Pydantic validation"
                }
            }
            
            logger.info(f"✅ Complete report generated with Pydantic validation")
            return response
            
        except Exception as e:
            logger.exception(f"Error generating professional report: {str(e)}")
            return {
                "status": "error", 
                "error": str(e),
                "message": "Professional report generation failed."
            }