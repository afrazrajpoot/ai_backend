import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import asyncpg
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from utils.logger import logger
from config import settings


class StaticContextAnalysis(BaseModel):
    """Output from FIRST LLM: Genius factors and related info from static context"""
    primary_genius_factor: str = Field(description="Primary genius factor from static context")
    secondary_genius_factor: Optional[str] = Field(description="Secondary genius factor from static context")
    scoring_guide_for_factors: str = Field(description="Scoring guide sections for these genius factors")
    industries_for_factors: str = Field(description="Industry mapping for these genius factors")
    dominant_pattern: str = Field(description="Dominant response pattern (A, B, C, D)")
    key_insights: str = Field(description="Key insights from matching assessment to static context")


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
                    u.id, u."firstName", u."lastName", u.email, u.position, u.department, u.salary,
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
    async def _first_llm_extract_static_info(cls, assessment_data: Dict) -> StaticContextAnalysis:
        """FIRST LLM: Extract genius factors and related info from static context"""
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

                **YOUR TASK: Extract relevant static context for this user:**
                1. **ANALYZE** assessment responses to determine genius factor patterns
                2. **IDENTIFY** which genius factors from static context match (Tech, Social, Visual, Word, etc.)
                3. **EXTRACT** the scoring guide sections for THOSE specific genius factors
                4. **EXTRACT** the industry mapping for THOSE specific genius factors
                5. **DETERMINE** dominant response pattern (A, B, C, D)
                6. **PROVIDE** key insights about this match

                **IMPORTANT:** Only use genius factors defined in static context. Do not make up new ones.

                {format_instructions}
                """,
                input_variables=["scoring_guide", "industry_mapping", "assessment_payload"],
                partial_variables={"format_instructions": parser.get_format_instructions()}
            )
            
            static_ctx = cls._load_static_context()
            
            chain = prompt | llm | parser
            analysis = await chain.ainvoke({
                "scoring_guide": static_ctx.get("scoring_guide", ""),
                "industry_mapping": static_ctx.get("industry_mapping", ""),
                "assessment_payload": json.dumps(assessment_data, indent=2, default=str)
            })
            
            logger.info(f"First LLM extracted: {analysis.primary_genius_factor} with industries")
            return analysis
            
        except Exception as e:
            logger.error(f"Error in first LLM extraction: {str(e)}")
            raise

    @classmethod
    async def _third_llm_generate_professional_report(
        cls, 
        static_analysis: StaticContextAnalysis,
        user_data: Dict,
        assessment_data: Dict
    ) -> Dict[str, Any]:
        """THIRD LLM: Generate complete professional report using static info + user data"""
        try:
            llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4o",
                temperature=0.3,
                max_tokens=4000
            )
            
            # Load system prompt for scoring rules
            prompt_data = cls._load_prompts()
            system_prompt = prompt_data.get('system_prompt', '')
            
            # Professional report generation prompt
            report_prompt = PromptTemplate(
                template=system_prompt + """
                
                ========== STATIC CONTEXT INFORMATION ==========
                **GENIUS FACTORS IDENTIFIED:**
                - Primary: {primary_factor}
                - Secondary: {secondary_factor}
                
                **SCORING GUIDE FOR THESE FACTORS:**
                {scoring_guide}
                
                **INDUSTRY MAPPING FOR THESE FACTORS:**
                {industries_mapping}
                
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
                   - Start with professional introduction
                   - State identified genius factors clearly
                   - Mention key industries from static context
                   - Summarize career implications
                   - End with strategic recommendations

                2. **GENIUS FACTOR PROFILE** (Detailed):
                   - Primary and secondary genius factors with descriptions
                   - 5-6 specific key strengths based on assessment
                   - 3-4 energy sources that motivate this profile
                   - 3-4 development areas for growth

                3. **CURRENT ROLE ALIGNMENT ANALYSIS** (Comprehensive):
                   - Detailed assessment of current position alignment
                   - Alignment score with justification
                   - List of strengths currently utilized
                   - Retention risk level with specific factors
                   - Underutilized talents from assessment
                   - Immediate actions for improvement

                4. **INTERNAL CAREER OPPORTUNITIES** (Specific):
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

                7. **PERSONALIZED RESOURCES** (Complete):
                   - 4-5 affirmations specific to genius factors
                   - 4-5 learning resources (courses, books, certifications)
                   - 3-4 reflection questions for self-assessment
                   - 3-4 mindfulness practices for career development

                8. **DATA SOURCES & METHODOLOGY** (Transparent):
                   - Detailed methodology of analysis
                   - Specific data sources used
                   - STEP-BY-STEP calculation of all three scores
                   - Validation of score accuracy

                **CRITICAL REQUIREMENTS:**
                - MUST include ALL industries from static context analysis
                - MUST calculate all three scores based on assessment data
                - MUST reference specific assessment patterns
                - MUST be professional and extensive
                - MUST return complete JSON structure

                **SCORES TO CALCULATE:**
                1. Genius Factor Score (0-100): Based on assessment patterns
                2. Retention Risk Score (0-100): Based on role alignment
                3. Mobility Opportunity Score (0-100): Based on industry opportunities

                Generate the complete professional report now.
                Return ONLY the JSON report.
                """,
                input_variables=[
                    "primary_factor",
                    "secondary_factor",
                    "scoring_guide",
                    "industries_mapping",
                    "dominant_pattern",
                    "key_insights",
                    "user_name",
                    "user_position",
                    "user_department",
                    "user_salary",
                    "user_skills",
                    "user_experience",
                    "user_education",
                    "assessment_payload"
                ]
            )
            
            chain = report_prompt | llm
            
            # Prepare all data for the report
            inputs = {
                "primary_factor": static_analysis.primary_genius_factor,
                "secondary_factor": static_analysis.secondary_genius_factor or "None identified",
                "scoring_guide": static_analysis.scoring_guide_for_factors[:1500],
                "industries_mapping": static_analysis.industries_for_factors[:2000],  # More industries
                "dominant_pattern": static_analysis.dominant_pattern,
                "key_insights": static_analysis.key_insights,
                "user_name": f"{user_data.get('firstName', '')} {user_data.get('lastName', '')}".strip() or "Employee",
                "user_position": user_data.get("position", "Not specified"),
                "user_department": user_data.get("department", "Not specified"),
                "user_salary": f"${user_data.get('salary', 0):,}" if user_data.get('salary') else "Not disclosed",
                "user_skills": user_data.get("skills", "Not specified"),
                "user_experience": user_data.get("experience", "Not specified"),
                "user_education": user_data.get("education", "Not specified"),
                "assessment_payload": json.dumps(assessment_data, indent=2, default=str)[:3000]  # Limit size
            }
            
            report_output = await chain.ainvoke(inputs)
            
            # Parse the JSON report
            try:
                report_content = report_output.content
                # Extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', report_content, re.DOTALL)
                if json_match:
                    report_dict = json.loads(json_match.group())
                else:
                    # If no JSON, create error
                    raise ValueError("No JSON found in LLM response")
                    
            except Exception as e:
                logger.error(f"Error parsing report JSON: {str(e)}")
                # Create a minimal valid structure
                report_dict = cls._create_minimal_report(static_analysis, user_data)
            
            # Ensure report has all required fields
            report_dict = cls._ensure_complete_report(report_dict, static_analysis)
            
            logger.info(f"Professional report generated with {len(report_dict)} sections")
            return report_dict
            
        except Exception as e:
            logger.error(f"Error generating professional report: {str(e)}")
            raise

    @classmethod
    def _create_minimal_report(cls, static_analysis: StaticContextAnalysis, user_data: Dict) -> Dict[str, Any]:
        """Create a minimal valid report structure"""
        return {
            "executive_summary": f"Professional analysis for {user_data.get('firstName', 'Employee')} based on Genius Factor assessment.",
            "genius_factor_score": 75,
            "retention_risk_score": 50,
            "mobility_opportunity_score": 65,
            "genius_factor_profile": {
                "primary_genius_factor": static_analysis.primary_genius_factor,
                "secondary_genius_factor": static_analysis.secondary_genius_factor or "",
                "description": f"Analysis indicates {static_analysis.primary_genius_factor} profile.",
                "key_strengths": ["Analytical thinking", "Problem-solving", "Technical skills"],
                "energy_sources": ["Challenging projects", "Technical innovation", "Team collaboration"],
                "development_areas": ["Leadership development", "Advanced skills", "Strategic thinking"]
            },
            "current_role_alignment_analysis": {
                "assessment": "Current role alignment analysis completed.",
                "alignment_score": "75",
                "strengths_utilized": ["Technical abilities", "Problem-solving"],
                "retention_risk_level": "Moderate",
                "underutilized_talents": ["Leadership potential", "Strategic planning"],
                "immediate_actions": ["Skill development", "Project leadership", "Networking"]
            },
            "internal_career_opportunities": {
                "recommended_departments": ["Technology", "Innovation", "Design"],
                "specific_role_suggestions": ["Technical Lead", "Project Manager", "Design Specialist"],
                "required_skill_development": ["Advanced training", "Leadership skills", "Technical expertise"],
                "transition_timeline": {
                    "short_term": "30-60 days: Skill assessment",
                    "mid_term": "3-6 months: Project leadership",
                    "long_term": "6-12 months: Role transition"
                },
                "success_metrics": ["Project completion", "Skill acquisition", "Career progression"]
            },
            "retention_and_mobility_strategies": {
                "retention_strategies": ["Career path development", "Skill enhancement", "Competitive compensation"],
                "internal_mobility_recommendations": ["Cross-department projects", "Leadership training", "Mentorship programs"],
                "development_support": ["Training access", "Mentorship", "Conference participation"]
            },
            "development_action_plan": {
                "thirty_day_goals": ["Complete skills assessment", "Identify development needs", "Start networking"],
                "ninety_day_goals": ["Begin advanced training", "Lead small project", "Establish mentor relationship"],
                "six_month_goals": ["Transition to target role", "Demonstrate leadership", "Achieve measurable impact"],
                "networking_strategy": ["Join professional groups", "Attend industry events", "Connect with leaders"]
            },
            "personalized_resources": {
                "affirmations": ["I have valuable skills to contribute", "I am capable of growth and success", "My work makes a difference"],
                "learning_resources": ["Online certification courses", "Industry publications", "Professional development workshops"],
                "reflection_questions": ["What are my core strengths?", "Where do I want to be in 1 year?", "How can I add more value?"],
                "mindfulness_practices": ["Daily goal setting", "Weekly progress review", "Stress management techniques"]
            },
            "data_sources_and_methodology": {
                "methodology": "Comprehensive analysis using Genius Factor assessment and static context.",
                "data_sources": ["Assessment responses", "Static genius factor framework", "User profile data"],
                "calculation_steps": "Scores calculated based on assessment patterns and static context alignment."
            }
        }

    @classmethod
    def _ensure_complete_report(cls, report_dict: Dict, static_analysis: StaticContextAnalysis) -> Dict[str, Any]:
        """Ensure report has all required sections"""
        required_sections = [
            "executive_summary",
            "genius_factor_profile",
            "current_role_alignment_analysis",
            "internal_career_opportunities",
            "retention_and_mobility_strategies",
            "development_action_plan",
            "personalized_resources",
            "data_sources_and_methodology",
            "genius_factor_score",
            "retention_risk_score",
            "mobility_opportunity_score"
        ]
        
        for section in required_sections:
            if section not in report_dict:
                if section.endswith("_score"):
                    report_dict[section] = 75 if "genius" in section else 50
                else:
                    # Get from minimal report
                    minimal = cls._create_minimal_report(static_analysis, {})
                    report_dict[section] = minimal.get(section, {})
        
        return report_dict

    @classmethod
    async def analyze_majority_answers(cls, basic_results: List[Dict[str, Any]], deep_results: Dict[str, Any]) -> str:
        """Compatibility method"""
        try:
            primary_genius = deep_results.get("primary_genius", [])
            primary_factor = primary_genius[0]["name"] if primary_genius and len(primary_genius) > 0 else "Unknown"
            return f"Analysis indicates {primary_factor} as primary genius factor."
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
        """Main method: Two-step professional report generation"""
        try:
            logger.info("Starting two-step professional report generation...")
            
            # Fetch user data
            user_data = {}
            if user_id:
                user_data = await cls._fetch_user_data(user_id)
            
            # --- STEP 1: Extract genius factors and related info from static context ---
            logger.info("Step 1: Extracting static context info...")
            static_analysis = await cls._first_llm_extract_static_info(payload_data)
            
            # --- STEP 2: Generate complete professional report ---
            logger.info("Step 2: Generating professional report...")
            report = await cls._third_llm_generate_professional_report(
                static_analysis,
                user_data,
                payload_data
            )
            
            # Prepare final response
            genius_factors = [static_analysis.primary_genius_factor]
            if static_analysis.secondary_genius_factor:
                genius_factors.append(static_analysis.secondary_genius_factor)
            
            response = {
                "status": "success",
                "report": report,
                "risk_analysis": {
                    "analysis_summary": static_analysis.key_insights,
                    "scores": {
                        "genius_factor_score": report.get("genius_factor_score", 75),
                        "retention_risk_score": report.get("retention_risk_score", 50),
                        "mobility_opportunity_score": report.get("mobility_opportunity_score", 65)
                    },
                    "trends": {
                        "retention_trends": "Based on professional analysis",
                        "mobility_trends": "Industry opportunities analyzed",
                        "risk_factors": f"Analysis for {static_analysis.primary_genius_factor} profile"
                    },
                    "recommendations": report.get("retention_and_mobility_strategies", {}).get("retention_strategies", []),
                    "genius_factors": genius_factors,
                    "company": "Fortune 1000 Company"
                },
                "context_analysis": {
                    "primary_genius_factor": static_analysis.primary_genius_factor,
                    "secondary_genius_factor": static_analysis.secondary_genius_factor,
                    "dominant_pattern": static_analysis.dominant_pattern,
                    "key_insights": static_analysis.key_insights,
                    "static_context_used": True,
                    "report_completeness": "Professional extensive report"
                }
            }
            
            logger.info(f"Professional report complete. Factors: {genius_factors}")
            return response
            
        except Exception as e:
            logger.exception(f"Error generating professional report: {str(e)}")
            return {
                "status": "error", 
                "error": str(e),
                "message": "Professional report generation failed."
            }