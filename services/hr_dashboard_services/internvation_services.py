import json
import logging
from typing import Dict, Any, List
import asyncpg
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from config import settings

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name="gpt-4o-mini",
            temperature=0.1
        )
        self.pool = None

    async def initialize_db(self):
        """Initialize database connection pool"""
        self.pool = await asyncpg.create_pool(
            dsn="postgresql://postgres:root@localhost:5432/genius_factor",
            min_size=1,
            max_size=10
        )
        logger.info("✅ Database connection pool ready")

    async def save_analysis_result(
        self,
        hrid: str,
        department_name: str,
        ai_response: Dict[str, Any],
        risk_score: float
    ):
        """Save analysis result to database"""
        insert_query = """
        INSERT INTO "AnalysisResult"  (hrid, department_name, ai_response, risk_score)
        VALUES ($1, $2, $3, $4)
        """
        async with self.pool.acquire() as connection:
            await connection.execute(
                insert_query,
                hrid,
                department_name,
                json.dumps(ai_response),
                risk_score
            )
        logger.info(f"✅ Saved analysis for HRID={hrid}, Dept={department_name}")

    async def analyze_retention_risk(self, departments_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze retention risk using LLM"""
        SYSTEM_PROMPT = """You are an expert HR analytics and retention risk specialist. 
        Analyze the provided department data and generate concise, actionable recommendations.

        Output format must be valid JSON with this structure:
        {
            "overall_risk_score": float,
            "department_recommendations": [
                {
                    "department": "string",
                    "risk_level": "low/medium/high",
                    "retention_score": float,
                    "mobility_opportunities": ["string"],
                    "recommendations": ["string"],
                    "action_items": ["string"]
                }
            ],
            "summary": "string"
        }

        Keep recommendations point-to-point, actionable, and specific to each department's data.
        """

        departments_str = "\n\n".join([
            f"Department: {dept['name']}\n"
            f"Employee Count: {dept['employee_count']}\n"
            f"Avg Scores: {json.dumps(dept['metrics']['avg_scores'])}\n"
            f"Engagement Distribution: {json.dumps(dept['metrics']['engagement_distribution'])}\n"
            f"Retention Risk Distribution: {json.dumps(dept['metrics']['retention_risk_distribution'])}\n"
            f"Mobility Trend: {json.dumps(dept['metrics']['mobility_trend'])}\n"
            f"Skills Alignment: {json.dumps(dept['metrics']['skills_alignment_distribution'])}"
            for dept in departments_data
        ])

        user_prompt = f"""
        Department Data Analysis:
        
        {departments_str}
        
        Analyze retention risk, mobility opportunities, and provide specific recommendations for each department.
        Return only valid JSON without any additional text.
        """

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_prompt)
        ]

        # FIX: async invoke
        response = await self.llm.ainvoke(messages)
        result = json.loads(response.content)
        return result

    async def analyze_and_save_retention_risk(self, departments_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        # print(departments_data,'my department data')
        """
        Analyze retention risk and save results to database for each department
        """
        # Perform LLM analysis
        analysis_result = await self.analyze_retention_risk(departments_data)

        # Save each department's analysis to database
        for dept_recommendation in analysis_result.get('department_recommendations', []):
            department_name = dept_recommendation.get('department', '').strip()
            risk_score = dept_recommendation.get('retention_score', 0.0)

            matched = False
            for dept in departments_data:
                if dept['name'].lower().strip() == department_name.lower():
                    hrid = dept.get('hrId')
                    matched = True
                    if hrid:
                        await self.save_analysis_result(
                            hrid=hrid,
                            department_name=department_name,
                            ai_response=dept_recommendation,
                            risk_score=risk_score
                        )
                    else:
                        logger.warning(f"⚠️ No HRID found for dept {department_name}")
                    break

            if not matched:
                logger.warning(f"⚠️ No match found for dept {department_name} in input data")

        return analysis_result

    async def close_db(self):
        """Close database connection pool"""
        if self.pool:
            await self.pool.close()
            logger.info("✅ Database connection closed")