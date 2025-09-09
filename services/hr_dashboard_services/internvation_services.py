from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
from typing import Dict, Any
from config import settings

class LLMService:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            model_name="gpt-4o-mini",
            temperature=0.1
        )

    def analyze_retention_risk(self, departments_data: list) -> Dict[str, Any]:
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

        try:
            messages = [
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ]
            
            response = self.llm(messages)
            result = json.loads(response.content)
            return result
            
        except Exception as e:
            raise Exception(f"LLM analysis failed: {str(e)}")