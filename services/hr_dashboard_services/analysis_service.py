from typing import List, Dict, Any
from utils.models import DepartmentInput, AnalysisResponse
from .internvation_services import LLMService

class AnalysisService:
    def __init__(self):
        self.llm_service = LLMService()

    async def analyze_departments(self, departments: List[DepartmentInput]) -> AnalysisResponse:
        """Main analysis service method"""
        
        # Convert Pydantic models to dict for LLM processing
        departments_data = [dept.dict() for dept in departments]
        
        # Get analysis from LLM
        llm_result = self.llm_service.analyze_retention_risk(departments_data)
        
        # Convert to response model
        return AnalysisResponse(**llm_result)