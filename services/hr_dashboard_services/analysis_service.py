from typing import List
from utils.models import DepartmentInput, AnalysisResponse
from .internvation_services import LLMService

class AnalysisService:
    def __init__(self):
        self.llm_service = LLMService()
        self.initialized = False

    async def initialize(self):
        """Initialize the LLM service with database"""
        await self.llm_service.initialize_db()
        self.initialized = True

    async def analyze_departments(self, departments: List[DepartmentInput]) -> AnalysisResponse:
        """Main analysis service method with database saving"""
        
        if not self.initialized:
            await self.initialize()
        
        # Convert Pydantic models to dict for LLM processing
        departments_data = [dept.dict() for dept in departments]
        
        # Get analysis from LLM and save to database
        llm_result = await self.llm_service.analyze_and_save_retention_risk(departments_data)
        
        # Convert to response model
        return AnalysisResponse(**llm_result)

    async def close(self):
        """Cleanup resources"""
        await self.llm_service.close_db()