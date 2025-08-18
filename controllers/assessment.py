from services.ai_service import AIService
from services.db_service import DBService
from schemas.assessment import AssessInput
from utils.logger import logger
from fastapi import HTTPException
class AssessmentController:
    @staticmethod
    async def assess_employee(input_data: AssessInput):
        try:
            logger.info(f"Processing assessment for userId: {input_data.userId}")
            
            # Get AI assessment
            assessment_result = await AIService.generate_assessment(input_data.answers)
            
            # Save to database
            await DBService.save_assessment(
                user_id=input_data.userId,
                results=assessment_result["results"],
                overall_score=assessment_result["overallScore"],
                message=assessment_result["message"]
            )
            
            return assessment_result
            
        except Exception as e:
            logger.error(f"Error in assess_employee: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def get_assessments(userId: str):
        try:
            return await DBService.get_assessments_by_user(userId)
        except Exception as e:
            logger.error(f"Error in get_assessments: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        


















