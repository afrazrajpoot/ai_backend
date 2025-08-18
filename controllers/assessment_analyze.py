from fastapi import HTTPException
from schemas.assessment import AssessmentData
from services.ai_service import AIService
from utils.analyze_assessment import analyze_assessment_data
from utils.logger import logger
from typing import Dict, Any
class AssessmentController:
    @staticmethod
    async def analyze_assessment(input_data: AssessmentData) -> Dict[str, Any]:
        """
        Endpoint for assessment analysis with pure RAG retrieval
        """
        try:
            logger.info("Starting assessment analysis")

            # 1. Get basic assessment results
            basic_results = analyze_assessment_data(input_data.data)

            # 2. Enhance with document retrieval from vector store
            rag_results = await AIService.analyze_majority_answers(basic_results)
            

            logger.info("Assessment analysis completed successfully")

            return {
                "status": "success",
                "basicResults": basic_results,
                "ragAnalysis": rag_results
            }

        except Exception as e:
            logger.error(f"Error in analyze_assessment: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def get_career_recommendations(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Endpoint for document-based career recommendations
        """
        try:
            logger.info("Generating document-based career recommendations")
            
            recommendations = await AIService.generate_career_recommendation(analysis_data)
            
            logger.info("Recommendations generated successfully")
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in get_career_recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
















#         from fastapi import HTTPException
# from schemas.assessment import AssessmentData
# from services.ai_service import AIService
# from utils.analyze_assessment import analyze_assessment_data
# from utils.logger import logger
# from typing import Dict, Any

# class AssessmentController:
#     @staticmethod
#     async def analyze_assessment(input_data: AssessmentData) -> Dict[str, Any]:
#         """
#         Endpoint for assessment analysis with RAG retrieval and professional report generation
#         """
#         try:
#             logger.info("Starting assessment analysis")

#             # 1. Get basic assessment results
#             basic_results = analyze_assessment_data(input_data.data)

#             # 2. Enhance with document retrieval from vector store
#             rag_results = await AIService.analyze_majority_answers(basic_results)
            
#             # 3. Generate professional career recommendation report
#             recommendations = await AIService.generate_career_recommendation(rag_results)
            
#             if recommendations.get("status") != "success":
#                 logger.error(f"Failed to generate recommendations: {recommendations.get('message', 'Unknown error')}")
#                 raise HTTPException(status_code=500, detail="Failed to generate career recommendations")

#             logger.info("Assessment analysis and report generation completed successfully")

#             # Return only the report and metadata, excluding raw RAG data
#             return {
#                 "status": "success",
#                 "report": recommendations.get("report"),
#                 "metadata": recommendations.get("metadata")
#             }

#         except Exception as e:
#             logger.error(f"Error in analyze_assessment: {str(e)}")
#             raise HTTPException(status_code=500, detail=str(e))

#     @staticmethod
#     async def get_career_recommendations(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
#         """
#         Endpoint for generating professional career recommendation report
#         """
#         try:
#             logger.info("Generating professional career recommendation report")
            
#             # Call AIService to generate the report using Azure Chat model
#             recommendations = await AIService.generate_career_recommendation(analysis_data)
            
#             if recommendations.get("status") != "success":
#                 logger.error(f"Failed to generate recommendations: {recommendations.get('message', 'Unknown error')}")
#                 raise HTTPException(status_code=500, detail="Failed to generate career recommendations")
            
#             logger.info("Recommendations generated successfully")
            
#             # Return only the report and metadata, excluding raw RAG data
#             return {
#                 "status": "success",
#                 "report": recommendations.get("report"),
#                 "metadata": recommendations.get("metadata")
#             }
            
#         except Exception as e:
#             logger.error(f"Error in get_career_recommendations: {str(e)}")
#             raise HTTPException(status_code=500, detail=str(e))