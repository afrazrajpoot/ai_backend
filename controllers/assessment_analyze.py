from fastapi import HTTPException
from schemas.assessment import AssessmentData
from services.ai_service import AIService
from utils.analyze_assessment import analyze_assessment_data
from utils.logger import logger
from services.notification_service import NotificationService
from services.db_service import DBService
from typing import Dict, Any
import requests
import asyncio

class AssessmentController:
    
    @staticmethod
    def send_to_nextjs(assessment_data: dict):
        """
        Send assessment data to Next.js API endpoint
        """
        try:
            url = "https://geniusfactor.ai/api/generate-report"
            headers = {"Content-Type": "application/json"}
            
            response = requests.post(url, json=assessment_data, headers=headers)
            response.raise_for_status()

            logger.info(f"Successfully sent data to Next.js: {response.json()}")
            return response.json()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Next.js API: {e}")
            return {"status": "error", "message": str(e)}

    @staticmethod
    async def save_to_database(assessment_data: dict):
        """
        Save assessment data to the database using Prisma
        """
        try:
            db = await DBService._get_db()
            
            # Extract report data
            report = assessment_data.get("report", {})
            
            # Get user details for hrId and department
            user = await db.user.find_unique(
                where={"id": assessment_data["userId"]}
            )
            
            if not user:
                raise ValueError("User not found")
            
            # Create the report
            saved_report = await db.individualemployeereport.create(
                data={
                    "userId": assessment_data["userId"],
                    "hrId": user.hrId,
                    "departement": user.department[-1] if user.department else "General",
                    "executiveSummary": report.get("executive_summary", ""),
                    "geniusFactorProfileJson": report.get("genius_factor_profile", {}),
                    "currentRoleAlignmentAnalysisJson": report.get("current_role_alignment_analysis", {}),
                    "internalCareerOpportunitiesJson": report.get("internal_career_opportunities", {}),
                    "retentionAndMobilityStrategiesJson": report.get("retention_and_mobility_strategies", {}),
                    "developmentActionPlanJson": report.get("development_action_plan", {}),
                    "personalizedResourcesJson": report.get("personalized_resources", {}),
                    "dataSourcesAndMethodologyJson": report.get("data_sources_and_methodology", {}),
                    "geniusFactorScore": report.get("genius_factor_score", 0),
                    "risk_analysis": assessment_data.get("risk_analysis", {})
                }
            )
            
            logger.info(f"Successfully saved report to database: {saved_report.id}")
            return {"status": "success", "report_id": saved_report.id}
            
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    async def analyze_assessment(input_data: AssessmentData) -> Dict[str, Any]:
        """
        Endpoint for assessment analysis with real-time notifications and Next.js integration
        """
        try:
            logger.info("Starting assessment analysis")

            # Send progress notification via Socket.IO
            # await NotificationService.send_user_notification(
            #     input_data.userId,
            #     {
            #         'message': 'Assessment analysis started',
            #         'progress': 0,
            #         'status': 'processing'
            #     }
            # )

            # 1. Get basic assessment results
            basic_results = analyze_assessment_data(input_data.data)
            # print(basic_results,'basic result')
            # Send progress update
            # await NotificationService.send_user_notification(
            #     input_data.userId,
            #     {
            #         'message': 'Basic analysis completed',
            #         'progress': 33,
            #         'status': 'processing'
            #     }
            # )

            # 2. Enhance with document retrieval from vector store
            rag_results = await AIService.analyze_majority_answers(basic_results)
            
            # Send progress update
            # await NotificationService.send_user_notification(
            #     input_data.userId,
            #     {
            #         'message': 'Advanced analysis completed',
            #         'progress': 66,
            #         'status': 'processing'
            #     }
            # )

            # 3. Generate professional career recommendation report
            recommendations = await AIService.generate_career_recommendation(rag_results)
            
            if recommendations.get("status") != "success":
                error_msg = f"Failed to generate recommendations: {recommendations.get('message', 'Unknown error')}"
                logger.error(error_msg)
                
                # Send error notification
                await NotificationService.send_user_notification(
                    input_data.userId,
                    {
                        'message': 'Analysis failed',
                        'progress': 100,
                        'status': 'error',
                        'error': error_msg
                    }
                )
                
                raise HTTPException(status_code=500, detail="Failed to generate career recommendations")

            # Prepare final result
            final_result = {
                "status": "success",
                "userId": input_data.userId,
                "report": recommendations.get("report"),
                "risk_analysis": recommendations.get("risk_analysis"),
                "metadata": recommendations.get("metadata")
            }

            # 4. Save data to database (synchronous call)
            db_response = await AssessmentController.save_to_database(final_result)
            
            if db_response.get("status") == "error":
                logger.warning(f"Database save failed but proceeding: {db_response.get('message')}")
                # Continue even if database save fails, but log the warning

            # Send success notification via Socket.IO
            await NotificationService.send_user_notification(
                input_data.userId,
                {
                    'message': 'Assessment analysis completed successfully!',
                    'progress': 100,
                    'status': 'completed',
                    'report_id': db_response.get("report_id"),
                    'db_response': db_response
                }
            )

            logger.info("Assessment analysis, report generation, and Next.js integration completed successfully")

            return final_result

        except Exception as e:
            logger.error(f"Error in analyze_assessment: {str(e)}")
            
            # Send error notification
            await NotificationService.send_user_notification(
                input_data.userId,
                {
                    'message': 'Assessment analysis failed',
                    'progress': 100,
                    'status': 'error',
                    'error': str(e)
                }
            )
            
            raise HTTPException(status_code=500, detail=str(e))

    @staticmethod
    async def get_career_recommendations(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Endpoint for generating professional career recommendation report
        """
        try:
            logger.info("Generating professional career recommendation report")
            
            # Call AIService to generate the report using Azure Chat model
            recommendations = await AIService.generate_career_recommendation(analysis_data)
            
            if recommendations.get("status") != "success":
                logger.error(f"Failed to generate recommendations: {recommendations.get('message', 'Unknown error')}")
                raise HTTPException(status_code=500, detail="Failed to generate career recommendations")
            
            logger.info("Recommendations generated successfully")
            
            # Return only the report and metadata, excluding raw RAG data
            return {
                "status": "success",
                "report": recommendations.get("report"),
                "metadata": recommendations.get("metadata")
            }
            
        except Exception as e:
            logger.error(f"Error in get_career_recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))