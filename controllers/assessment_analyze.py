from fastapi import HTTPException
from schemas.assessment import AssessmentData
from services.ai_service import AIService
# from services.database_notification_service import DatabaseNotificationService
from services.db_service import DBService
from utils.analyze_assessment import analyze_assessment_data
from utils.logger import logger
from services.notification_service import NotificationService
from typing import Dict, Any
import httpx

# Singleton AIService instance (assumed to be defined elsewhere)
ai_service = AIService()

# Singleton DatabaseNotificationService instance
db_notification_service = DBService()

class AssessmentController:
    
    @staticmethod
    async def send_to_nextjs(assessment_data: dict):
        """
        Send assessment data to Next.js API endpoint asynchronously
        """
        try:
            url = "http://localhost:3000/api/generate-report"
            headers = {"Content-Type": "application/json"}
            
            async with httpx.AsyncClient() as client:
                response = await client.post(url, json=assessment_data, headers=headers)
                response.raise_for_status()

            logger.info(f"Successfully sent data to Next.js: {response.json()}")
            return response.json()
            
        except Exception as e:
            logger.error(f"Error calling Next.js API: {e}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    async def analyze_assessment(input_data: AssessmentData) -> Dict[str, Any]:
        """
        Endpoint for assessment analysis with real-time notifications and Next.js integration
        """
        try:
            logger.info(f"Starting assessment analysis for userId: {input_data.userId}, hrId: {input_data.hrId}")

            # Validate input data for notification
            notification_data = {
                'employeeId': input_data.userId,
                'hrId': input_data.hrId,
                'employeeName': input_data.employeeName,
                'employeeEmail': input_data.employeeEmail,
                'message': 'Assessment analysis completed successfully!',
                'status':'unread'
            }
            for key, value in notification_data.items():
                if not isinstance(value, str) or not value.strip():
                    logger.error(f"Invalid notification data: {key} is empty or not a string")
                    await NotificationService.send_user_notification(
                        input_data.userId,
                        input_data.hrId,
                        {
                            'message': 'Invalid notification data',
                            'progress': 0,
                            'status': 'error',
                            'error': f"Field {key} is invalid"
                        }
                    )
                    raise HTTPException(status_code=400, detail=f"Invalid notification data: {key}")

            # 1. Get basic assessment results
            try:
                basic_results = analyze_assessment_data(input_data.data)
                logger.info("Basic analysis completed")
            except Exception as e:
                logger.error(f"Failed to analyze assessment data: {str(e)}")
                await NotificationService.send_user_notification(
                    input_data.userId,
                    input_data.hrId,
                    {
                        'message': 'Basic analysis failed',
                        'progress': 100,
                        'status': 'error',
                        'error': str(e)
                    }
                )
                raise HTTPException(status_code=500, detail=str(e))

            # 2. Enhance with document retrieval from vector store
            try:
                rag_results = await ai_service.analyze_majority_answers(basic_results)
            except Exception as e:
                logger.error(f"Failed to analyze majority answers: {str(e)}")
                await NotificationService.send_user_notification(
                    input_data.userId,
                    input_data.hrId,
                    {
                        'message': 'Failed to perform advanced analysis',
                        'progress': 100,
                        'status': 'error',
                        'error': str(e)
                    }
                )
                raise HTTPException(status_code=500, detail=str(e))
            
            # 3. Generate professional career recommendation report
            try:
                recommendations = await ai_service.generate_career_recommendation(rag_results)
            except Exception as e:
                logger.error(f"Failed to generate recommendations: {str(e)}")
                await NotificationService.send_user_notification(
                    input_data.userId,
                    input_data.hrId,
                    {
                        'message': 'Failed to generate recommendations',
                        'progress': 100,
                        'status': 'error',
                        'error': str(e)
                    }
                )
                raise HTTPException(status_code=500, detail=str(e))
            
            if recommendations.get("status") != "success":
                error_msg = f"Failed to generate recommendations: {recommendations.get('message', 'Unknown error')}"
                logger.error(error_msg)
                
                await NotificationService.send_user_notification(
                    input_data.userId,
                    input_data.hrId,
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

            # 4. Send data to Next.js API (asynchronous call)
            nextjs_response = await AssessmentController.send_to_nextjs(final_result)
            
            if nextjs_response.get("status") == "error":
                logger.warning(f"Next.js API call failed but proceeding: {nextjs_response.get('message')}")
                # Continue even if Next.js call fails, but log the warning

            # Send success notification via Socket.IO
            await NotificationService.send_user_notification(
                input_data.userId,
                input_data.hrId,
                {
                    'message': 'Assessment analysis completed successfully!',
                    'employeeName': input_data.employeeName,
                    'employeeEmail': input_data.employeeEmail,
                 
                    'progress': 100,
                    'status': 'unread',
                    # 'report_id': recommendations.get("metadata", {}).get("report_id"),
                    # 'nextjs_response': nextjs_response
                }
            )

            # Save notification to database using DatabaseNotificationService
            try:
                await db_notification_service.save_notification(notification_data)
            except Exception as e:
                logger.error(f"Failed to save notification to database: {str(e)}")
                await NotificationService.send_user_notification(
                    input_data.userId,
                    input_data.hrId,
                    {
                        'message': 'Failed to save notification to database',
                        'progress': 100,
                        'status': 'error',
                        'error': str(e)
                    }
                )
                # Continue even if database save fails, but log and notify

            logger.info("Assessment analysis, report generation, and Next.js integration completed successfully")
            return final_result

        except Exception as e:
            logger.error(f"Error in analyze_assessment: {str(e)}")
            
            await NotificationService.send_user_notification(
                input_data.userId,
                input_data.hrId,
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
            
            # Call AIService to generate the report
            recommendations = await ai_service.generate_career_recommendation(analysis_data)
            
            if recommendations.get("status") != "success":
                logger.error(f"Failed to generate recommendations: {recommendations.get('message', 'Unknown error')}")
                raise HTTPException(status_code=500, detail="Failed to generate career recommendations")
            
            logger.info("Recommendations generated successfully")
            
            return {
                "status": "success",
                "report": recommendations.get("report"),
                "metadata": recommendations.get("metadata")
            }
            
        except Exception as e:
            logger.error(f"Error in get_career_recommendations: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))