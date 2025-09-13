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
    async def save_to_database(assessment_data: dict):
        """
        Save assessment data to the database using Prisma
        """
        try:
            logger.info("Starting database save process for assessment data")
            db = await DBService._get_db()
            logger.info("Database connection established")
            
            # Extract report data
            report = assessment_data.get("report", {})
            logger.info(f"Extracted report data: {bool(report)}")
            
            # Get user details for hrId and department
            user_id = assessment_data["userId"]
            logger.info(f"Looking up user with ID: {user_id}")
            user = await db.user.find_unique(
                where={"id": user_id}
            )
            
            if not user:
                logger.error(f"User not found with ID: {user_id}")
                raise ValueError("User not found")
            
            logger.info(f"User found: {user.id}, hrId: {user.hrId}, department: {user.department}")
            
            # Prepare data for report creation
            report_data = {
                "userId": user_id,
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
            
            logger.info("Prepared report data for creation")
            
            # Create the report
            logger.info("Creating individual employee report in database...")
            saved_report = await db.individualemployeereport.create(
                data=report_data
            )
            
            logger.info(f"Successfully saved report to database with ID: {saved_report.id}")
            return {"status": "success", "report_id": saved_report.id}
            
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            logger.error(f"Full exception details: {type(e).__name__}: {e}")
            return {"status": "error", "message": str(e)}
    
    @staticmethod
    async def analyze_assessment(input_data: AssessmentData) -> Dict[str, Any]:
        """
        Endpoint for assessment analysis with real-time notifications and Next.js integration
        """
        try:
            logger.info("Starting assessment analysis")
            basic_results = analyze_assessment_data(input_data.data)
            rag_results = await AIService.analyze_majority_answers(basic_results)

            # 3. Generate professional career recommendation report
            recommendations = await AIService.generate_career_recommendation(rag_results)
            print(f"Recommendations generated: {recommendations}")

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
            
            print(f"Recommendations generated")
            # Prepare final result
            final_result = {
                "status": "success",
                "userId": input_data.userId,
                "report": recommendations.get("report"),
                "risk_analysis": recommendations.get("risk_analysis"),
                "metadata": recommendations.get("metadata")
            }
            print(f"Assessment analysis and recommendation generation completed: {final_result}")

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

    # @staticmethod
    # async def get_career_recommendations(analysis_data: Dict[str, Any]) -> Dict[str, Any]:
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