from fastapi import HTTPException
from schemas.assessment import AssessmentData
from services.ai_service import AIService
from utils.analyze_assessment import analyze_assessment_data
from utils.logger import logger
from services.notification_service import NotificationService
from typing import Dict, Any, Union
from prisma import Prisma
import json


class AssessmentController:
    
    @staticmethod
    def sanitize_and_validate_json(obj: Any) -> Union[Dict, List, None]:
        """
        Sanitize keys and validate JSON data for Prisma
        """
        def sanitize_keys(data):
            if isinstance(data, dict):
                # Handle empty dict
                if not data:
                    return {}
                return {str(k).replace("[", "_").replace("]", "_"): sanitize_keys(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [sanitize_keys(i) for i in data]
            elif data is None:
                return None
            else:
                return data

        try:
            sanitized = sanitize_keys(obj)
            
            # Validate that the result is JSON serializable
            json.dumps(sanitized)
            
            # Return None for empty objects to use JsonNullValueInput
            if sanitized == {} or sanitized is None:
                return None
                
            return sanitized
            
        except (TypeError, ValueError) as e:
            logger.error(f"JSON serialization error: {e}")
            logger.error(f"Problematic data: {obj}")
            return None

    @staticmethod
    async def save_to_database(assessment_data: dict):
        """
        Save assessment data to the database using Prisma
        """
        prisma = Prisma()
        await prisma.connect()

        try:
            logger.info("Starting database save process for assessment data")
            logger.info("Database connection established")
            
            # Extract report data
            report = assessment_data.get("report", {})
            logger.info(f"Extracted report data: {bool(report)}")
            
            # Get user details for hrId and department
            user_id = assessment_data["userId"]
            logger.info(f"Looking up user with ID: {user_id}")
            user = await prisma.user.find_unique(
                where={"id": user_id}
            )
            
            if not user:
                logger.error(f"User not found with ID: {user_id}")
                raise ValueError("User not found")
            
            logger.info(f"User found: {user.id}, hrId: {user.hrId}, department: {user.department}")
            
            # Process JSON fields with proper validation
            genius_factor_profile = AssessmentController.sanitize_and_validate_json(
                report.get("genius_factor_profile", {})
            )
            current_role_alignment = AssessmentController.sanitize_and_validate_json(
                report.get("current_role_alignment_analysis", {})
            )
            internal_career_opportunities = AssessmentController.sanitize_and_validate_json(
                report.get("internal_career_opportunities", {})
            )
            retention_strategies = AssessmentController.sanitize_and_validate_json(
                report.get("retention_and_mobility_strategies", {})
            )
            development_action_plan = AssessmentController.sanitize_and_validate_json(
                report.get("development_action_plan", {})
            )
            personalized_resources = AssessmentController.sanitize_and_validate_json(
                report.get("personalized_resources", {})
            )
            data_sources_methodology = AssessmentController.sanitize_and_validate_json(
                report.get("data_sources_and_methodology", {})
            )
            risk_analysis = AssessmentController.sanitize_and_validate_json(
                assessment_data.get("risk_analysis", {})
            )
            
            # Log the processed data for debugging
            logger.info(f"Genius factor profile type: {type(genius_factor_profile)}")
            logger.info(f"Genius factor profile content: {genius_factor_profile}")
            
            # Prepare data for report creation
            report_data = {
                "userId": user_id,
                "hrId": user.hrId,
                "departement": user.department[-1] if user.department else "General",
                "executiveSummary": report.get("executive_summary", ""),
                "geniusFactorScore": int(report.get("genius_factor_score", 0)),
                "risk_analysis": risk_analysis,
            }
            
            # Only add JSON fields if they're not None
            if genius_factor_profile is not None:
                report_data["geniusFactorProfileJson"] = genius_factor_profile
            if current_role_alignment is not None:
                report_data["currentRoleAlignmentAnalysisJson"] = current_role_alignment
            if internal_career_opportunities is not None:
                report_data["internalCareerOpportunitiesJson"] = internal_career_opportunities
            if retention_strategies is not None:
                report_data["retentionAndMobilityStrategiesJson"] = retention_strategies
            if development_action_plan is not None:
                report_data["developmentActionPlanJson"] = development_action_plan
            if personalized_resources is not None:
                report_data["personalizedResourcesJson"] = personalized_resources
            if data_sources_methodology is not None:
                report_data["dataSourcesAndMethodologyJson"] = data_sources_methodology

            logger.info("Prepared report data for creation")
            logger.info(f"Report data keys: {list(report_data.keys())}")
            
            # Create the report
            logger.info("Creating individual employee report in database...")
            saved_report = await prisma.individualemployeereport.create(
                data=report_data
            )
            
            logger.info(f"Successfully saved report to database with ID: {saved_report.id}")
            await prisma.disconnect()
            return {"status": "success", "report_id": saved_report.id}
            
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
            logger.error(f"Full exception details: {type(e).__name__}: {e}")
            await prisma.disconnect()
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