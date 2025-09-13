from fastapi import HTTPException
from schemas.assessment import AssessmentData
from services.ai_service import AIService
from utils.analyze_assessment import analyze_assessment_data
from utils.logger import logger
from services.notification_service import NotificationService
from typing import Dict, Any
from prisma import Prisma
import json


class AssessmentController:
    
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
            
            # DEBUGGING: Log the raw data types and content
            genius_factor_raw = report.get("genius_factor_profile", {})
            logger.info(f"Raw genius_factor_profile type: {type(genius_factor_raw)}")
            logger.info(f"Raw genius_factor_profile content: {genius_factor_raw}")
            
            # Direct JSON processing - NO sanitize_keys function
            # Just ensure the data is JSON serializable
            try:
                genius_factor_json = json.loads(json.dumps(genius_factor_raw)) if genius_factor_raw else {}
                logger.info(f"Processed genius_factor_profile: {type(genius_factor_json)}")
                logger.info(f"JSON test passed for genius_factor_profile")
            except (TypeError, ValueError) as e:
                logger.error(f"JSON processing failed for genius_factor_profile: {e}")
                genius_factor_json = {}
            
            try:
                current_role_json = json.loads(json.dumps(report.get("current_role_alignment_analysis", {})))
            except (TypeError, ValueError) as e:
                logger.error(f"JSON processing failed for current_role_alignment_analysis: {e}")
                current_role_json = {}
                
            try:
                internal_career_json = json.loads(json.dumps(report.get("internal_career_opportunities", {})))
            except (TypeError, ValueError) as e:
                logger.error(f"JSON processing failed for internal_career_opportunities: {e}")
                internal_career_json = {}
                
            try:
                retention_strategies_json = json.loads(json.dumps(report.get("retention_and_mobility_strategies", {})))
            except (TypeError, ValueError) as e:
                logger.error(f"JSON processing failed for retention_and_mobility_strategies: {e}")
                retention_strategies_json = {}
                
            try:
                development_plan_json = json.loads(json.dumps(report.get("development_action_plan", {})))
            except (TypeError, ValueError) as e:
                logger.error(f"JSON processing failed for development_action_plan: {e}")
                development_plan_json = {}
                
            try:
                personalized_resources_json = json.loads(json.dumps(report.get("personalized_resources", {})))
            except (TypeError, ValueError) as e:
                logger.error(f"JSON processing failed for personalized_resources: {e}")
                personalized_resources_json = {}
                
            try:
                data_sources_json = json.loads(json.dumps(report.get("data_sources_and_methodology", {})))
            except (TypeError, ValueError) as e:
                logger.error(f"JSON processing failed for data_sources_and_methodology: {e}")
                data_sources_json = {}
                
            try:
                risk_analysis_json = json.loads(json.dumps(assessment_data.get("risk_analysis", {})))
            except (TypeError, ValueError) as e:
                logger.error(f"JSON processing failed for risk_analysis: {e}")
                risk_analysis_json = {}
            
            # Prepare data for report creation
            report_data = {
                "userId": user_id,
                "hrId": user.hrId,
                "departement": user.department[-1] if user.department else "General",
                "executiveSummary": str(report.get("executive_summary", "")),
                "geniusFactorScore": int(report.get("genius_factor_score", 0)),
                "geniusFactorProfileJson": genius_factor_json,
                "currentRoleAlignmentAnalysisJson": current_role_json,
                "internalCareerOpportunitiesJson": internal_career_json,
                "retentionAndMobilityStrategiesJson": retention_strategies_json,
                "developmentActionPlanJson": development_plan_json,
                "personalizedResourcesJson": personalized_resources_json,
                "dataSourcesAndMethodologyJson": data_sources_json,
                "risk_analysis": risk_analysis_json,
            }

            logger.info("Prepared report data for creation")
            logger.info(f"Report data keys: {list(report_data.keys())}")
            
            # DEBUGGING: Log each JSON field type
            for key, value in report_data.items():
                if "Json" in key or key == "risk_analysis":
                    logger.info(f"{key} type: {type(value)}, empty: {not bool(value)}")
            
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