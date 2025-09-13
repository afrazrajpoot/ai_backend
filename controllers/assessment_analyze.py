from fastapi import HTTPException
from schemas.assessment import AssessmentData
from services.ai_service import AIService
from utils.analyze_assessment import analyze_assessment_data
from utils.logger import logger
from services.notification_service import NotificationService
from typing import Dict, Any
from prisma import Prisma
import json
import re


class AssessmentController:

    number_words = {
        "0": "zero",
        "1": "one",
        "2": "two",
        "3": "three",
        "4": "four",
        "5": "five",
        "6": "six",
        "7": "seven",
        "8": "eight",
        "9": "nine",
        "10": "ten",
        "30": "thirty",
        "60": "sixty",
        "90": "ninety",
    }

    
    @staticmethod
    def sanitize_json_keys(obj):
        """
        Recursively sanitize JSON keys to be valid for GraphQL/Prisma
        - Convert numeric keys to word equivalents (6_months -> six_months)
        - Replace invalid characters with underscores
        """

        
        
        if isinstance(obj, dict):
            sanitized = {}
            for key, value in obj.items():
                str_key = str(key)

                # Handle keys starting with numbers
                if re.match(r'^\d', str_key):
                    # Match one or more leading digits
                    match = re.match(r'^\d+', str_key)
                    if match:
                        leading_num = match.group()
                        # Convert number to word if possible
                        word = AssessmentController.number_words.get(leading_num, f"num{leading_num}")
                        str_key = str_key.replace(leading_num, word, 1)
                        logger.debug(f"Converted numeric key: '{key}' -> '{str_key}' (replaced '{leading_num}' with '{word}')")

                # Replace invalid characters with underscore
                sanitized_key = re.sub(r'[^a-zA-Z0-9_]', '_', str_key)

                # Ensure key starts with a letter (GraphQL requirement)
                if sanitized_key and not re.match(r'^[a-zA-Z]', sanitized_key):
                    if re.match(r'^\d', sanitized_key):
                        # If it still starts with a number, prefix with 'key_'
                        sanitized_key = 'key_' + sanitized_key
                    elif sanitized_key.startswith('_'):
                        # If it starts with underscore, prefix with 'field'
                        sanitized_key = 'field' + sanitized_key

                # Log transformation for debugging
                if sanitized_key != key:
                    logger.debug(f"Sanitized key: {key} -> {sanitized_key}")

                # Recurse
                sanitized[sanitized_key] = AssessmentController.sanitize_json_keys(value)
            return sanitized

        elif isinstance(obj, list):
            return [AssessmentController.sanitize_json_keys(item) for item in obj]
        else:
            return obj

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
            
            # Process JSON fields with proper Prisma formatting
            def process_json_field(data):
                if not data:
                    return {}  # Return empty dict for Prisma
                try:
                    logger.debug(f"Processing JSON field with keys: {list(data.keys())[:5]}...")  # Log first 5 keys
                    # First sanitize the keys
                    sanitized = AssessmentController.sanitize_json_keys(data)
                    logger.debug(f"Sanitized JSON field with keys: {list(sanitized.keys())[:5]}...")  # Log first 5 sanitized keys
                    # Return as dict for Prisma (not JSON string)
                    return sanitized
                except (TypeError, ValueError) as e:
                    logger.error(f"JSON processing failed: {e}")
                    return {}
            
            genius_factor_json = process_json_field(report.get("genius_factor_profile", {}))
            current_role_json = process_json_field(report.get("current_role_alignment_analysis", {}))
            internal_career_json = process_json_field(report.get("internal_career_opportunities", {}))
            retention_strategies_json = process_json_field(report.get("retention_and_mobility_strategies", {}))
            development_plan_json = process_json_field(report.get("development_action_plan", {}))
            personalized_resources_json = process_json_field(report.get("personalized_resources", {}))
            data_sources_json = process_json_field(report.get("data_sources_and_methodology", {}))
            risk_analysis_json = process_json_field(assessment_data.get("risk_analysis", {}))
            
            # Log the problematic field for debugging
            logger.info(f"Internal career opportunities keys: {list(internal_career_json.get('transition_timeline', {}).keys()) if 'transition_timeline' in internal_career_json else 'No transition_timeline'}")
            
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