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
        "12": "twelve",
        "18": "eighteen",
        "24": "twentyfour",
        "30": "thirty",
        "36": "thirtysix",
        "48": "fortyeight",
        "60": "sixty",
        "72": "seventytwo",
        "90": "ninety",
    }

    @staticmethod
    def sanitize_json_keys(obj):
        """
        Recursively sanitize JSON keys to be valid for GraphQL/Prisma
        - Convert numeric keys to word equivalents (6_months -> six_months)
        - Replace invalid characters with underscores
        - Handle square brackets and special characters
        """
        if isinstance(obj, dict):
            sanitized = {}
            for key, value in obj.items():
                str_key = str(key)

                # Handle keys starting with numbers
                if re.match(r'^\d', str_key):
                    match = re.match(r'^\d+', str_key)
                    if match:
                        leading_num = match.group()
                        word = AssessmentController.number_words.get(
                            leading_num, f"num{leading_num}"
                        )
                        str_key = str_key.replace(leading_num, word, 1)
                        logger.debug(
                            f"Converted numeric key: '{key}' -> '{str_key}' (replaced '{leading_num}' with '{word}')"
                        )

                # Replace ALL invalid characters with underscore
                sanitized_key = re.sub(r'[^a-zA-Z0-9_]', '_', str_key)
                sanitized_key = re.sub(r'_+', '_', sanitized_key)  # collapse __
                sanitized_key = sanitized_key.strip('_')

                # Ensure key starts with a letter
                if sanitized_key and not re.match(r'^[a-zA-Z]', sanitized_key):
                    sanitized_key = 'field_' + sanitized_key

                # Ensure not empty
                if not sanitized_key:
                    sanitized_key = f'field_{hash(str(key)) % 1000}'

                if sanitized_key != key:
                    logger.debug(f"Sanitized key: '{key}' -> '{sanitized_key}'")

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

            # Extract and sanitize the entire report
            report = assessment_data.get("report", {})
            logger.info(f"Extracted report data: {bool(report)}")

            sanitized_report = AssessmentController.sanitize_json_keys(report)

            # Get user details for hrId and department
            user_id = assessment_data["userId"]
            logger.info(f"Looking up user with ID: {user_id}")
            user = await prisma.user.find_unique(where={"id": user_id})

            if not user:
                logger.error(f"User not found with ID: {user_id}")
                raise ValueError("User not found")

            logger.info(
                f"User found: {user.id}, hrId: {user.hrId}, department: {user.department}"
            )

            # Extract sanitized JSON fields
            genius_factor_json = sanitized_report.get("genius_factor_profile", {})
            current_role_json = sanitized_report.get("current_role_alignment_analysis", {})
            internal_career_json = sanitized_report.get("internal_career_opportunities", {})
            retention_strategies_json = sanitized_report.get("retention_and_mobility_strategies", {})
            development_plan_json = sanitized_report.get("development_action_plan", {})
            personalized_resources_json = sanitized_report.get("personalized_resources", {})
            data_sources_json = sanitized_report.get("data_sources_and_methodology", {})
            risk_analysis_json = AssessmentController.sanitize_json_keys(
                assessment_data.get("risk_analysis", {})
            )

            # Debug: transition_timeline keys
            logger.info(
                f"Sanitized transition_timeline keys: "
                f"{list(internal_career_json.get('transition_timeline', {}).keys())}"
            )

            # Prepare data for report creation
            report_data = {
                "userId": user_id,
                "hrId": user.hrId,
                "departement": user.department[-1] if user.department else "General",
                "executiveSummary": str(sanitized_report.get("executive_summary", "")),
                "geniusFactorScore": int(sanitized_report.get("genius_factor_score", 0)),
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
            saved_report = await prisma.individualemployeereport.create(data=report_data)

            logger.info(
                f"Successfully saved report to database with ID: {saved_report.id}"
            )
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

            recommendations = await AIService.generate_career_recommendation(rag_results)
            print(f"Recommendations generated: {recommendations}")

            if recommendations.get("status") != "success":
                error_msg = f"Failed to generate recommendations: {recommendations.get('message', 'Unknown error')}"
                logger.error(error_msg)

                await NotificationService.send_user_notification(
                    input_data.userId,
                    {
                        "message": "Analysis failed",
                        "progress": 100,
                        "status": "error",
                        "error": error_msg,
                    },
                )

                raise HTTPException(
                    status_code=500, detail="Failed to generate career recommendations"
                )

            final_result = {
                "status": "success",
                "userId": input_data.userId,
                "report": recommendations.get("report"),
                "risk_analysis": recommendations.get("risk_analysis"),
                "metadata": recommendations.get("metadata"),
            }
            print(
                f"Assessment analysis and recommendation generation completed: {final_result}"
            )

            db_response = await AssessmentController.save_to_database(final_result)

            if db_response.get("status") == "error":
                logger.warning(
                    f"Database save failed but proceeding: {db_response.get('message')}"
                )

            await NotificationService.send_user_notification(
                input_data.userId,
                {
                    "message": "Assessment analysis completed successfully!",
                    "progress": 100,
                    "status": "completed",
                    "report_id": db_response.get("report_id"),
                    "db_response": db_response,
                },
            )

            logger.info(
                "Assessment analysis, report generation, and Next.js integration completed successfully"
            )

            return final_result

        except Exception as e:
            logger.error(f"Error in analyze_assessment: {str(e)}")

            await NotificationService.send_user_notification(
                input_data.userId,
                {
                    "message": "Assessment analysis failed",
                    "progress": 100,
                    "status": "error",
                    "error": str(e),
                },
            )

            raise HTTPException(status_code=500, detail=str(e))
