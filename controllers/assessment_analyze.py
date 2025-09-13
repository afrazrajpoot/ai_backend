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

    # Expanded and more readable number mapping (use underscores for multi-word)
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
        "11": "eleven",
        "12": "twelve",
        "18": "eighteen",
        "24": "twenty_four",
        "30": "thirty",
        "36": "thirty_six",
        "48": "forty_eight",
        "60": "sixty",
        "72": "seventy_two",
        "90": "ninety",
    }

    @staticmethod
    def sanitize_json_keys(obj):
        """
        Recursively sanitize JSON keys to be valid for GraphQL/Prisma:
         - convert leading numbers to words using number_words
         - replace invalid characters (spaces, brackets, punctuation) with underscores
         - collapse multiple underscores
         - trim leading/trailing underscores
         - ensure key starts with a letter by prefixing 'field_' if necessary
        """
        if isinstance(obj, dict):
            sanitized = {}
            for key, value in obj.items():
                str_key = str(key)

                # Convert leading numeric sequence to words (e.g. '6_month' -> 'six_month')
                match = re.match(r'^(\d+)', str_key)
                if match:
                    leading_num = match.group(1)
                    word = AssessmentController.number_words.get(leading_num, f"num{leading_num}")
                    str_key = str_key.replace(leading_num, word, 1)
                    logger.debug(f"Converted numeric key: '{key}' -> '{str_key}' (replaced '{leading_num}' with '{word}')")

                # Replace all non-alphanumeric/underscore characters with underscore
                sanitized_key = re.sub(r'[^a-zA-Z0-9_]', '_', str_key)
                # Collapse multiple underscores
                sanitized_key = re.sub(r'_+', '_', sanitized_key)
                # Trim leading/trailing underscores
                sanitized_key = sanitized_key.strip('_')

                # Ensure key starts with a letter (GraphQL requirement)
                if sanitized_key and not re.match(r'^[a-zA-Z]', sanitized_key):
                    sanitized_key = 'field_' + sanitized_key

                # Ensure key is not empty
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
    def normalize_timeline_keys(timeline_obj: dict) -> dict:
        """
        Specifically normalize keys in transition_timeline objects.
        Accepts keys like '6_month', '6-month', '6 month', '6months', '6_months'
        and returns keys like 'six_month'.

        Rules:
        - Extract leading digits (if any), convert to words using number_words.
        - Keep rest of the key (e.g. 'month', 'months', 'year', etc.), normalize to safe chars.
        - Output: <number_word>_<rest> or just <number_word> if no rest.
        """
        if not isinstance(timeline_obj, dict):
            return timeline_obj

        normalized = {}
        for raw_k, v in timeline_obj.items():
            k = str(raw_k).strip()

            # Try to match leading digits plus optional separators and the rest
            m = re.match(r'^(\d+)[\s_\-]*([A-Za-z0-9_]*)', k)
            if m:
                leading = m.group(1)
                rest = m.group(2) or ''
                word = AssessmentController.number_words.get(leading, f"num{leading}")
                if rest:
                    # normalize rest portion (remove invalid chars)
                    rest_norm = re.sub(r'[^a-zA-Z0-9_]', '_', rest)
                    rest_norm = re.sub(r'_+', '_', rest_norm).strip('_')
                    new_key = f"{word}_{rest_norm}" if rest_norm else word
                else:
                    new_key = word
            else:
                # no leading digits, sanitize generically
                new_key = re.sub(r'[^a-zA-Z0-9_]', '_', k)
                new_key = re.sub(r'_+', '_', new_key).strip('_')
                if new_key and not re.match(r'^[a-zA-Z]', new_key):
                    new_key = 'field_' + new_key
                if not new_key:
                    new_key = f'field_{hash(k) % 1000}'

            # final safety: ensure it starts with a letter
            if new_key and not re.match(r'^[a-zA-Z]', new_key):
                new_key = 'field_' + new_key

            # avoid accidental collisions: if key exists already, append suffix
            if new_key in normalized:
                suffix = 1
                base = new_key
                while f"{base}_{suffix}" in normalized:
                    suffix += 1
                new_key = f"{base}_{suffix}"

            normalized[new_key] = v

            logger.debug(f"Normalized timeline key: '{raw_k}' -> '{new_key}'")

        return normalized

    @staticmethod
    async def save_to_database(assessment_data: dict):
        """
        Save assessment data to the database using Prisma.
        Sanitizes entire report first, then specifically normalizes transition_timeline.
        """
        prisma = Prisma()
        await prisma.connect()

        try:
            logger.info("Starting database save process for assessment data")
            logger.info("Database connection established")

            # Extract the raw report and sanitize the entire report at once
            report_raw = assessment_data.get("report", {}) or {}
            logger.info(f"Extracted report data: {bool(report_raw)}")

            # 1) Sanitize whole report keys
            sanitized_report = AssessmentController.sanitize_json_keys(report_raw)

            # 2) If internal_career_opportunities.transition_timeline exists, normalize its keys
            if isinstance(sanitized_report.get("internal_career_opportunities"), dict):
                internal_career = sanitized_report["internal_career_opportunities"]
                if isinstance(internal_career.get("transition_timeline"), dict):
                    normalized_timeline = AssessmentController.normalize_timeline_keys(internal_career["transition_timeline"])
                    sanitized_report["internal_career_opportunities"]["transition_timeline"] = normalized_timeline
                    logger.info(f"Normalized transition_timeline keys: {list(normalized_timeline.keys())}")
                else:
                    logger.debug("No transition_timeline dict found to normalize.")
            else:
                logger.debug("No internal_career_opportunities present or not a dict.")

            # Get user details for hrId and department
            user_id = assessment_data["userId"]
            logger.info(f"Looking up user with ID: {user_id}")
            user = await prisma.user.find_unique(where={"id": user_id})

            if not user:
                logger.error(f"User not found with ID: {user_id}")
                raise ValueError("User not found")

            logger.info(f"User found: {user.id}, hrId: {user.hrId}, department: {user.department}")

            # Extract sanitized subfields (these are already sanitized)
            genius_factor_json = sanitized_report.get("genius_factor_profile", {})
            current_role_json = sanitized_report.get("current_role_alignment_analysis", {})
            internal_career_json = sanitized_report.get("internal_career_opportunities", {})
            retention_strategies_json = sanitized_report.get("retention_and_mobility_strategies", {})
            development_plan_json = sanitized_report.get("development_action_plan", {})
            personalized_resources_json = sanitized_report.get("personalized_resources", {})
            data_sources_json = sanitized_report.get("data_sources_and_methodology", {})

            # Sanitize risk_analysis separately (it may be outside report)
            risk_analysis_json = AssessmentController.sanitize_json_keys(assessment_data.get("risk_analysis", {}) or {})

            # Debug output right before insert
            logger.info(f"Prepared sanitized keys for insert. Internal career transition keys: "
                        f"{list(internal_career_json.get('transition_timeline', {}).keys()) if isinstance(internal_career_json, dict) else 'n/a'}")

            # Prepare data for report creation (use sanitized_report for string fields too)
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

                raise HTTPException(status_code=500, detail="Failed to generate career recommendations")

            final_result = {
                "status": "success",
                "userId": input_data.userId,
                "report": recommendations.get("report"),
                "risk_analysis": recommendations.get("risk_analysis"),
                "metadata": recommendations.get("metadata"),
            }
            print(f"Assessment analysis and recommendation generation completed: {final_result}")

            db_response = await AssessmentController.save_to_database(final_result)

            if db_response.get("status") == "error":
                logger.warning(f"Database save failed but proceeding: {db_response.get('message')}")

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

            logger.info("Assessment analysis, report generation, and Next.js integration completed successfully")

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
