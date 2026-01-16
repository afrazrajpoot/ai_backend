from prisma import Prisma
from utils.logger import logger
import json
import asyncpg
from schemas.assessment import AssessmentData, AssessmentPart
from services.ai_service import AIService
# from services.database_notification_service import DatabaseNotificationService
from services.db_service import DBService
from utils.analyze_assessment import analyze_assessment_data
from utils.analysis_utils import analyze_full_from_parts, categorize_part_name
from services.notification_service import NotificationService
from typing import Dict, Any
import httpx
from fastapi import HTTPException
import asyncio


# Singleton AIService instance (assumed to be defined elsewhere)
ai_service = AIService()

# Singleton DatabaseNotificationService instance
db_notification_service = DBService()


class AssessmentController:
    
    @staticmethod
    def _validate_report_for_db(final_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that the report structure matches the database schema.
        Returns validated data with defaults for missing fields.
        """
        try:
            # Check that required top-level fields exist
            required_fields = ["userId", "hrId", "departement", "report", "risk_analysis"]
            for field in required_fields:
                if field not in final_result:
                    logger.warning(f"Missing required field: {field}")
                    if field == "departement":
                        final_result[field] = "Unknown"
                    elif field in ["report", "risk_analysis"]:
                        final_result[field] = {}
            
            report = final_result.get("report", {})
            
            # Validate required report sections
            required_sections = [
                "executive_summary",
                "genius_factor_profile",
                "current_role_alignment_analysis",
                "internal_career_opportunities",
                "retention_and_mobility_strategies",
                "development_action_plan",
                "personalized_resources",
                "data_sources_and_methodology",
                "genius_factor_score"
            ]
            
            for section in required_sections:
                if section not in report:
                    logger.warning(f"Missing report section: {section}")
                    if section == "executive_summary":
                        report[section] = "Assessment analysis completed."
                    elif section == "genius_factor_score":
                        report[section] = 75
                    else:
                        report[section] = {}
            
            # Validate genius_factor_score is an integer
            try:
                report["genius_factor_score"] = int(report.get("genius_factor_score", 75))
            except (TypeError, ValueError):
                logger.warning("Invalid genius_factor_score, using default 75")
                report["genius_factor_score"] = 75
            
            # Validate executive_summary is a string
            if not isinstance(report.get("executive_summary"), str):
                logger.warning("executive_summary is not a string, converting")
                report["executive_summary"] = str(report.get("executive_summary", "Analysis completed."))
            
            # Ensure risk_analysis exists
            if "risk_analysis" not in final_result or not final_result["risk_analysis"]:
                final_result["risk_analysis"] = {
                    "analysis_summary": "Risk analysis completed",
                    "scores": {
                        "genius_factor_score": report.get("genius_factor_score", 75),
                        "retention_risk_score": 50,
                        "mobility_opportunity_score": 65
                    }
                }
            
            logger.info(f"✅ Report validation passed for user {final_result.get('userId')}")
            return final_result
            
        except Exception as e:
            logger.error(f"Error validating report: {str(e)}")
            raise
    
    @staticmethod
    async def save_to_db(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimal version: saves hardcoded data using raw PostgreSQL (asyncpg).
        """

        conn = None
        try:
            # Connection parameters
            db_params = {
                "user": "postgres",
                "password": "root",
                "database": "genius_factor",
                "host": "localhost",
                "port": 5432
            }

            # Connect to database
            conn = await asyncpg.connect(**db_params)

            # Test connection with simple query
            test_result = await conn.fetchval("SELECT 1")

            table_check = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'IndividualEmployeeReport'
                );
            """)

            if not table_check:
                return {"status": "error", "message": "Table 'IndividualEmployeeReport' does not exist"}

            # Get table structure (optional, for debugging)
            columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'IndividualEmployeeReport'
                ORDER BY ordinal_position;
            """)

            # Prepare INSERT query with better formatting
            query = """
                INSERT INTO "IndividualEmployeeReport" (
                    "userId",
                    "hrId",
                    "departement",
                    "executiveSummary",
                    "geniusFactorProfileJson",
                    "currentRoleAlignmentAnalysisJson",
                    "internalCareerOpportunitiesJson",
                    "retentionAndMobilityStrategiesJson",
                    "developmentActionPlanJson",
                    "personalizedResourcesJson",
                    "dataSourcesAndMethodologyJson",
                    "risk_analysis",
                    "geniusFactorScore",
                    "createdAt",
                    "updatedAt"
                )
                VALUES (
                    $1, $2, $3, $4, 
                    $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb, 
                    $9::jsonb, $10::jsonb, $11::jsonb, $12::jsonb, 
                    $13, NOW(), NOW()
                )
                RETURNING id, "createdAt", "updatedAt"
            """

            # Log data being saved (for debugging)
            logger.info(f"💾 Saving report to database for user: {input_data['userId']}")
            logger.info(f"   - Department: {input_data['departement']}")
            logger.info(f"   - Genius Factor Score: {input_data['report']['genius_factor_score']}")
            logger.info(f"   - Executive Summary (first 100 chars): {input_data['report']['executive_summary'][:100]}...")
            
            # Execute the query
            result = await conn.fetchrow(
                query,
                input_data["userId"],
                input_data["hrId"],
                input_data["departement"],
                input_data["report"]["executive_summary"],
                json.dumps(input_data["report"]["genius_factor_profile"]),
                json.dumps(input_data["report"]["current_role_alignment_analysis"]),
                json.dumps(input_data["report"]["internal_career_opportunities"]),
                json.dumps(input_data["report"]["retention_and_mobility_strategies"]),
                json.dumps(input_data["report"]["development_action_plan"]),
                json.dumps(input_data["report"]["personalized_resources"]),
                json.dumps(input_data["report"]["data_sources_and_methodology"]),
                json.dumps(input_data["risk_analysis"]),
                input_data["report"]["genius_factor_score"]
            )

            if result:
                # Verify the record was saved by reading it back
                verify_record = await conn.fetchrow(
                    'SELECT id, "userId", "createdAt" FROM "IndividualEmployeeReport" WHERE id = $1',
                    result['id']
                )

                return {
                    "status": "success", 
                    "saved_record_id": result['id'],
                    "created_at": result['createdAt'].isoformat() if result['createdAt'] else None
                }
            else:
                return {"status": "error", "message": "No result returned from INSERT query"}

        except asyncpg.PostgresError as db_error:
            logger.error(f"Database error: {str(db_error)}")
            return {"status": "error", "message": f"Database error: {str(db_error)}"}

        except (TypeError, ValueError) as json_error:
            logger.error(f"JSON encoding error: {str(json_error)}")
            return {"status": "error", "message": f"JSON encoding error: {str(json_error)}"}

        except Exception as e:
            logger.error(f"Unexpected error in save_to_db: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}

        finally:
            if conn:
                try:
                    await conn.close()
                except Exception as close_error:
                    logger.error(f"❌ Error closing database connection: {str(close_error)}")
                
    @staticmethod
    async def analyze_assessment(input_data: AssessmentData) -> Dict[str, Any]:
        """
        Endpoint for assessment analysis with deep section-by-section
        genius detection and real-time notifications with retry logic.
        """
        max_retries = 3
        last_error = None
        
        for attempt in range(max_retries):
            try:
                # Extract is_paid from input_data
                is_paid = input_data.is_paid if hasattr(input_data, 'is_paid') else False
                
                # === fetch department ===
                db_params = {
                    "user": "postgres", "password": "root",
                    "database": "genius_factor", "host": "localhost", "port": 5432
                }
                conn = await asyncpg.connect(**db_params)
                departement = await conn.fetchval(
                    'SELECT "department"[array_length("department", 1)] FROM "users" WHERE id = $1',
                    input_data.userId
                )
                await conn.close()
                if not departement:
                    departement = "Unknown"

                input_dict = input_data.dict()
                input_dict["departement"] = departement

                # === validate notification data ===
                notification_data = {
                    'employeeId': input_dict['userId'],
                    'hrId': input_dict['hrId'],
                    'employeeName': input_dict['employeeName'],
                    'employeeEmail': input_dict['employeeEmail'],
                    'message': 'Assessment analysis completed successfully!',
                    'status': 'unread'
                }
                for k, v in notification_data.items():
                    if not isinstance(v, str) or not v.strip():
                        logger.error(f"Invalid notification data: {k}")
                        await NotificationService.send_user_notification(
                            input_dict['userId'], input_dict['hrId'],
                            {'message': 'Invalid notification data', 'progress': 0, 'status': 'error', 'error': f"Field {k} is invalid"}
                        )
                        raise HTTPException(status_code=400, detail=f"Invalid notification data: {k}")

                # === 1. Convert and run basic analysis ===
                assessment_parts = [AssessmentPart(**part) for part in input_dict['data']]
                basic_results = analyze_assessment_data(assessment_parts)
                
                # === 2. Build raw answers for deep section analysis ===
                user_answers = {
                    "SelfAwareness": [],
                    "Talent": [],
                    "Passion": [],
                    "Mapping": [],
                    "Goals": []
                }

                for part in assessment_parts:
                    letters = []
                    for letter, count in part.optionCounts.dict(exclude_none=True).items():
                        letters.extend([letter] * count)

                    # ✅ categorize using the robust helper
                    section_key = categorize_part_name(part.part)
                    user_answers[section_key].extend(letters)

                # === 3. Deep analysis aggregated across all sections ===
                try:
                    deep_results = analyze_full_from_parts(basic_results)
                    deep_results["departement"] = input_data.departement
                except Exception as e:
                    logger.error(f"Deep analysis failed: {str(e)}")
                    deep_results = {"error": f"Deep analysis failed: {str(e)}"}

                # === 4. RAG step with improved inputs ===
                try:
                    rag_results = await ai_service.analyze_majority_answers(basic_results, deep_results)
                except Exception as e:
                    logger.exception("RAG analysis failed")
                    rag_results = f"RAG analysis failed: {str(e)}"

                # === 5. Generate professional career recommendation report ===
                try:
                    recommendations = await ai_service.generate_career_recommendation(
                        rag_results, 
                        input_dict.get('allAnswers', []),
                        user_id=input_dict.get('userId'),
                        payload_data=input_dict
                    )
                except Exception as e:
                    logger.error(f"Failed to generate recommendations: {str(e)}")
                    # We don't raise here yet, we'll check recommendations status below
                    recommendations = {"status": "error", "message": str(e)}
                
                if recommendations.get("status") != "success":
                    error_msg = f"Failed to generate recommendations: {recommendations.get('message', 'Unknown error')}"
                    logger.error(error_msg)
                    raise Exception(error_msg) # Raise to trigger retry

                # Prepare final result
                final_result = {
                    "success": True,
                    "data": basic_results,
                    "hrId": input_dict['hrId'],
                    "departement": input_dict['departement'],
                    "userId": input_dict['userId'],
                    "report": recommendations.get("report"),
                    "user_type": recommendations.get("user_type", "free"),
                    "risk_analysis": recommendations.get("risk_analysis"),
                    "metadata": recommendations.get("metadata")
                }

                # === 6. Validate and save data to database ===
                logger.info(f"Validating report structure for database schema compliance...")
                validated_result = AssessmentController._validate_report_for_db(final_result)
                
                save_response = await AssessmentController.save_to_db(validated_result)
                
                if save_response.get("status") == "error":
                    logger.warning(f"Database save failed but proceeding: {save_response.get('message')}")

                # Send success notification via Socket.IO
                await NotificationService.send_user_notification(
                    input_dict['userId'],
                    input_dict['hrId'],
                    {
                        'message': 'Assessment analysis completed successfully!',
                        'employeeName': input_dict['employeeName'],
                        'employeeEmail': input_dict['employeeEmail'],
                        'progress': 100,
                        'status': 'unread',
                        'user_type': 'paid' if is_paid else 'free'
                    }
                )

                # Save notification to database
                try:
                    await db_notification_service.save_notification(notification_data)
                except Exception as e:
                    logger.error(f"Error saving notification: {str(e)}")

                return final_result

            except HTTPException as e:
                # If it's a client error (4xx), don't retry
                if e.status_code < 500:
                    raise e
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed with HTTPException {e.status_code}: {e.detail}. Retrying...")
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {attempt + 1} failed with error: {str(e)}. Retrying...")
            
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.info(f"Waiting {wait_time} seconds before retry...")
                await asyncio.sleep(wait_time)

        # If we reach here, all retries failed
        logger.error(f"All {max_retries} attempts failed for analyze_assessment. Last error: {str(last_error)}")
        
        # Get userId and hrId safely
        user_id = getattr(input_data, 'userId', None)
        hr_id = getattr(input_data, 'hrId', None)
        
        if user_id and hr_id:
            await NotificationService.send_user_notification(
                user_id,
                hr_id,
                {
                    'message': 'Assessment analysis failed',
                    'progress': 100,
                    'status': 'error',
                    'error': "We encountered a persistent issue while analyzing your assessment. Please try again later."
                }
            )
        
        raise HTTPException(
            status_code=500, 
            detail="We're sorry, but we encountered a persistent issue while analyzing your assessment. Please try again in a few moments. If the problem persists, please contact support."
        )