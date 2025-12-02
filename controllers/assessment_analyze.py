from prisma import Prisma
from utils.logger import logger
import json, asyncpg
from schemas.assessment import AssessmentData, AssessmentPart
from services.ai_service import AIService
# from services.database_notification_service import DatabaseNotificationService
from services.db_service import DBService
from utils.analyze_assessment import analyze_assessment_data
from utils.logger import logger
from utils.analysis_utils import analyze_full_from_parts, categorize_part_name
from utils.logger import logger
from services.notification_service import NotificationService
from typing import Dict, Any
import httpx
from fastapi import HTTPException

# Singleton AIService instance (assumed to be defined elsewhere)
ai_service = AIService()

# Singleton DatabaseNotificationService instance
db_notification_service = DBService()



class AssessmentController:
    
    @staticmethod
    async def save_to_db(input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Minimal version: saves hardcoded data using raw PostgreSQL (asyncpg).
        """

        

        conn = None
        try:
         
            # Validate JSON data before saving
            json_test = json.dumps(input_data)

       
        
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

            # Get table structure
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
          
            return {"status": "error", "message": f"Database error: {str(db_error)}"}

        except (TypeError, ValueError) as json_error:
         
            return {"status": "error", "message": f"JSON encoding error: {str(json_error)}"}

        except Exception as e:
        
            import traceback
     
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
        genius detection and real-time notifications
        """
        try:
            # Extract is_paid from input_data
            is_paid = input_data.is_paid if hasattr(input_data, 'is_paid') else False
            logger.info(f"User {input_data.userId} is_paid status: {is_paid}")
            
            # === fetch department ===
            db_params = {
                "user": "postgres", "password": "root",
                "database": "genius_factor", "host": "localhost", "port": 5432
            }
            conn = await asyncpg.connect(**db_params)
            departement = await conn.fetchval(
                'SELECT "department"[array_length("department", 1)] FROM "User" WHERE id = $1',
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
            logger.info("Basic analysis completed")
            logger.debug(json.dumps(basic_results, indent=2))

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

                logger.debug(f"Part '{part.part}' → bucket '{section_key}', letters={letters}")

            # === 3. Deep analysis aggregated across all sections ===
            # Pass basic_results (list of dicts) instead of user_answers (dict)
            try:
                deep_results = analyze_full_from_parts(basic_results)
                deep_results["departement"] = input_data.departement
                logger.info("Deep analysis completed")
                logger.debug(json.dumps(deep_results, indent=2))
            except Exception as e:
                logger.error(f"Deep analysis failed: {str(e)}")
                # Continue without deep results
                deep_results = {"error": f"Deep analysis failed: {str(e)}"}

            # === 4. RAG step with improved inputs ===
            try:
                rag_results = await ai_service.analyze_majority_answers(basic_results, deep_results)
                print(f'RAG results: {rag_results}')
            except Exception as e:
                logger.exception("RAG analysis failed")
                rag_results = f"RAG analysis failed: {str(e)}"

            # === 5. Generate professional career recommendation report ===
            try:
                recommendations = await ai_service.generate_career_recommendation(
                    rag_results, 
                    input_dict.get('allAnswers', []),
                    is_paid=is_paid  # Pass the is_paid parameter
                )
            except Exception as e:
                logger.error(f"Failed to generate recommendations: {str(e)}")
                await NotificationService.send_user_notification(
                    input_dict['userId'],
                    input_dict['hrId'],
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
                    input_dict['userId'],
                    input_dict['hrId'],
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
                "hrId": input_dict['hrId'],
                "departement": input_dict['departement'],
                "userId": input_dict['userId'],
                "report": recommendations.get("report"),
                "user_type": recommendations.get("user_type", "free"),  # Add user type to response
                "risk_analysis": recommendations.get("risk_analysis"),
                "metadata": recommendations.get("metadata")
            }

            # === 6. Save data to database ===
            save_response = await AssessmentController.save_to_db(final_result)
            
            if save_response.get("status") == "error":
                logger.warning(f"Database save failed but proceeding: {save_response.get('message')}")
                # Continue even if database save fails, but log the warning

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

            # Save notification to database using DatabaseNotificationService
            try:
                await db_notification_service.save_notification(notification_data)
                
            except Exception as e:
                logger.error(f"Error saving notification: {str(e)}")
                return
                # Continue even if notification save fails

        except Exception as e:
            logger.error(f"Error in analyze_assessment: {str(e)}")
            
            # Use input_dict for consistent access to user data
            await NotificationService.send_user_notification(
                input_dict['userId'],
                input_dict['hrId'],
                {
                    'message': 'Assessment analysis failed',
                    'progress': 100,
                    'status': 'error',
                    'error': str(e)
                }
            )
            
            raise HTTPException(status_code=500, detail=str(e))