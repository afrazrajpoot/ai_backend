from prisma import Prisma
from utils.logger import logger
import json, asyncpg
from schemas.assessment import AssessmentData, AssessmentPart
from services.ai_service import AIService
# from services.database_notification_service import DatabaseNotificationService
from services.db_service import DBService
from utils.analyze_assessment import analyze_assessment_data
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
        logger.info("=== Starting analyze_assessment method ===")
        

        conn = None
        try:
            logger.info("=== Validating JSON data ===")
            # Validate JSON data before saving
            json_test = json.dumps(input_data)

            print(f"input_data complete data: {input_data}")
            logger.info(f"JSON validation passed. Data size: {len(json_test)} characters")

            logger.info("=== Attempting database connection ===")
            # Connection parameters
            db_params = {
                "user": "postgres",
                "password": "root",
                "database": "genius_factor",
                "host": "localhost",
                "port": 5432
            }
            logger.info(f"Connecting to database: {db_params['host']}:{db_params['port']}/{db_params['database']}")

            # Connect to database
            conn = await asyncpg.connect(**db_params)
            logger.info("✓ Database connection established successfully")

            # Test connection with simple query
            test_result = await conn.fetchval("SELECT 1")
            logger.info(f"✓ Connection test passed. Result: {test_result}")

            # Check if table exists
            logger.info("=== Checking table structure ===")
            table_check = await conn.fetchval("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'IndividualEmployeeReport'
                );
            """)
            logger.info(f"Table 'IndividualEmployeeReport' exists: {table_check}")

            if not table_check:
                logger.error("❌ Table 'IndividualEmployeeReport' does not exist!")
                return {"status": "error", "message": "Table 'IndividualEmployeeReport' does not exist"}

            # Get table structure
            columns = await conn.fetch("""
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_name = 'IndividualEmployeeReport'
                ORDER BY ordinal_position;
            """)
            
            logger.info("Table structure:")
            for col in columns:
                logger.info(f"  - {col['column_name']}: {col['data_type']} (nullable: {col['is_nullable']})")

            logger.info("=== Preparing INSERT query ===")
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

            
            logger.info("Query prepared successfully")
            
            # Log parameter values (truncated for readability)
            logger.info("=== Parameter values ===")
            logger.info(f"$1 userId: {input_data['userId']}")
            logger.info(f"$2 hrId: {input_data['hrId']}")
            logger.info(f"$3 departement: {input_data['departement']}")
            logger.info(f"$4 executive_summary: {input_data['report']['executive_summary'][:100]}...")
            logger.info(f"$13 genius_factor_score: {input_data['report']['genius_factor_score']}")

            logger.info("=== Executing INSERT query ===")
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
                logger.info(f"✓ Record inserted successfully!")
                logger.info(f"  - ID: {result['id']}")
                logger.info(f"  - Created at: {result['createdAt']}")
                
                # Verify the record was saved by reading it back
                verify_record = await conn.fetchrow(
                    'SELECT id, "userId", "createdAt" FROM "IndividualEmployeeReport" WHERE id = $1',
                    result['id']
                )
                
                if verify_record:
                    logger.info(f"✓ Record verification successful: {verify_record}")
                else:
                    logger.warning("⚠ Could not verify saved record")

                return {
                    "status": "success", 
                    "saved_record_id": result['id'],
                    "created_at": result['createdAt'].isoformat() if result['createdAt'] else None
                }
            else:
                logger.error("❌ No result returned from INSERT query")
                return {"status": "error", "message": "No result returned from INSERT query"}

        except asyncpg.PostgresError as db_error:
            logger.error(f"❌ PostgreSQL Error: {str(db_error)}")
            logger.error(f"Error code: {db_error.sqlstate if hasattr(db_error, 'sqlstate') else 'Unknown'}")
            logger.error(f"Error detail: {getattr(db_error, 'detail', 'No detail available')}")
            return {"status": "error", "message": f"Database error: {str(db_error)}"}

        except (TypeError, ValueError) as json_error:
            logger.error(f"❌ JSON encoding error: {str(json_error)}")
            return {"status": "error", "message": f"JSON encoding error: {str(json_error)}"}

        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {"status": "error", "message": f"Unexpected error: {str(e)}"}

        finally:
            if conn:
                try:
                    await conn.close()
                    logger.info("✓ Database connection closed")
                except Exception as close_error:
                    logger.error(f"❌ Error closing database connection: {str(close_error)}")
                

    @staticmethod
    async def analyze_assessment(input_data: AssessmentData) -> Dict[str, Any]:
        """
        Endpoint for assessment analysis with real-time notifications and Next.js integration
        """
        try:
            logger.info(f"Starting assessment analysis for userId: {input_data.userId}, hrId: {input_data.hrId}")
            # generate a raw query to get the departement of the user from the userId
            # using asyncpg for raw query
            db_params = {
                "user": "postgres",
                "password": "root",
                "database": "genius_factor",
                "host": "localhost",
                "port": 5432
            }
            conn = await asyncpg.connect(**db_params)
                        # departement is text[] in db, so fetch last element of the array
            departement = await conn.fetchval('SELECT "department"[array_length("department", 1)] FROM "User" WHERE id = $1', input_data.userId)
            await conn.close()
            if not departement:
                departement = "Unknown"
            input_data = input_data.dict()
            input_data['departement'] = departement
            logger.info(f"User departement: {departement}") 

            # Validate input data for notification
            notification_data = {
                'employeeId': input_data['userId'],
                'hrId': input_data['hrId'],
                'employeeName': input_data['employeeName'],
                'employeeEmail': input_data['employeeEmail'],
                'message': 'Assessment analysis completed successfully!',
                'status':'unread'
            }
            for key, value in notification_data.items():
                if not isinstance(value, str) or not value.strip():
                    logger.error(f"Invalid notification data: {key} is empty or not a string")
                    await NotificationService.send_user_notification(
                        input_data['userId'],
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
                # Convert dict data back to AssessmentPart objects
                assessment_parts = [AssessmentPart(**part) for part in input_data['data']]
                basic_results = analyze_assessment_data(assessment_parts)
                logger.info("Basic analysis completed")
            except Exception as e:
                logger.error(f"Failed to analyze assessment data: {str(e)}")
                await NotificationService.send_user_notification(
                    input_data['userId'],
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
                    input_data['userId'],
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
                    input_data['userId'],
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
                    input_data['userId'],
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
                "hrId": input_data['hrId'],
                "departement": departement,
                "userId": input_data['userId'],
                "report": recommendations.get("report"),
                "risk_analysis": recommendations.get("risk_analysis"),
                "metadata": recommendations.get("metadata")
            }

            # 4. Send data to Next.js API (asynchronous call)
            nextjs_response = await AssessmentController.save_to_db(final_result)
            
            if nextjs_response.get("status") == "error":
                logger.warning(f"Next.js API call failed but proceeding: {nextjs_response.get('message')}")
                # Continue even if Next.js call fails, but log the warning

            # Send success notification via Socket.IO
            await NotificationService.send_user_notification(
                input_data['userId'],
                {
                    'message': 'Assessment analysis completed successfully!',
                    'employeeName': input_data['employeeName'],
                    'employeeEmail': input_data['employeeEmail'],
                 
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
                    input_data['userId'],
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
                input_data['userId'],
                {
                    'message': 'Assessment analysis failed',
                    'progress': 100,
                    'status': 'error',
                    'error': str(e)
                }
            )
            
            raise HTTPException(status_code=500, detail=str(e))
