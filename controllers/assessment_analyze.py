from schemas.assessment import AssessmentData
from typing import Dict, Any
from prisma import Prisma
from utils.logger import logger
import json
import asyncpg


class AssessmentController:

    @staticmethod
    async def analyze_assessment(input_data: AssessmentData) -> Dict[str, Any]:
        """
        Minimal version: saves hardcoded data using raw PostgreSQL (asyncpg).
        """
        logger.info("=== Starting analyze_assessment method ===")
        
        hardcoded_data = {
            "userId": "33b003c7-ad86-4ecf-bb54-5b2a2c633926",
            "hrId": "42632d76-8d0d-4a59-af5c-b0172c5aaa6f",
            "departement": "Media",
            "executiveSummary": "This comprehensive report evaluates the Genius Factor assessment results for the employee, identifying them as a Tech Genius.",
            "geniusFactorProfileJson": {
                "primary_genius_factor": "Tech Genius",
                "description": "Individuals identified as Tech Geniuses excel in systematic thinking and technical problem-solving.",
                "key_strengths": [
                    "Systematic Problem-Solving: Tech Geniuses can break down complex issues into manageable components.",
                    "Analytical Thinking: Their ability to analyze data and trends enables them to make informed decisions."
                ],
                "secondary_genius_factor": "Tech Genius",
                "secondary_description": "As the secondary genius factor, the Tech Genius trait reinforces the primary genius.",
                "energy_sources": [
                    "Engaging in complex problem-solving tasks energizes Tech Geniuses.",
                    "Collaborating with like-minded individuals on technical projects."
                ]
            },
            "currentRoleAlignmentAnalysisJson": {
                "alignment_score": "85",
                "assessment": "The employee's alignment score of 85 reflects a strong fit between their genius factors and their current role.",
                "strengths_utilized": [
                    "The employee effectively applies their systematic problem-solving skills in project management."
                ],
                "underutilized_talents": [
                    "The employee's potential for leadership in technical projects is not fully realized."
                ],
                "retention_risk_level": "Low retention risk level due to strong alignment with role."
            },
            "internalCareerOpportunitiesJson": {
                "primary_industry": "Technology",
                "secondary_industry": "Financial Services",
                "recommended_departments": ["Software Development", "Cybersecurity"],
                "specific_role_suggestions": ["Software Engineer", "Cybersecurity Analyst"],
                "career_pathways": {
                    "Development Track": "Software Engineer → Senior Developer → Technical Lead"
                },
                "transition_timeline": {
                    "six_month": "Transition into a Senior Developer role with increased responsibilities."
                },
                "required_skill_development": [
                    "Advanced programming languages (Python, JavaScript)"
                ]
            },
            "retentionAndMobilityStrategiesJson": {
                "retention_strategies": [
                    "Implement mentorship programs that pair Tech Geniuses with leadership roles."
                ],
                "internal_mobility_recommendations": [
                    "Encourage participation in cross-departmental projects."
                ],
                "development_support": [
                    "Allocate resources for continuous learning and professional development."
                ]
            },
            "developmentActionPlanJson": {
                "thirty_day_goals": [
                    "Identify and enroll in a relevant online course focused on advanced programming languages."
                ],
                "ninety_day_goals": [
                    "Lead a small project team to enhance leadership skills."
                ],
                "six_month_goals": [
                    "Take on a technical lead role in a major project."
                ],
                "networking_strategy": [
                    "Join professional associations related to technology and cybersecurity."
                ]
            },
            "personalizedResourcesJson": {
                "affirmations": [
                    "I am a natural problem solver, capable of overcoming any technical challenge."
                ],
                "mindfulness_practices": [
                    "Practice deep breathing exercises for 5 minutes daily."
                ],
                "reflection_questions": [
                    "What technical challenges have I successfully overcome in the past month?"
                ],
                "learning_resources": [
                    "Online courses in advanced programming languages."
                ]
            },
            "dataSourcesAndMethodologyJson": {
                "data_sources": [
                    "Genius Factor Assessment for Fortune 1000 HR Departments.pdf"
                ],
                "methodology": "The analysis was conducted using a comprehensive framework."
            },
            "risk_analysis": {
                "scores": {
                    "genius_factor_score": 90,
                    "retention_risk_score": 20,
                    "mobility_opportunity_score": 95
                },
                "trends": {
                    "risk_factors": "Low retention risk due to strong alignment.",
                    "mobility_trends": "Internal mobility is gaining traction.",
                    "retention_trends": "Organizations are prioritizing employee retention."
                },
                "company": "Fortune 1000 Company",
                "genius_factors": ["General Talent"],
                "recommendations": [
                    "Implement a mentorship program."
                ],
                "analysis_summary": "Comprehensive risk analysis completed."
            },
            "geniusFactorScore": 85
        }

        conn = None
        try:
            logger.info("=== Validating JSON data ===")
            # Validate JSON data before saving
            json_test = json.dumps(hardcoded_data)
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
                    "geniusFactorScore"
                )
                VALUES ($1, $2, $3, $4, $5::jsonb, $6::jsonb, $7::jsonb, $8::jsonb, $9::jsonb, $10::jsonb, $11::jsonb, $12::jsonb, $13)
                RETURNING id, "createdAt"
            """
            
            logger.info("Query prepared successfully")
            
            # Log parameter values (truncated for readability)
            logger.info("=== Parameter values ===")
            logger.info(f"$1 userId: {hardcoded_data['userId']}")
            logger.info(f"$2 hrId: {hardcoded_data['hrId']}")
            logger.info(f"$3 departement: {hardcoded_data['departement']}")
            logger.info(f"$4 executiveSummary: {hardcoded_data['executiveSummary'][:100]}...")
            logger.info(f"$13 geniusFactorScore: {hardcoded_data['geniusFactorScore']}")

            logger.info("=== Executing INSERT query ===")
            # Execute the query
            result = await conn.fetchrow(
                query,
                hardcoded_data["userId"],
                hardcoded_data["hrId"],
                hardcoded_data["departement"],
                hardcoded_data["executiveSummary"],
                json.dumps(hardcoded_data["geniusFactorProfileJson"]),
                json.dumps(hardcoded_data["currentRoleAlignmentAnalysisJson"]),
                json.dumps(hardcoded_data["internalCareerOpportunitiesJson"]),
                json.dumps(hardcoded_data["retentionAndMobilityStrategiesJson"]),
                json.dumps(hardcoded_data["developmentActionPlanJson"]),
                json.dumps(hardcoded_data["personalizedResourcesJson"]),
                json.dumps(hardcoded_data["dataSourcesAndMethodologyJson"]),
                json.dumps(hardcoded_data["risk_analysis"]),
                hardcoded_data["geniusFactorScore"]
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

        except json.JSONEncodeError as json_error:
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
                    
        logger.info("=== analyze_assessment method completed ===")