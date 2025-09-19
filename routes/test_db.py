from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
from controllers.assessment_analyze import AssessmentController
from utils.logger import logger
import json

router = APIRouter()

class TestReportRequest(BaseModel):
    userId: str
    report: Dict[str, Any]
    risk_analysis: Optional[Dict[str, Any]] = {}
    metadata: Optional[Dict[str, Any]] = {}

@router.post("/test-db-save")
async def test_database_save(request: TestReportRequest):
    """
    Test endpoint for saving assessment report to database
    """
    try:
   
        
        # Prepare assessment data in the same format as the main endpoint
        assessment_data = {
            "status": "success",
            "userId": request.userId,
            "report": request.report,
            "risk_analysis": request.risk_analysis,
            "metadata": request.metadata
        }
        
        # Log the incoming data structure

        
        # Try to save to database
        db_response = await AssessmentController.save_to_database(assessment_data)
        
        if db_response.get("status") == "success":
            return {
                "status": "success",
                "message": "Report saved successfully to database",
                "report_id": db_response.get("report_id"),
                "db_response": db_response
            }
        else:
            return {
                "status": "error",
                "message": "Failed to save report to database",
                "error": db_response.get("message"),
                "db_response": db_response
            }
            
    except Exception as e:
        logger.error(f"Error in test database save endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/test-db-save-simple")
async def test_database_save_simple(data: Dict[str, Any]):
    """
    Simple test endpoint that accepts any JSON and tries to save it
    """
    try:
   
        
        # Create minimal test data if userId not provided
        if "userId" not in data:
            data["userId"] = "test-user-id"
        
        # Ensure required structure
        if "report" not in data:
            data["report"] = {}
            
        if "risk_analysis" not in data:
            data["risk_analysis"] = {}
            
        # Try to save to database
        db_response = await AssessmentController.save_to_database(data)
        
        return {
            "status": "test_completed",
            "input_data_keys": list(data.keys()),
            "db_response": db_response
        }
            
    except Exception as e:
        logger.error(f"Error in simple test database save endpoint: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "input_data_keys": list(data.keys()) if isinstance(data, dict) else "not_dict"
        }

@router.get("/test-db-connection")
async def test_database_connection():
    """
    Test database connection
    """
    try:
        from prisma import Prisma
        
        prisma = Prisma()
        await prisma.connect()
        
        # Try to count users
        user_count = await prisma.user.count()
        
        await prisma.disconnect()
        
        return {
            "status": "success",
            "message": "Database connection successful",
            "user_count": user_count
        }
        
    except Exception as e:
        logger.error(f"Database connection test failed: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }
