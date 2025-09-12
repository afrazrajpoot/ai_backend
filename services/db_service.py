from prisma import Prisma
from utils.logger import logger
from typing import Dict, Any
import json

class DBService:
    _db = None
    
    @classmethod
    async def _get_db(cls):
        if cls._db is None:
            cls._db = Prisma()
            await cls._db.connect()
            logger.info("Prisma client connected in DBService")
        return cls._db
    
    @classmethod
    async def save_assessment(cls, user_id: str, results: Dict[str, Any], overall_score: float, message: str):
        try:
            db = await cls._get_db()
            await db.assessment.create(
                data={
                    "userId": user_id,
                    "results": json.dumps(results),
                    "overallScore": overall_score,
                    "message": message,
                }
            )
            logger.info(f"Assessment saved for user {user_id}")
        except Exception as e:
            logger.error(f"Error saving assessment: {str(e)}")
            raise
    
    @classmethod
    async def get_assessments_by_user(cls, user_id: str):
        try:
            db = await cls._get_db()
            assessments = await db.assessment.find_many(
                where={"userId": user_id},
                order={"createdAt": "desc"}
            )
            return [
                {
                    "id": assess.id,
                    "results": assess.results,
                    "overallScore": assess.overallScore,
                    "message": assess.message,
                    "createdAt": assess.createdAt.isoformat()
                }
                for assess in assessments
            ]
        except Exception as e:
            logger.error(f"Error getting assessments: {str(e)}")
            raise
    
    @classmethod
    async def save_notification(cls, notification_data: Dict[str, Any]):
        """
        Save a notification to the database.
        
        Args:
            notification_data: Dictionary containing notification fields (employeeId, hrId, employeeName, employeeEmail, message)
        """
        try:
            db = await cls._get_db()
            await db.notification.create(data=notification_data)
            logger.info("Notification saved to database")
        except Exception as e:
            logger.error(f"Failed to save notification to database: {str(e)}")
            raise
    
    @classmethod
    async def close_connection(cls):
        if cls._db is not None:
            try:
                await cls._db.disconnect()
                logger.info("Prisma client disconnected in DBService")
                cls._db = None
            except Exception as e:
                logger.error(f"Failed to disconnect from Prisma in DBService: {str(e)}")