from prisma import Prisma
from datetime import datetime
from utils.logger import logger
import json
class DBService:
    _db = None
    
    @classmethod
    async def _get_db(cls):
        if cls._db is None:
            cls._db = Prisma()
            await cls._db.connect()
        return cls._db
    
    @classmethod
    async def save_assessment(cls, user_id, results, overall_score, message):
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
        except Exception as e:
            logger.error(f"Error saving assessment: {str(e)}")
            raise
    
    @classmethod
    async def get_assessments_by_user(cls, user_id):
        try:
            db = await cls._get_db()
            assessments = await db.assessment.find_many(
                where={"userId": user_id},
                order={"createdAt": "desc"}  # using dict instead of array
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
    async def close_connection(cls):
        if cls._db is not None:
            await cls._db.disconnect()
            cls._db = None