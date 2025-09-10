# app/controllers/user_controller.py
from fastapi import HTTPException, status
from typing import List, Optional, Dict, Any
from services.hr_dashboard_services.department_summary_service import UserService
class UserController:
    def __init__(self):
        self.user_service = UserService()

    async def get_department_aggregation(
        self, 
        department: Optional[str] = None,
        hr_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Controller method to get department aggregation
        """
        try:
            # Connect to database
            await self.user_service.connect()
            
            if department:
                # Get specific department aggregation
                result = await self.user_service.aggregate_users_by_department(department, hr_id)
                
                if not result:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"No employees found in department: {department}"
                    )
                
                return {
                    'success': True,
                    'data': result[0] if department and len(result) == 1 else result,
                    'message': f"Department aggregation for {department}" if department else "All departments aggregation"
                }
            else:
                # Get all departments aggregation
                result = await self.user_service.get_department_statistics(hr_id=hr_id)
                return {
                    'success': True,
                    'data': result,
                    'message': "All departments aggregation"
                }
                
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Error aggregating departments: {str(e)}"
            )
        finally:
            # Disconnect from database
            await self.user_service.disconnect()
