import asyncio
import json
from unittest.mock import MagicMock, patch
from controllers.assessment_analyze import AssessmentController
from schemas.assessment import AssessmentData, AssessmentPart, OptionCounts
from services.notification_service import NotificationService
from services.db_service import DBService

async def test_controller():
    print("Starting AssessmentController Verification...")

    # Mock Data
    input_data = AssessmentData(
        data=[
            AssessmentPart(part="Part I: Self-Awareness", optionCounts=OptionCounts(A=5, B=2, C=1, D=2)),
            AssessmentPart(part="Part II: Talent Audit", optionCounts=OptionCounts(A=10, B=3, C=2, D=0)),
            AssessmentPart(part="Part III: Passion Audit", optionCounts=OptionCounts(A=8, B=4, C=3, D=0)),
            AssessmentPart(part="Part IV: Genius Factor Mapping", optionCounts=OptionCounts(A=10, B=0, C=0, D=0, E=0, F=2, G=0, H=0, I=0)),
            AssessmentPart(part="Part V: Goals", optionCounts=OptionCounts(A=5, B=5))
        ],
        allAnswers=[],
        userId="test_user_id",
        hrId="test_hr_id",
        departement="Engineering",
        employeeName="Test Employee",
        employeeEmail="test@example.com",
        is_paid=True
    )

    # Mock Dependencies
    with patch('controllers.assessment_analyze.asyncpg.connect') as mock_connect, \
         patch('services.notification_service.NotificationService.send_user_notification') as mock_notify, \
         patch('services.db_service.DBService.save_notification') as mock_save_notif, \
         patch('controllers.assessment_analyze.AssessmentController.save_to_db') as mock_save_db, \
         patch('services.ai_service.AIService._fetch_user_data', new_callable=MagicMock) as mock_fetch_user:
         
        # Make _fetch_user_data async
        async def mock_fetch_user_data(*args, **kwargs):
            return {}
        mock_fetch_user.side_effect = mock_fetch_user_data

        # Setup Mock DB Connection
        mock_conn = MagicMock()
        mock_connect.return_value = mock_conn
        
        # Make fetchval an async mock
        async def mock_fetchval(*args, **kwargs):
            return "Engineering"
        mock_conn.fetchval = mock_fetchval
        
        # Make close an async mock
        async def mock_close(*args, **kwargs):
            pass
        mock_conn.close = mock_close
        
        # Mock save_to_db response
        mock_save_db.return_value = {"status": "success", "saved_record_id": "123"}

        try:
            print("Calling analyze_assessment...")
            result = await AssessmentController.analyze_assessment(input_data)
            
            if result.get("success"):
                print("\nSUCCESS: Assessment analyzed successfully!")
                print("-" * 50)
                print(f"Report Generated: {result.get('report') is not None}")
                print(f"Risk Analysis: {result.get('risk_analysis') is not None}")
                print("-" * 50)
            else:
                print(f"\nFAILED: {result}")

        except Exception as e:
            print(f"\nEXCEPTION: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_controller())
