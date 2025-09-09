from fastapi import HTTPException
from fastapi.responses import StreamingResponse
# from services.chat_service import ChatService
# from models.chat_models import ChatMessage
from services.hr_dashboard_services.chat_services import ChatService
from utils.models import ChatMessage
class ChatController:
    def __init__(self):
        self.chat_service = ChatService()

    async def chat_with_ai(self, chat_message: ChatMessage):
        """Handle streaming chat request with LangChain"""
        try:
            result = await self.chat_service.chat_with_ai(
                chat_message.hr_id,
                chat_message.department,
                chat_message.message,
                chat_message.dashboard_data or []
            )
            
            return StreamingResponse(
                result["stream"],
                media_type="text/plain",
                headers={"X-Conversation-ID": result["conversation_id"]}
            )
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat error: {str(e)}")

    async def get_conversation_history(self, hr_id: str, department: str):
        try:
            return self.chat_service.get_conversation_history(hr_id, department)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

    async def clear_conversation(self, hr_id: str, department: str):
        try:
            return self.chat_service.clear_conversation(hr_id, department)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")