from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from services.employee_services.chat_service import ChatService

router = APIRouter(prefix="/chat", tags=["chat"])

# Create a single instance of ChatService to reuse across requests
chat_service = ChatService()

class ChatController:
    @staticmethod
    @router.post("/")
    async def handle_chat(body: dict):
        user_id = body.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="user_id is required")
        
        message = body.get("message")
        if not message:
            raise HTTPException(status_code=400, detail="message is required")
        
        try:
            # Return a streaming response
            return StreamingResponse(
                chat_service.process_chat_stream(user_id, message),
                media_type="text/plain"
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

    @staticmethod
    @router.get("/{user_id}")
    async def get_chats(user_id: str):
        """Retrieve all chat messages for a given user_id"""
        try:
            await chat_service.connect()
            chats = await chat_service.prisma.employeechat.find_many(
                where={"userId": user_id}
            )
            await chat_service.disconnect()
            
            if not chats:
                raise HTTPException(status_code=404, detail=f"No chats found for user_id: {user_id}")
            
            # Convert Prisma objects to a list of dictionaries for JSON response
            chat_list = [
                {
                    "id": chat.id,
                    "userId": chat.userId,
                    "message": chat.message,
                    "response": chat.response,
                    "createdAt": chat.createdAt.isoformat() if chat.createdAt else None
                }
                for chat in chats
            ]
            return chat_list
        except Exception as e:
            # logger.error(f"Failed to retrieve chats for user_id {user_id}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to retrieve chats: {str(e)}")
            await chat_service.disconnect()