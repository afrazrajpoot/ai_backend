from fastapi import APIRouter
# from models.chat_models import ChatMessage
# from controllers.chat_controller import ChatController
from controllers.hr_dashboard_controllers.chat_controller import ChatController
from utils.models import ChatMessage

router = APIRouter(prefix="/api/chat", tags=["chat"])
controller = ChatController()

@router.post("")
async def chat_with_ai(chat_message: ChatMessage):
    return await controller.chat_with_ai(chat_message)

@router.get("/{hr_id}/{department}")
async def get_conversation_history(hr_id: str, department: str):
    return await controller.get_conversation_history(hr_id, department)

@router.delete("/{hr_id}/{department}")
async def clear_conversation(hr_id: str, department: str):
    return await controller.clear_conversation(hr_id, department)