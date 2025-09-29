import os
import json
import logging
from typing import TypedDict, Annotated, List, AsyncGenerator
from uuid import uuid4
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from prisma import Prisma
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Use streaming-enabled LLM
llm = ChatOpenAI(
    model="gpt-4o", 
    temperature=0,
    streaming=True  # Enable streaming
)
checkpointer = MemorySaver()

# System prompt with explicit instructions for career recommendations
system_prompt_template = """You are a helpful HR assistant. Use the following employee data and report to provide context for your responses:
Employee Profile: {employee_profile}
Employee Report: {employee_report}
For career-related queries, prioritize the employee data to provide personalized recommendations, leveraging details such as skills, department, genius factor score, and internal career opportunities. If no employee data is available, inform the user that specific details are needed and request them to provide more information. Answer the user's query accurately and professionally, using the provided data and limiting conversational context to the last 5 messages."""

class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], "add_messages"]
    user_id: str

class ChatService:
    def __init__(self):
        try:
            self.prisma = Prisma(auto_register=True)
        except Exception as e:
            logger.error(f"Failed to initialize Prisma client: {str(e)}")
            raise RuntimeError(f"Prisma initialization failed: {str(e)}")
        self.graph = self._build_graph()
        self.runnable = self.graph.compile(checkpointer=checkpointer)

    async def connect(self):
        if not isinstance(self.prisma, Prisma):
            logger.error(f"Prisma is not a valid Prisma client: {type(self.prisma)}")
            raise RuntimeError("Invalid Prisma client")
        try:
            await self.prisma.connect()
        except Exception as e:
            logger.error(f"Prisma connection failed: {str(e)}")
            raise

    async def disconnect(self):
        if not isinstance(self.prisma, Prisma):
            logger.error(f"Prisma is not a valid Prisma client: {type(self.prisma)}")
            return
        try:
            await self.prisma.disconnect()
        except Exception as e:
            logger.error(f"Prisma disconnection failed: {str(e)}")

    def custom_json_serializer(self, obj):
        """Custom JSON serializer to handle datetime objects and other non-serializable types."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    async def fetch_employee_data(self, user_id: str) -> dict:
        """Fetch employee profile and report from the database."""
        try:
            user = await self.prisma.user.find_unique(
                where={"id": user_id},
                include={"employee": True}
            )
            report = await self.prisma.individualemployeereport.find_first(
                where={"userId": user_id}
            )
            user_dict = user.dict() if user else {}
            report_dict = report.dict() if report else {}
            employee_data = {
                "employee_profile": json.dumps(user_dict, default=self.custom_json_serializer),
                "employee_report": json.dumps(report_dict, default=self.custom_json_serializer)
            }
            return employee_data
        except Exception as e:
            logger.error(f"Failed to fetch employee data for user_id {user_id}: {str(e)}")
            return {
                "employee_profile": json.dumps({}),
                "employee_report": json.dumps({})
            }

    async def save_chat_message(self, user_id: str, message: str, response: str = None):
        """Save a chat message and its response to the database."""
        try:
            await self.prisma.employeechat.create(
                data={
                    "id": str(uuid4()),
                    "userId": user_id,
                    "message": message,
                    "response": response,
                    "createdAt": datetime.now()
                }
            )
            logger.info(f"Chat message saved for user_id {user_id}")
        except Exception as e:
            logger.error(f"Failed to save chat message for user_id {user_id}: {str(e)}")
            raise

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("call_model", self.call_model)
        graph.set_entry_point("call_model")
        graph.add_edge("call_model", END)
        return graph

    async def call_model(self, state: AgentState):
        # Fetch employee data
        employee_data = await self.fetch_employee_data(state["user_id"])
        
        # Get last 5 messages for context
        messages = state["messages"][-5:] if len(state["messages"]) > 5 else state["messages"]
        
        # Format system prompt with employee data
        system_message = SystemMessage(
            content=system_prompt_template.format(
                employee_profile=employee_data["employee_profile"],
                employee_report=employee_data["employee_report"]
            )
        )
        
        # Combine system message with recent messages
        input_messages = [system_message] + messages
        try:
            response = llm.invoke(input_messages)
            return {"messages": [response]}
        except Exception as e:
            logger.error(f"LLM invocation failed: {str(e)}")
            return {"messages": [AIMessage(content="Sorry, I encountered an error while processing your request.")]}

    async def process_chat_stream(self, user_id: str, message: str) -> AsyncGenerator[str, None]:
        """Streaming version of process_chat that returns plain text"""
        try:
            await self.connect()
            
            # Generate a unique ID for this chat session
            chat_id = str(uuid4())
            
            # Save the user's message with a unique ID
            await self.prisma.employeechat.create(
                data={
                    "id": chat_id,
                    "userId": user_id,
                    "message": message,
                    "createdAt": datetime.now()
                }
            )
            
            # Fetch employee data for system prompt
            employee_data = await self.fetch_employee_data(user_id)
            
            # Format system prompt with employee data
            system_message = SystemMessage(
                content=system_prompt_template.format(
                    employee_profile=employee_data["employee_profile"],
                    employee_report=employee_data["employee_report"]
                )
            )
            
            # Create messages with system prompt and user message
            messages = [system_message, HumanMessage(content=message)]
            
            # Collect response chunks to save later
            response_chunks = []
            async for chunk in llm.astream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    response_chunks.append(chunk.content)
                    yield chunk.content
            
            # Update the existing record with the full AI response
            full_response = "".join(response_chunks)
            await self.prisma.employeechat.update(
                where={"id": chat_id},  # Use the same ID to update the record
                data={"response": full_response}
            )
            
            logger.info(f"Chat stream processed successfully for user_id {user_id}")
            
        except Exception as e:
            logger.error(f"Chat stream processing failed for user_id {user_id}: {str(e)}")
            yield f"Error: {str(e)}"
        finally:
            await self.disconnect()
    async def process_chat(self, user_id: str, message: str) -> dict:
        try:
            await self.connect()
            session_id = str(uuid4())
            input_dict = {
                "messages": [HumanMessage(content=message)],
                "user_id": user_id
            }
            config = {"configurable": {"thread_id": session_id}}
            result = await self.runnable.ainvoke(input_dict, config)
            
            response_message = result["messages"][-1].content if result["messages"] else "No response generated."
            
            # Save the user message and AI response
            await self.save_chat_message(user_id, message, response_message)
            
            logger.info(f"Chat processed successfully for user_id {user_id}: {response_message[:100]}...")
            return {
                "response": response_message
            }
        except Exception as e:
            logger.error(f"Chat processing failed for user_id {user_id}: {str(e)}")
            raise
        finally:
            await self.disconnect()