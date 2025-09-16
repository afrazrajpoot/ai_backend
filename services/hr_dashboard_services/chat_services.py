from typing import List, Dict, AsyncGenerator
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import asyncpg
import json
import uuid
import logging
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class ChatService:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True
        )
        self.pool = None

    async def connect_db(self):
        """Connect to database"""
        try:
            db_params = {
                "user": "postgres",
                "password": "root",
                "database": "hr-management",
                "host": "localhost",
                "port": 5432
            }
            logger.info(f"Connecting to database: {db_params['host']}:{db_params['port']}/{db_params['database']}")

            # Connect to database using db_params
            self.pool = await asyncpg.create_pool(**db_params)
            async with self.pool.acquire() as connection:
                await connection.execute("SET search_path TO public")
            # print("Database connection established successfully")
        except Exception as e:
            # print(f"Database connection error: {e}")
            
            raise

    async def disconnect_db(self):
        """Disconnect from database"""
        if self.pool:
            await self.pool.close()

    def extract_department_metrics(self, dashboard_data: List[Dict], department: str) -> Dict:
        """Extract relevant metrics from dashboard data"""
        for dept in dashboard_data:
            if dept.get('name') == department:
                metrics = dept.get('metrics', {}).get('avg_scores', {})
                return {
                    'employee_count': dept.get('employee_count', 'N/A'),
                    'retention_risk_score': metrics.get('retention_risk_score', 'N/A'),
                    'engagement_score': metrics.get('engagement_score', 'N/A'),
                    'mobility_opportunity_score': metrics.get('mobility_opportunity_score', 'N/A'),
                    'productivity_score': metrics.get('productivity_score', 'N/A'),
                }
        return {}

    async def get_conversation(self, hr_id: str, department: str):
        """Get existing conversation using raw SQL"""
        async with self.pool.acquire() as conn:
            result = await conn.fetchrow(
                'SELECT id, hr_id, department, messages, "createdAt", "updatedAt" '
                'FROM chat_conversations '
                'WHERE hr_id = $1 AND department = $2',
                hr_id, department
            )
            return result

    async def save_conversation(self, hr_id: str, department: str, messages: list):
        """Save conversation to database using raw SQL with UUID"""
        async with self.pool.acquire() as conn:
            # Check if conversation exists
            existing = await self.get_conversation(hr_id, department)
            
            if existing:
                # Update existing conversation
                await conn.execute(
                    'UPDATE chat_conversations SET messages = $1, "updatedAt" = NOW() '
                    'WHERE hr_id = $2 AND department = $3',
                    json.dumps(messages), hr_id, department
                )
            else:
                # Insert new conversation with generated UUID
                new_id = str(uuid.uuid4())
                await conn.execute(
                    'INSERT INTO chat_conversations (id, hr_id, department, messages, "createdAt", "updatedAt") '
                    'VALUES ($1, $2, $3, $4, NOW(), NOW())',
                    new_id, hr_id, department, json.dumps(messages)
                )

    async def clear_conversation(self, hr_id: str, department: str) -> bool:
        """Clear conversation messages using raw SQL"""
        async with self.pool.acquire() as conn:
            result = await conn.execute(
                'UPDATE chat_conversations SET messages = \'[]\', "updatedAt" = NOW() '
                'WHERE hr_id = $1 AND department = $2',
                hr_id, department
            )
            return "UPDATE 1" in result

    async def generate_ai_response_stream(self, hr_id: str, department: str, message: str, dashboard_data: List[Dict]) -> AsyncGenerator[str, None]:
        """Generate AI response using LangChain with streaming"""
        try:
            # Get existing conversation
            conversation = await self.get_conversation(hr_id, department)
            previous_messages = json.loads(conversation['messages']) if conversation and conversation['messages'] else []
            
            # Extract metrics
            metrics = self.extract_department_metrics(dashboard_data, department)
            
            # Create system prompt
            system_prompt = f"""You are an expert HR retention and risk analysis specialist. 
            You're analyzing the {department} department with metrics:
            Employee Count: {metrics.get('employee_count', 'N/A')}
            Retention Risk Score: {metrics.get('retention_risk_score', 'N/A')}/100
            Engagement Score: {metrics.get('engagement_score', 'N/A')}/100  
            Provide specific, actionable advice about retention strategies and risk mitigation."""
            
            # Build messages with conversation history
            messages = [SystemMessage(content=system_prompt)]
            
            # Add conversation history if exists (limit to last 5 exchanges)
            for msg in previous_messages[-10:]:  # Keep last 10 messages (5 exchanges)
                if msg['role'] == 'user':
                    messages.append(HumanMessage(content=msg['content']))
            
            # Add current user message
            messages.append(HumanMessage(content=message))
            
            # Stream the response
            full_response = ""
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    full_response += content
                    yield content
            
            # Save conversation to database
            new_messages = previous_messages + [
                {'role': 'user', 'content': message, 'timestamp': datetime.now().isoformat()},
                {'role': 'assistant', 'content': full_response, 'timestamp': datetime.now().isoformat()}
            ]
            await self.save_conversation(hr_id, department, new_messages)
                    
        except Exception as e:
            error_msg = f"⚠️ Error: {str(e)}"
            yield error_msg

    async def chat_with_ai(self, hr_id: str, department: str, message: str, dashboard_data: List[Dict]) -> Dict:
        """Handle chat with AI"""
        response_stream = self.generate_ai_response_stream(hr_id, department, message, dashboard_data)
        
        return {
            "stream": response_stream,
            "conversation_id": f"{hr_id}_{department}"
        }

    async def get_conversation_history(self, hr_id: str, department: str) -> List[Dict]:
        """Get conversation history from database"""
        conversation = await self.get_conversation(hr_id, department)
        if conversation and conversation['messages']:
            return json.loads(conversation['messages'])
        return []