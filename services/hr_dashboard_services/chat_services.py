from typing import List, Dict, Generator
import asyncio
from datetime import datetime
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv
from typing import AsyncGenerator
load_dotenv()

class ChatService:
    def __init__(self):
        self.conversations = {}
        self.llm = ChatOpenAI(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            model_name="gpt-3.5-turbo",
            temperature=0.7,
            streaming=True
        )

    def get_conversation_id(self, hr_id: str, department: str) -> str:
        return f"{hr_id}_{department}"

    def extract_department_metrics(self, dashboard_data: List[Dict], department: str) -> Dict:
        """Extract relevant metrics from dashboard data"""
        dept_data = None
        for dept in dashboard_data:
            if dept.get('name') == department:
                dept_data = dept
                break
        
        if not dept_data:
            return {}
        
        metrics = dept_data.get('metrics', {})
        avg_scores = metrics.get('avg_scores', {})
        
        return {
            'employee_count': dept_data.get('employee_count', 'N/A'),
            'retention_risk_score': avg_scores.get('retention_risk_score', 'N/A'),
            'engagement_score': avg_scores.get('engagement_score', 'N/A'),
            'mobility_opportunity_score': avg_scores.get('mobility_opportunity_score', 'N/A'),
            'productivity_score': avg_scores.get('productivity_score', 'N/A'),
            'genius_factor_score': avg_scores.get('genius_factor_score', 'N/A')
        }

    async def generate_ai_response_stream(self, hr_id: str, department: str, message: str, dashboard_data: List[Dict]) -> AsyncGenerator[str, None]:
        """Generate AI response using LangChain with streaming"""
        try:
            # Extract metrics from dashboard data
            metrics = self.extract_department_metrics(dashboard_data, department)
            
            # Create system prompt with department context
            system_prompt = f"""You are an expert HR retention and risk analysis specialist. 
            You're analyzing the {department} department with the following metrics:
            
            Employee Count: {metrics.get('employee_count', 'N/A')}
            Retention Risk Score: {metrics.get('retention_risk_score', 'N/A')}/100
            Engagement Score: {metrics.get('engagement_score', 'N/A')}/100  
            Mobility Opportunity Score: {metrics.get('mobility_opportunity_score', 'N/A')}/100
            Productivity Score: {metrics.get('productivity_score', 'N/A')}/100
            
            Provide specific, actionable advice about retention strategies, risk mitigation, 
            and department-specific recommendations. Be concise, professional, and data-driven.
            Focus on practical solutions that HR can implement."""
            
            # Use LangChain for streaming response
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=message)
            ]
            
            # Stream the response
            full_response = ""
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    full_response += content
                    yield content
                    
        except Exception as e:
            yield f"⚠️ I encountered an error: {str(e)}. Please try again."

    async def chat_with_ai(self, hr_id: str, department: str, message: str, dashboard_data: List[Dict]) -> Dict:
        """Handle chat with AI using LangChain"""
        conversation_id = self.get_conversation_id(hr_id, department)
        
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = []
        
        # Add user message to history
        self.conversations[conversation_id].append({
            "role": "user",
            "content": message,
            "timestamp": datetime.now().isoformat()
        })
        
        # Generate response
        response_stream = self.generate_ai_response_stream(hr_id, department, message, dashboard_data)
        
        # For true streaming, we return the stream immediately
        # The AI response will be added to history after completion
        return {
            "stream": response_stream,
            "conversation_id": conversation_id
        }

    # Add this method to get full conversation history
    def get_full_conversation_history(self, hr_id: str, department: str) -> List[Dict]:
        conversation_id = self.get_conversation_id(hr_id, department)
        return self.conversations.get(conversation_id, [])
    def clear_conversation(self, hr_id: str, department: str) -> Dict:
        conversation_id = self.get_conversation_id(hr_id, department)
        if conversation_id in self.conversations:
            self.conversations[conversation_id] = []
            return {"message": "Conversation cleared"}
        return {"message": "No conversation found"}