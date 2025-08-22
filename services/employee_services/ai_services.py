from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from typing import Dict, Any


class RecommendationService:
    def __init__(self):
        # Initialize OpenAI model
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

    def recommend_companies(self, employee: Dict[str, Any], companies: Dict[str, Any]) -> str:
        """
        Recommend best companies for an employee based on metadata.
        """
        prompt_template = PromptTemplate(
            input_variables=["employee", "companies"],
            template=(
                "You are an AI recommendation engine.\n\n"
                "Employee data:\n{employee}\n\n"
                "Available companies:\n{companies}\n\n"
                "Recommend the most suitable companies for this employee. "
                "Return a JSON list of recommended companies with reasons."
                "Format the output as a JSON array with each company having 'name' and 'reason' and match 'score' in range 10 to 100 percent fields.\n\n"
            )
        )

        # Format prompt
        prompt = prompt_template.format(
            employee=str(employee),
            companies=str(companies)
        )

        response = self.llm.invoke(prompt)
        return response.content
