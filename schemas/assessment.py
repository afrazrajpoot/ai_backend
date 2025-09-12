from pydantic import BaseModel, Field
from typing import List

class QuestionAnswer(BaseModel):
    question: str = Field(description="The question text")
    answer: str = Field(description="The user's answer")
    type: str = Field(description="The question type (theory or multiple-choice)")

class AssessInput(BaseModel):
    userId: str = Field(description="The ID of the user")
    answers: List[QuestionAnswer] = Field(description="List of question-answer pairs")

class GeniusFactor(BaseModel):
    name: str = Field(description="Name of the genius factor")
    score: int = Field(description="Score out of 100", ge=0, le=100)
    description: str = Field(description="Detailed description of the genius factor")

class LLMAssessmentResult(BaseModel):
    geniusFactors: List[GeniusFactor] = Field(description="List of genius factors")
    strengths: List[str] = Field(description="List of strengths")
    growthAreas: List[str] = Field(description="List of growth areas")

from typing import Optional






class OptionCounts(BaseModel):
    A: Optional[int] = 0
    B: Optional[int] = 0
    C: Optional[int] = 0
    D: Optional[int] = 0
    E: Optional[int] = 0
    F: Optional[int] = 0
    G: Optional[int] = 0
    H: Optional[int] = 0
    I: Optional[int] = 0

class AssessmentPart(BaseModel):
    part: str
    optionCounts: OptionCounts

class AssessmentData(BaseModel):
    data: List[AssessmentPart]
    userId: str
    hrId: str
    employeeName: str
    employeeEmail: str