
from typing import List, Dict, Optional,Any

from pydantic import BaseModel, Field


from utils.logger import logger
from config import settings
# Pydantic Model for the Report Structure
class GeniusFactorProfile(BaseModel):
    primary_genius_factor: str = Field(description="Primary Genius Factor with confidence level")
    description: str = Field(description="In-depth description of the primary factor")
    key_strengths: List[str] = Field(description="Top 3 natural abilities with detailed descriptions")
    secondary_genius_factor: str = Field(default="", description="Secondary Genius Factor with confidence level, if applicable")
    secondary_description: str = Field(default="", description="Description of the secondary factor's complementary traits")
    energy_sources: List[str] = Field(description="Top 3 activities that create flow state with detailed descriptions")

class CurrentRoleAlignmentAnalysis(BaseModel):
    alignment_score: str = Field(description="Alignment score out of 100")
    assessment: str = Field(description="Thorough explanation of role alignment")
    strengths_utilized: List[str] = Field(description="Aligned aspects with detailed descriptions")
    underutilized_talents: List[str] = Field(description="Misaligned aspects with detailed descriptions")
    retention_risk_level: str = Field(description="Retention risk level with rationale")

class InternalCareerOpportunities(BaseModel):
    primary_industry: str = Field(description="Primary industry aligned with Genius Factor")
    secondary_industry: str = Field(description="Secondary industry with growth opportunities")
    recommended_departments: List[str] = Field(description="Departments aligned with Genius Factor")
    specific_role_suggestions: List[str] = Field(description="3-5 internal position suggestions")
    career_pathways: Dict[str, str] = Field(description="Short-term and long-term career pathways")
    transition_timeline: Dict[str, str] = Field(description="6-month, 1-year, 2-year pathways")
    required_skill_development: List[str] = Field(description="Specific competencies to build")

class RetentionAndMobilityStrategies(BaseModel):
    retention_strategies: List[str] = Field(description="Organizational retention approaches")
    internal_mobility_recommendations: List[str] = Field(description="Strategies for internal mobility")
    development_support: List[str] = Field(description="Support mechanisms for development")

class DevelopmentActionPlan(BaseModel):
    thirty_day_goals: List[str] = Field(description="Immediate actionable steps")
    ninety_day_goals: List[str] = Field(description="Short-term skill-building activities")
    six_month_goals: List[str] = Field(description="Project leadership or career roadmap tasks")
    networking_strategy: List[str] = Field(description="Key relationships to build with actions")

class PersonalizedResources(BaseModel):
    affirmations: List[str] = Field(description="Genius Factor-specific affirmations")
    mindfulness_practices: List[str] = Field(description="Daily mindfulness practices")
    reflection_questions: List[str] = Field(description="Weekly reflection questions")
    learning_resources: List[str] = Field(description="Recommended courses, books, or tools")

class DataSourcesAndMethodology(BaseModel):
    data_sources: List[str] = Field(description="Sources used in the analysis")
    methodology: str = Field(description="Summary of the assessment process")

class IndividualEmployeeReport(BaseModel):
    executive_summary: str = Field(description="Detailed overview of Genius Factors and recommendations")
    genius_factor_profile: GeniusFactorProfile
    current_role_alignment_analysis: CurrentRoleAlignmentAnalysis
    internal_career_opportunities: InternalCareerOpportunities
    retention_and_mobility_strategies: RetentionAndMobilityStrategies
    development_action_plan: DevelopmentActionPlan
    personalized_resources: PersonalizedResources
    data_sources_and_methodology: DataSourcesAndMethodology
    genius_factor_score: int


class EmployeeRequest(BaseModel):
    employeeId: str




class DepartmentMetrics(BaseModel):
    avg_scores: Dict[str, float]
    employee_count: int
    engagement_distribution: Dict[str, int]
    first_report_date: str
    genius_factor_distribution: Dict[str, int]
    last_report_date: str
    mobility_trend: Dict[str, int]
    productivity_distribution: Dict[str, int]
    retention_risk_distribution: Dict[str, int]
    skills_alignment_distribution: Dict[str, int]

class DepartmentInput(BaseModel):
    color: str
    completion: int
    employee_count: int
    metrics: DepartmentMetrics
    name: str

class AnalysisRequest(BaseModel):
    departments: List[DepartmentInput]

class RecommendationCard(BaseModel):
    department: str
    risk_level: str
    retention_score: float
    mobility_opportunities: List[str]
    recommendations: List[str]
    action_items: List[str]

class AnalysisResponse(BaseModel):
    overall_risk_score: float
    department_recommendations: List[RecommendationCard]
    summary: str


class ChatMessage(BaseModel):
    hr_id: str
    department: str
    message: str
    dashboard_data: Optional[List[Dict[str, Any]]] = None

class ChatResponse(BaseModel):
    response: str
    conversation_id: str




class UserIdRequest(BaseModel):
    user_id: str

class Skill(BaseModel):
    name: str
    proficiency: int

class RecommendedCourse(BaseModel):
    title: str
    provider: str
    url: str
    reason: str

class ProgressTracking(BaseModel):
    current_position: str
    previous_position: Optional[str]
    current_department: str
    previous_department: Optional[str]

class EmployeeLearningResponse(BaseModel):
    employee_id: str
    employee_name: str
    current_skills: List[Skill]  # Changed to List[Skill]
    recommended_courses: List[RecommendedCourse]
    progress_tracking: ProgressTracking