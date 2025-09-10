# app/services/hr_dashboard_services/department_summary_service.py
import os
from typing import List, Optional, Dict, Any
from prisma import Prisma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Pydantic models for structured output with detailed field descriptions
class TeamCompositionOverview(BaseModel):
    overview: str = Field(description="Detailed analysis of team composition including roles, size, genius factor patterns, and overall structure. Should be 3-5 sentences with specific observations.")

class StrengthsIdentified(BaseModel):
    strengths: List[str] = Field(description="List 5-8 specific strengths identified based on genius factors and team composition. Each strength should be a complete sentence explaining why it's valuable.")

class CriticalGaps(BaseModel):
    gaps: List[str] = Field(description="List 3-5 critical gaps or areas for improvement. Each gap should be specific and explain the potential impact on department performance.")

class RecommendationsForBalance(BaseModel):
    recommendations: List[str] = Field(description="List 4-6 specific recommendations for balancing the team composition. Include hiring suggestions, role adjustments, and skill development.")

class TargetedTrainingDevelopment(BaseModel):
    training_recommendations: List[str] = Field(description="List 5-7 targeted training and development suggestions. Include specific skills, technologies, and development programs.")

class TeamBuildingCollaboration(BaseModel):
    collaboration_recommendations: List[str] = Field(description="List 4-6 team building and collaboration recommendations. Include specific activities, processes, and communication strategies.")

class RiskMitigation(BaseModel):
    risks: List[str] = Field(description="List 3-5 potential risks identified from the analysis. Each risk should be specific and explain the potential consequences.")
    mitigation_strategies: List[str] = Field(description="List 3-5 specific strategies to mitigate the identified risks. Each strategy should be actionable and practical.")

class DepartmentRecommendations(BaseModel):
    team_composition_overview: TeamCompositionOverview = Field(description="Team Composition Overview")
    strengths_identified: StrengthsIdentified = Field(description="Strengths Identified")
    critical_gaps: CriticalGaps = Field(description="Critical Gaps")
    recommendations_for_balance: RecommendationsForBalance = Field(description="Recommendations for Balance")
    targeted_training_development: TargetedTrainingDevelopment = Field(description="Targeted Training & Development")
    team_building_collaboration: TeamBuildingCollaboration = Field(description="Team Building & Collaboration")
    risk_mitigation: RiskMitigation = Field(description="Risk Mitigation")

class UserService:
    def __init__(self):
        self.db = Prisma()
        # Initialize LLM with GPT-4o for better reasoning
        self.llm = ChatOpenAI(
            model_name="gpt-4o",
            temperature=0.7,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize output parser
        self.parser = PydanticOutputParser(pydantic_object=DepartmentRecommendations)

    async def connect(self):
        await self.db.connect()

    async def disconnect(self):
        await self.db.disconnect()

    async def aggregate_users_by_department(
        self, 
        department_name: Optional[str] = None,
        hr_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Aggregate users by department, counting employees in each department.
        """
        # Build the where clause for users
        where_clause = {}
        
        if hr_id:
            where_clause['hrId'] = hr_id
        
        if department_name:
            # Filter users where the last item in department array matches the department_name
            where_clause['department'] = {
                'hasSome': [department_name]
            }
        
        # Get all users with the filters
        users = await self.db.user.find_many(
            where=where_clause,
            include={
                'employee': True
            }
        )
        
        # If specific department requested, return detailed data with reports
        if department_name:
            department_employees = []
            
            for user in users:
                if user.department and len(user.department) > 0:
                    # Get the current department (last item in array)
                    current_department = user.department[-1]
                    
                    if current_department == department_name:
                        # Get the latest report for this user from IndividualEmployeeReport
                        latest_report = await self.db.individualemployeereport.find_first(
                            where={
                                'userId': user.id
                            },
                            order={
                                'createdAt': 'desc'
                            }
                        )
                        
                        employee_data = {
                            'userId': user.id,
                            'firstName': user.firstName,
                            'lastName': user.lastName,
                            'email': user.email,
                            'position': user.position[-1] if user.position and len(user.position) > 0 else None,
                            'hrId': user.hrId,
                            'hasReport': latest_report is not None
                        }
                        
                        # Add employee ID if available
                        if user.employee:
                            employee_data['employeeId'] = user.employee.id
                        
                        # Add genius factor profile data if report exists
                        if latest_report:
                            employee_data['geniusFactorProfile'] = latest_report.geniusFactorProfileJson
                            employee_data['geniusFactorScore'] = latest_report.geniusFactorScore
                            employee_data['reportId'] = latest_report.id
                            employee_data['reportCreatedAt'] = latest_report.createdAt.isoformat()
                        else:
                            employee_data['geniusFactorProfile'] = None
                            employee_data['geniusFactorScore'] = None
                            employee_data['reportId'] = None
                            employee_data['reportCreatedAt'] = None
                        
                        department_employees.append(employee_data)
            
            # Sort employees by last name, then first name
            department_employees.sort(key=lambda x: (x['lastName'] or '', x['firstName'] or ''))
            
            # Generate LLM recommendations for the department
            llm_recommendations = await self.generate_department_recommendations(
                department_name, 
                department_employees
            )
            
            return [{
                'department': department_name,
                'employee_count': len(department_employees),
                'employees': department_employees,
                'llm_recommendations': llm_recommendations
            }]
        
        # For general aggregation (no specific department)
        department_counts = {}
        
        for user in users:
            if user.department and len(user.department) > 0:
                # Get the current department (last item in array)
                current_department = user.department[-1]
                
                if current_department not in department_counts:
                    department_counts[current_department] = {
                        'count': 0,
                        'employees': []
                    }
                
                department_counts[current_department]['count'] += 1
                
                # Add basic employee details
                employee_info = {
                    'id': user.id,
                    'firstName': user.firstName,
                    'lastName': user.lastName,
                    'email': user.email,
                    'position': user.position[-1] if user.position and len(user.position) > 0 else None
                }
                
                if user.employee:
                    employee_info['employeeId'] = user.employee.id
                
                department_counts[current_department]['employees'].append(employee_info)
        
        # Convert to list format for response
        result = [
            {
                'department': dept,
                'employee_count': data['count'],
                'employees': data['employees']
            }
            for dept, data in department_counts.items()
        ]
        
        # Sort by department name
        result.sort(key=lambda x: x['department'])
        
        return result

    async def generate_department_recommendations(
        self, 
        department_name: str, 
        employees: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate detailed LLM-powered recommendations for department enhancement.
        """
        try:
            # Prepare data for LLM analysis
            employees_with_reports = sum(1 for emp in employees if emp['hasReport'])
            genius_factor_scores = [emp['geniusFactorScore'] for emp in employees if emp['geniusFactorScore'] is not None]
            positions = list(set(emp['position'] for emp in employees if emp['position']))
            
            # Extract genius factor profiles for analysis
            genius_profiles = []
            for emp in employees:
                if emp['geniusFactorProfile']:
                    profile = emp['geniusFactorProfile']
                    if isinstance(profile, dict):
                        genius_profiles.append(profile)
                    elif isinstance(profile, str):
                        try:
                            genius_profiles.append(json.loads(profile))
                        except:
                            genius_profiles.append({'raw_data': profile})
            
            # Calculate average score
            avg_score = sum(genius_factor_scores) / len(genius_factor_scores) if genius_factor_scores else None
            
            # Get employee names for context
            employee_names = [f"{emp['firstName']} {emp['lastName']}" for emp in employees if emp['firstName'] and emp['lastName']]
            
            # Create the prompt with detailed format instructions
            format_instructions = self.parser.get_format_instructions()
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert HR consultant and organizational psychologist with 20+ years of experience. 
                Analyze department data and provide comprehensive, detailed, and actionable recommendations to enhance 
                department efficiency using genius factor insights.

                CRITICAL INSTRUCTIONS:
                1. Provide EXTENSIVE, DETAILED recommendations in EXACTLY this structured format
                2. Each section should contain 3-8 specific, actionable items
                3. Be data-driven and reference specific genius factor attributes
                4. Focus on practical, implementable strategies
                5. Use professional HR consulting language
                6. Consider both individual strengths and team dynamics

                REQUIRED SECTIONS (use these exact headings):
                1. Team Composition Overview - Detailed analysis of team structure
                2. Strengths Identified - 5-8 specific strengths with explanations
                3. Critical Gaps - 3-5 specific gaps with impact analysis
                4. Recommendations for Balance - 4-6 specific balancing recommendations
                5. Targeted Training & Development - 5-7 specific training suggestions
                6. Team Building & Collaboration - 4-6 collaboration strategies
                7. Risk Mitigation - 3-5 risks with mitigation strategies

                {format_instructions}"""),
                ("human", """
                COMPREHENSIVE DEPARTMENT ANALYSIS REQUEST:

                DEPARTMENT: {department_name}
                TOTAL EMPLOYEES: {total_employees}
                EMPLOYEE NAMES: {employee_names}
                POSITIONS: {positions}
                EMPLOYEES WITH GENIUS FACTOR REPORTS: {employees_with_reports}
                AVERAGE GENIUS FACTOR SCORE: {avg_score}

                DETAILED GENIUS FACTOR PROFILES:
                {genius_profiles}

                Please provide EXTENSIVE, DETAILED recommendations following the exact structure above.
                Each section should contain multiple specific, actionable items. Focus on practical strategies
                that can be implemented to enhance department efficiency and team performance.
                """)
            ])
            
            # Prepare genius profiles section
            genius_profiles_text = json.dumps(genius_profiles, indent=2) if genius_profiles else "No genius factor reports available. Provide recommendations based on department type and roles."
            
            # Format the prompt
            formatted_prompt = prompt.format_prompt(
                format_instructions=format_instructions,
                department_name=department_name,
                total_employees=len(employees),
                employee_names=", ".join(employee_names),
                positions=", ".join(positions),
                employees_with_reports=employees_with_reports,
                avg_score=f"{avg_score:.1f}" if avg_score else 'N/A',
                genius_profiles=genius_profiles_text
            )
            
            # Generate recommendations with GPT-4o
            response = await self.llm.ainvoke(formatted_prompt.to_string())
            
            # Parse the structured output
            parsed_recommendations = self.parser.parse(response.content)
            
            return {
                'structured_recommendations': parsed_recommendations.dict(),
                'summary': f"Comprehensive recommendations for {department_name} based on {len(genius_profiles)} genius factor profiles",
                'data_quality': {
                    'employees_with_data': employees_with_reports,
                    'total_employees': len(employees),
                    'coverage_percentage': round((employees_with_reports / len(employees)) * 100, 2) if employees else 0
                }
            }
            
        except Exception as e:
            # Enhanced fallback with detailed recommendations
            return await self.generate_detailed_fallback_recommendations(department_name, employees)

    async def generate_detailed_fallback_recommendations(
        self,
        department_name: str,
        employees: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate detailed fallback recommendations when LLM fails.
        """
        positions = list(set(emp['position'] for emp in employees if emp['position']))
        employee_count = len(employees)
        employee_names = [f"{emp['firstName']} {emp['lastName']}" for emp in employees if emp['firstName'] and emp['lastName']]
        
        # Department-specific detailed recommendations
        department_recommendations = {
            "IT": {
                "team_composition_overview": {
                    "overview": f"The IT department consists of {employee_count} employees: {', '.join(employee_names)}. Roles include {', '.join(positions)}. This technical department requires a balance of software development, infrastructure management, and technical support capabilities to effectively serve organizational technology needs."
                },
                "strengths_identified": {
                    "strengths": [
                        "Technical expertise in specific domains based on current roles",
                        "Potential for innovation and problem-solving in technology solutions",
                        "Ability to adapt to changing technology landscapes",
                        "Cross-functional collaboration capabilities with other departments",
                        "Technical documentation and knowledge sharing potential"
                    ]
                },
                "critical_gaps": {
                    "gaps": [
                        f"Limited team size of {employee_count} may restrict ability to handle multiple concurrent projects effectively",
                        "Potential lack of diverse technical specializations (e.g., only {', '.join(positions) if positions else 'limited roles'})",
                        "Possible knowledge silos with limited cross-training opportunities",
                        "Capacity constraints for supporting growing technology infrastructure needs"
                    ]
                },
                "recommendations_for_balance": {
                    "recommendations": [
                        "Hire additional technical staff with complementary skills (backend development, DevOps, cybersecurity)",
                        "Implement job rotation program to develop cross-functional expertise",
                        "Consider outsourcing specialized technical functions to augment internal capabilities",
                        "Develop clear career progression paths to retain and develop existing talent",
                        "Create technical specialization tracks to build depth in critical technology areas"
                    ]
                },
                "targeted_training_development": {
                    "training_recommendations": [
                        "Advanced programming languages and frameworks relevant to current tech stack",
                        "Cloud computing and infrastructure management certifications",
                        "Agile methodology and project management training",
                        "Cybersecurity best practices and threat mitigation strategies",
                        "Technical leadership and architecture design principles",
                        "DevOps practices and continuous integration/continuous deployment",
                        "Emerging technologies assessment and implementation training"
                    ]
                },
                "team_building_collaboration": {
                    "collaboration_recommendations": [
                        "Weekly technical knowledge sharing sessions and code reviews",
                        "Cross-functional project teams with business units to understand requirements",
                        "Regular hackathons or innovation challenges to foster creativity",
                        "Mentorship program pairing senior and junior technical staff",
                        "Collaboration tools implementation for better remote teamwork",
                        "Quarterly team offsites focused on technical strategy and bonding"
                    ]
                },
                "risk_mitigation": {
                    "risks": [
                        "Single points of failure for critical technical systems and knowledge",
                        "Technology debt accumulation without dedicated resources for maintenance",
                        "Skill gaps in emerging technologies affecting competitiveness",
                        "Burnout risk due to limited staffing and high workload demands"
                    ],
                    "mitigation_strategies": [
                        "Implement comprehensive documentation and knowledge management systems",
                        "Develop disaster recovery and business continuity plans for critical systems",
                        "Create cross-training matrix to ensure multiple staff can cover key functions",
                        "Establish workload management and prioritization processes",
                        "Regular technology stack assessments and modernization planning"
                    ]
                }
            }
        }
        
        # Get department-specific recommendations or use general ones
        if department_name in department_recommendations:
            recommendations = department_recommendations[department_name]
        else:
            # General detailed recommendations for unknown departments
            recommendations = {
                "team_composition_overview": {
                    "overview": f"The {department_name} department consists of {employee_count} employees: {', '.join(employee_names)}. Roles include {', '.join(positions)}. This department plays a critical role in organizational operations and requires strategic planning for optimal performance."
                },
                "strengths_identified": {
                    "strengths": [
                        f"Specialized expertise in {positions[0] if positions else 'respective fields'}",
                        "Established processes and institutional knowledge",
                        "Cross-functional collaboration experience",
                        "Problem-solving capabilities in domain-specific challenges",
                        "Adaptability to changing organizational needs"
                    ]
                },
                "critical_gaps": {
                    "gaps": [
                        f"Limited team size may restrict capacity for strategic initiatives",
                        "Potential lack of diverse skill sets and perspectives",
                        "Possible process inefficiencies without dedicated optimization resources",
                        "Knowledge transfer challenges with limited staffing"
                    ]
                },
                "recommendations_for_balance": {
                    "recommendations": [
                        "Conduct comprehensive skills assessment to identify capability gaps",
                        "Develop strategic hiring plan for complementary roles and skills",
                        "Implement job rotation and cross-training programs",
                        "Establish clear role definitions and responsibility matrices",
                        "Create talent development pipeline for future leadership needs"
                    ]
                },
                "targeted_training_development": {
                    "training_recommendations": [
                        "Domain-specific technical skills enhancement programs",
                        "Leadership and management development training",
                        "Process optimization and efficiency improvement methodologies",
                        "Stakeholder management and communication skills",
                        "Data analysis and decision-making frameworks",
                        "Change management and innovation adoption techniques",
                        "Professional certification programs in relevant fields"
                    ]
                },
                "team_building_collaboration": {
                    "collaboration_recommendations": [
                        "Regular team meetings with structured agenda and action items",
                        "Cross-departmental project collaboration initiatives",
                        "Professional development and learning circles",
                        "Team building activities focused on trust and communication",
                        "Knowledge sharing platforms and best practice documentation",
                        "Mentorship and coaching programs for skill development"
                    ]
                },
                "risk_mitigation": {
                    "risks": [
                        "Workload concentration and potential burnout risks",
                        "Knowledge silos and single points of failure",
                        "Skill obsolescence without continuous learning investment",
                        "Process inefficiencies affecting departmental performance"
                    ],
                    "mitigation_strategies": [
                        "Implement workload assessment and redistribution processes",
                        "Develop comprehensive documentation and standard operating procedures",
                        "Establish continuous learning and development programs",
                        "Create performance metrics and regular review processes",
                        "Build external network for knowledge sharing and best practices"
                    ]
                }
            }
        
        return {
            'structured_recommendations': recommendations,
            'summary': f"Detailed recommendations for {department_name} department",
            'data_quality': {
                'employees_with_data': 0,
                'total_employees': employee_count,
                'coverage_percentage': 0.0
            }
        }

    async def get_department_statistics(
        self, 
        hr_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get overall department statistics without detailed employee data.
        """
        # Build the where clause
        where_clause = {}
        
        if hr_id:
            where_clause['hrId'] = hr_id
        
        # Get all users with the filters
        users = await self.db.user.find_many(
            where=where_clause
        )
        
        # Aggregate department counts
        department_counts = {}
        total_employees = 0
        
        for user in users:
            if user.department and len(user.department) > 0:
                # Get the current department (last item in array)
                current_department = user.department[-1]
                
                if current_department not in department_counts:
                    department_counts[current_department] = 0
                
                department_counts[current_department] += 1
                total_employees += 1
        
        # Convert to list format for response
        departments = [
            {
                'department': dept,
                'employee_count': count
            }
            for dept, count in department_counts.items()
        ]
        
        # Sort by department name
        departments.sort(key=lambda x: x['department'])
        
        return {
            'total_departments': len(departments),
            'total_employees': total_employees,
            'departments': departments
        }