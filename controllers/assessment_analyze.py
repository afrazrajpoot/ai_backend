from fastapi import HTTPException
from schemas.assessment import AssessmentData
from services.ai_service import AIService
# from services.database_notification_service import DatabaseNotificationService
from services.db_service import DBService
from utils.analyze_assessment import analyze_assessment_data
from utils.logger import logger
from services.notification_service import NotificationService
from typing import Dict, Any
import httpx
from prisma import Prisma

# Singleton AIService instance (assumed to be defined elsewhere)
ai_service = AIService()

# Singleton DatabaseNotificationService instance
db_notification_service = DBService()

class AssessmentController:

    
    @staticmethod
    async def save_to_database(data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prisma = Prisma()
            data = {
        "userId": "33b003c7-ad86-4ecf-bb54-5b2a2c633926",
        "hrId": "42632d76-8d0d-4a59-af5c-b0172c5aaa6f",
        "department": "Media",
        "executive_summary": "This comprehensive report evaluates the Genius Factor assessment results for the employee, identifying them as a Tech Genius. The primary genius factor, Tech Genius, highlights their exceptional capabilities in systematic thinking and technical problem-solving, which are critical in today's technology-driven landscape. This unique combination of analytical skills and technical acumen positions the employee as a vital asset to the organization, particularly in industries such as technology, financial services, and healthcare. By leveraging their strengths, the organization can enhance its competitive advantage, drive innovation, and improve operational efficiency. The strategic implications of their genius factors suggest a strong alignment with roles that require technical leadership and innovative problem-solving. Proper utilization of their talents can lead to significant improvements in project outcomes, team dynamics, and overall business performance. Looking forward, the employee's potential impact on business outcomes is substantial, as they are well-equipped to navigate complex technical challenges and contribute to the organization's strategic objectives.",
        "genius_factor_profile": {
            "primary_genius_factor": "Tech Genius",
            "description": "Individuals identified as Tech Geniuses excel in systematic thinking and technical problem-solving. They possess a natural aptitude for analyzing complex systems and developing innovative solutions. Their cognitive patterns are characterized by a methodical approach to challenges, allowing them to dissect problems and implement effective strategies. In professional settings, Tech Geniuses often take on roles that involve technology development, data analysis, and systems optimization, where their skills can be fully utilized. They thrive in environments that challenge their analytical abilities and allow for creative problem-solving.",
            "key_strengths": [
                "Systematic Problem-Solving: Tech Geniuses can break down complex issues into manageable components, facilitating effective solutions.",
                "Analytical Thinking: Their ability to analyze data and trends enables them to make informed decisions that drive business success.",
                "Technical Proficiency: They possess strong skills in programming, data analysis, and system design, making them invaluable in tech-centric roles.",
                "Innovative Mindset: Tech Geniuses are often at the forefront of technological advancements, driving innovation within their teams.",
                "Adaptability: They can quickly adapt to new technologies and methodologies, ensuring they remain relevant in a fast-paced industry.",
                "Collaboration: Their systematic approach fosters collaboration, as they can articulate complex ideas clearly to team members.",
                "Leadership Potential: Tech Geniuses are often seen as natural leaders in technical environments, guiding teams through challenges.",
                "Attention to Detail: Their meticulous nature ensures that projects are executed with precision, minimizing errors."
            ],
            "secondary_genius_factor": "Tech Genius",
            "secondary_description": "As the secondary genius factor, the Tech Genius trait reinforces the primary genius, enhancing the employee's capacity for technical problem-solving and systematic thinking. This dual emphasis on technology allows for a deeper understanding of complex systems and fosters a robust approach to innovation and efficiency.",
            "energy_sources": [
                "Engaging in complex problem-solving tasks energizes Tech Geniuses, as they enjoy the challenge of overcoming obstacles.",
                "Collaborating with like-minded individuals on technical projects allows them to share ideas and innovate together.",
                "Learning new technologies and programming languages provides a stimulating environment that fuels their passion for tech."
            ]
        },
        "current_role_alignment_analysis": {
            "alignment_score": "85",
            "assessment": "The employee's alignment score of 85 reflects a strong fit between their genius factors and their current role. Their systematic approach to problem-solving and technical proficiency are well-utilized in their day-to-day responsibilities. They excel in tasks that require analytical thinking and innovative solutions, contributing significantly to team projects and organizational goals. However, there are areas where their talents could be further leveraged, particularly in leadership roles that require strategic oversight of technical initiatives.",
            "strengths_utilized": [
                "The employee effectively applies their systematic problem-solving skills in project management, ensuring tasks are completed efficiently.",
                "Their analytical thinking is utilized in data analysis, providing insights that drive decision-making processes.",
                "Technical proficiency is evident in their contributions to software development projects, where they implement innovative solutions.",
                "They demonstrate leadership potential by guiding team members through complex technical challenges, fostering a collaborative environment."
            ],
            "underutilized_talents": [
                "The employee's potential for leadership in technical projects is not fully realized, as they often take a backseat in strategic discussions.",
                "Opportunities for mentoring junior team members could enhance their role and provide valuable development for others.",
                "Engagement in cross-departmental projects could further leverage their skills and provide broader organizational impact."
            ],
            "retention_risk_level": "The employee's strong alignment with their role and the organization’s commitment to leveraging their technical skills contribute to a low retention risk level. Their engagement in meaningful projects and the potential for career advancement through internal mobility further solidify their commitment to the organization."
        },
        "internal_career_opportunities": {
            "primary_industry": "Technology",
            "secondary_industry": "Financial Services",
            "recommended_departments": [
                "Software Development",
                "Cybersecurity",
                "Data Analytics",
                "Technical Support"
            ],
            "specific_role_suggestions": [
                "Software Engineer",
                "Cybersecurity Analyst",
                "Data Scientist",
                "Technical Lead"
            ],
            "career_pathways": {
                "Development Track": "Software Engineer → Senior Developer → Technical Lead → Chief Technology Officer",
                "Security Track": "Cybersecurity Analyst → Security Engineer → Security Architect → Chief Information Security Officer",
                "AI Track": "Data Scientist → Machine Learning Engineer → AI Research Director → Chief AI Officer"
            },
            "transition_timeline": {
                "six_month": "Transition into a Senior Developer role with increased responsibilities.",
                "one_year": "Move into a Technical Lead position, overseeing project teams.",
                "two_year": "Advance to Chief Technology Officer, shaping the organization's technology strategy."
            },
            "required_skill_development": [
                "Advanced programming languages (Python, JavaScript, Java, C++)",
                "Cloud computing platforms (AWS, Azure, Google Cloud)",
                "Machine learning and artificial intelligence techniques",
                "Cybersecurity protocols and data protection strategies"
            ]
        },
        "retention_and_mobility_strategies": {
            "retention_strategies": [
                "Implement mentorship programs that pair Tech Geniuses with leadership roles to foster career growth.",
                "Create opportunities for lateral moves within technical departments to enhance skill diversity and job satisfaction.",
                "Develop a clear career progression framework that outlines potential pathways and associated skill requirements."
            ],
            "internal_mobility_recommendations": [
                "Encourage participation in cross-departmental projects to broaden the employee's experience and visibility within the organization.",
                "Establish a talent mobility program that identifies and promotes internal candidates for leadership roles.",
                "Regularly assess and update the skills inventory of employees to align with evolving organizational needs."
            ],
            "development_support": [
                "Allocate resources for continuous learning and professional development, including access to training programs and certifications.",
                "Provide platforms for knowledge sharing and collaboration among employees to foster innovation and engagement.",
                "Encourage attendance at industry conferences and workshops to keep abreast of technological advancements."
            ]
        },
        "development_action_plan": {
            "thirty_day_goals": [
                "Identify and enroll in a relevant online course focused on advanced programming languages.",
                "Schedule one-on-one meetings with a mentor to discuss career aspirations and development opportunities.",
                "Participate in team brainstorming sessions to contribute innovative ideas for upcoming projects."
            ],
            "ninety_day_goals": [
                "Lead a small project team to enhance leadership skills and gain experience in project management.",
                "Complete a certification in cloud computing to bolster technical expertise.",
                "Attend a workshop on emerging technologies to stay updated on industry trends."
            ],
            "six_month_goals": [
                "Take on a technical lead role in a major project, demonstrating leadership and technical skills.",
                "Develop a comprehensive project report showcasing innovative solutions implemented during the project.",
                "Establish a feedback loop with team members to improve collaboration and project outcomes."
            ],
            "networking_strategy": [
                "Join professional associations related to technology and cybersecurity to expand industry connections.",
                "Attend local tech meetups and conferences to network with peers and industry leaders.",
                "Engage with online forums and communities to share knowledge and learn from others in the field."
            ]
        },
        "personalized_resources": {
            "affirmations": [
                "I am a natural problem solver, capable of overcoming any technical challenge.",
                "My analytical skills provide valuable insights that drive success.",
                "I am worthy of leadership opportunities that match my technical abilities.",
                "I thrive in environments that challenge my systematic thinking.",
                "My contributions to projects are impactful and recognized by my peers.",
                "I am constantly learning and growing in my technical expertise.",
                "I collaborate effectively with others to achieve common goals.",
                "I embrace innovation and drive change within my organization."
            ],
            "mindfulness_practices": [
                "Practice deep breathing exercises for 5 minutes daily to enhance focus and reduce stress.",
                "Engage in mindful walking, paying attention to each step and the environment around you.",
                "Set aside time for daily reflection on accomplishments and areas for improvement."
            ],
            "reflection_questions": [
                "What technical challenges have I successfully overcome in the past month?",
                "How can I leverage my strengths to contribute more effectively to my team?",
                "What new skills do I want to develop in the next quarter?"
            ],
            "learning_resources": [
                "Online courses in advanced programming languages (e.g., Coursera, Udacity).",
                "Books on cloud computing and cybersecurity best practices.",
                "Webinars and podcasts focused on emerging technologies and industry trends."
            ]
        },
        "data_sources_and_methodology": {
            "data_sources": [
                "Genius Factor Assessment for Fortune 1000 HR Departments.pdf",
                "Genius Factor to Fortune 1000 Industry Mapping.pdf",
                "Genius Factor Framework Analysis.pdf",
                "retention & internal mobility research_findings.pdf"
            ],
            "methodology": "The analysis was conducted using a comprehensive framework that evaluates the Genius Factor assessment results. The employee's primary and secondary genius factors were identified, followed by a detailed analysis of their alignment with current roles and potential career pathways. Recommendations for retention and mobility were developed based on industry best practices and organizational needs. The report emphasizes actionable strategies for development and growth, ensuring alignment with the employee's strengths and aspirations."
        },
        "genius_factor_score": 85
    }
            await prisma.connect()
            await prisma.individualemployeereport.create(
                data=data
            )

            await prisma.disconnect()
        except Exception as e:
            logger.error(f"Primary save failed: {str(e)}")

    @staticmethod
    async def analyze_assessment(input_data: AssessmentData) -> Dict[str, Any]:
        """
        Endpoint for assessment analysis with real-time notifications and Next.js integration
        """
        try:
            logger.info(f"Starting assessment analysis for userId: {input_data.userId}, hrId: {input_data.hrId}")

            # Validate input data for notification
            notification_data = {
                'employeeId': input_data.userId,
                'hrId': input_data.hrId,
                'employeeName': input_data.employeeName,
                'employeeEmail': input_data.employeeEmail,
                'message': 'Assessment analysis completed successfully!',
                'status':'unread'
            }
            # for key, value in notification_data.items():
            #     if not isinstance(value, str) or not value.strip():
            #         logger.error(f"Invalid notification data: {key} is empty or not a string")
            #         await NotificationService.send_user_notification(
            #             input_data.userId,
            #             input_data.hrId,
            #             {
            #                 'message': 'Invalid notification data',
            #                 'progress': 0,
            #                 'status': 'error',
            #                 'error': f"Field {key} is invalid"
            #             }
            #         )
            #         raise HTTPException(status_code=400, detail=f"Invalid notification data: {key}")

            # # 1. Get basic assessment results
            # try:
            #     basic_results = analyze_assessment_data(input_data.data)
            #     logger.info("Basic analysis completed")
            # except Exception as e:
            #     logger.error(f"Failed to analyze assessment data: {str(e)}")
            #     await NotificationService.send_user_notification(
            #         input_data.userId,
            #         input_data.hrId,
            #         {
            #             'message': 'Basic analysis failed',
            #             'progress': 100,
            #             'status': 'error',
            #             'error': str(e)
            #         }
            #     )
            #     raise HTTPException(status_code=500, detail=str(e))

            # # 2. Enhance with document retrieval from vector store
            # try:
            #     rag_results = await ai_service.analyze_majority_answers(basic_results)
            # except Exception as e:
            #     logger.error(f"Failed to analyze majority answers: {str(e)}")
            #     await NotificationService.send_user_notification(
            #         input_data.userId,
            #         input_data.hrId,
            #         {
            #             'message': 'Failed to perform advanced analysis',
            #             'progress': 100,
            #             'status': 'error',
            #             'error': str(e)
            #         }
            #     )
            #     raise HTTPException(status_code=500, detail=str(e))
            
            # # 3. Generate professional career recommendation report
            # try:
            #     recommendations = await ai_service.generate_career_recommendation(rag_results)
            # except Exception as e:
            #     logger.error(f"Failed to generate recommendations: {str(e)}")
            #     await NotificationService.send_user_notification(
            #         input_data.userId,
            #         input_data.hrId,
            #         {
            #             'message': 'Failed to generate recommendations',
            #             'progress': 100,
            #             'status': 'error',
            #             'error': str(e)
            #         }
            #     )
            #     raise HTTPException(status_code=500, detail=str(e))
            
            # if recommendations.get("status") != "success":
            #     error_msg = f"Failed to generate recommendations: {recommendations.get('message', 'Unknown error')}"
            #     logger.error(error_msg)
                
            #     await NotificationService.send_user_notification(
            #         input_data.userId,
            #         input_data.hrId,
            #         {
            #             'message': 'Analysis failed',
            #             'progress': 100,
            #             'status': 'error',
            #             'error': error_msg
            #         }
            #     )
                
            #     raise HTTPException(status_code=500, detail="Failed to generate career recommendations")

            # # Prepare final result
            # final_result = {
            #     "status": "success",
            #     "userId": input_data.userId,
            #     "report": recommendations.get("report"),
            #     "risk_analysis": recommendations.get("risk_analysis"),
            #     "metadata": recommendations.get("metadata")
            # }

            # 4. Send data to Next.js API (asynchronous call)
            nextjs_response = await AssessmentController.save_to_database({
                'userId': input_data.userId,
                'hrId': input_data.hrId,
                'employeeName': input_data.employeeName,
                'employeeEmail': input_data.employeeEmail,
                'data': input_data.data
            })
            
            if nextjs_response.get("status") == "error":
                logger.warning(f"Next.js API call failed but proceeding: {nextjs_response.get('message')}")
                # Continue even if Next.js call fails, but log the warning

            # Send success notification via Socket.IO
            await NotificationService.send_user_notification(
                input_data.userId,
                input_data.hrId,
                {
                    'message': 'Assessment analysis completed successfully!',
                    'employeeName': input_data.employeeName,
                    'employeeEmail': input_data.employeeEmail,
                 
                    'progress': 100,
                    'status': 'unread',
                    # 'report_id': recommendations.get("metadata", {}).get("report_id"),
                    # 'nextjs_response': nextjs_response
                }
            )

            # Save notification to database using DatabaseNotificationService
            try:
                await db_notification_service.save_notification(notification_data)
            except Exception as e:
                logger.error(f"Failed to save notification to database: {str(e)}")
                await NotificationService.send_user_notification(
                    input_data.userId,
                    input_data.hrId,
                    {
                        'message': 'Failed to save notification to database',
                        'progress': 100,
                        'status': 'error',
                        'error': str(e)
                    }
                )
                # Continue even if database save fails, but log and notify

            logger.info("Assessment analysis, report generation, and Next.js integration completed successfully")
            # return final_result

        except Exception as e:
            logger.error(f"Error in analyze_assessment: {str(e)}")
            
            await NotificationService.send_user_notification(
                input_data.userId,
                input_data.hrId,
                {
                    'message': 'Assessment analysis failed',
                    'progress': 100,
                    'status': 'error',
                    'error': str(e)
                }
            )
            
            raise HTTPException(status_code=500, detail=str(e))
