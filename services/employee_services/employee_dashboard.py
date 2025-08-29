# app/services/dashboard_service.py
from prisma import Prisma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


async def get_dashboard_service(employee_id: str):
    prisma = Prisma()
    await prisma.connect()

    try:
        # 1. Get user
        user = await prisma.user.find_unique(
            where={"id": employee_id},
            include={"employee": True}
        )
        if not user:
            return None, "User not found"

        # 2. Fetch reports
        reports = await prisma.individualemployeereport.find_many(
            where={"userId": user.id}
        )
        completed_assessments = len(reports)

        # 3. Dashboard stats
        assessment_progress = f"{completed_assessments * 100}%"  # dummy
        avg_genius_score = (
            sum(r.geniusFactorScore for r in reports) / completed_assessments
            if completed_assessments > 0 else None
        )

        # 4. Check career recommendation
        existing_rec = await prisma.aicareerrecommendation.find_first(
            where={"employeeId": employee_id}
        )

        if existing_rec:
            career_recommendation = existing_rec.careerRecommendation
        else:
            # Generate with GPT-4o
            llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

            prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are an expert HR career advisor. Write highly professional, extensive, and tailored career recommendations."),
                ("human", "Employee: {name}\nDepartment: {department}\nReports: {reports}\nAverage Genius Factor Score: {avg_score}\n\nGenerate a professional career recommendation.")
            ])

            prompt = prompt_template.format(
                name=f"{user.firstName} {user.lastName}",
                department=user.department,
                reports=[r.executiveSummary for r in reports],
                avg_score=avg_genius_score
            )

            response = await llm.ainvoke(prompt)
            career_recommendation = response.content

            # Save in DB
            await prisma.aicareerrecommendation.create(
                data={
                    "employeeId": employee_id,
                    "careerRecommendation": career_recommendation
                }
            )

        return {
            "employeeId": user.employeeId,
            "name": f"{user.firstName} {user.lastName}",
            "department": user.department,
            "assessmentProgress": assessment_progress,
            "completedAssessments": completed_assessments,
            "averageGeniusFactorScore": avg_genius_score,
            "careerRecommendation": career_recommendation
        }, None

    finally:
        await prisma.disconnect()
