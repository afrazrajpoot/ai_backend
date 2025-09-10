# app/services/dashboard_service.py
from prisma import Prisma
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate


async def generate_ai_recommendation(employee_id: str):
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

        # 2. Fetch reports - order by creation date to get latest
        reports = await prisma.individualemployeereport.find_many(
            where={"userId": user.id},
            order={"createdAt": "desc"}  # Get latest reports first
        )
        completed_assessments = len(reports)

        # 3. Dashboard stats
        assessment_progress = f"{completed_assessments * 100}%"  # dummy
        avg_genius_score = (
            sum(r.geniusFactorScore for r in reports) / completed_assessments
            if completed_assessments > 0 else None
        )

        # 4. Delete existing career recommendation if it exists
        existing_rec = await prisma.aicareerrecommendation.find_first(
            where={"employeeId": employee_id},
        )

        if existing_rec:
            # Delete the existing recommendation
            await prisma.aicareerrecommendation.delete(
                where={"id": existing_rec.id}
            )

        # 5. Generate new recommendation with GPT-4o using the latest report
        llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

        # Get the latest report for recommendation generation
        latest_report = reports[0] if reports else None
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", "You are an expert HR career advisor. Write highly professional, extensive, and tailored career recommendations."),
            ("human", "Employee: {name}\nDepartment: {department}\nLatest Report Summary: {report_summary}\nGenius Factor Score: {latest_score}\nAverage Genius Factor Score: {avg_score}\n\nGenerate a professional career recommendation based on the most recent assessment.")
        ])

        prompt = prompt_template.format(
            name=f"{user.firstName} {user.lastName}",
            department=user.department,
            report_summary=latest_report.executiveSummary if latest_report else "No reports available",
            latest_score=latest_report.geniusFactorScore if latest_report else "N/A",
            avg_score=avg_genius_score
        )

        response = await llm.ainvoke(prompt)
        career_recommendation = response.content

        # 6. Save new recommendation in DB
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

# app/services/dashboard_service.py



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

        # 2. Fetch reports - order by creation date to get latest
        reports = await prisma.individualemployeereport.find_many(
            where={"userId": user.id},
            order={"createdAt": "desc"}  # Get latest reports first
        )
        completed_assessments = len(reports)

        # 3. Dashboard stats
        assessment_progress = f"{completed_assessments * 100}%"  # dummy
        avg_genius_score = (
            sum(r.geniusFactorScore for r in reports) / completed_assessments
            if completed_assessments > 0 else None
        )

        # 4. Get career recommendation from DB only (no generation)
        existing_rec = await prisma.aicareerrecommendation.find_first(
            where={"employeeId": employee_id},
        )

        career_recommendation = existing_rec.careerRecommendation if existing_rec else None

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