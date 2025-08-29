from fastapi import APIRouter, UploadFile, File, Form
# from services.jobs_service import parse_and_save_jobs
from services.job_service import parse_and_save_jobs

router = APIRouter(prefix="/jobs", tags=["jobs"])

@router.post("/upload")
async def upload_jobs(file: UploadFile = File(...), recruiter_id: str = Form(...)):
    """
    Upload a file containing jobs, parse and save into database.
    """
    if not recruiter_id:
        return {"error": "recruiter_id is required"}

    try:
        inserted_jobs = await parse_and_save_jobs(file, recruiter_id)
        return {
            "inserted": len(inserted_jobs),
            "jobs": [{"id": job.id, "title": job.title} for job in inserted_jobs]
        }
    except ValueError as ve:
        return {"error": str(ve)}
    except Exception as e:
        return {"error": "Failed to process file", "details": str(e)}
