from fastapi import APIRouter, UploadFile, File, Form
# from services.jobs_service import parse_and_save_jobs
from services.employee_services.employee_parse_service import parse_and_save_employees

router = APIRouter(prefix="/employees", tags=["employees"])
@router.post("/upload")
async def upload_employees(file: UploadFile = File(...), hr_id: str = Form(...)):
    try:
        result = await parse_and_save_employees(file, hr_id)
        return result
    except Exception as e:
        return {"error": "Failed to process file", "details": str(e)}
