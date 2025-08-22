


# Updated routes/parse.py
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from controllers.parse_companies import ParseCompaniesController

router = APIRouter(prefix="/parse", tags=["Parse"])

# Create controller instance
controller = ParseCompaniesController()

@router.post("/companies")
async def parse_excel_files(files: List[UploadFile] = File(...)):
    results = await controller.parse_files(files)
    return {"status": "success", "data": results}

@router.get("/companies/{company_id}")
async def get_company(company_id: str):
    company = await controller.get_company(company_id)
    if not company:
        raise HTTPException(status_code=404, detail="Company not found")
    return {"status": "success", "data": company}

@router.get("/companies")
async def get_all_companies(skip: int = 0, take: int = 100):
    companies = await controller.get_all_companies(skip, take)
    return {"status": "success", "data": companies, "pagination": {"skip": skip, "take": take}}