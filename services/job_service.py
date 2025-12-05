import io
import pandas as pd
import pyexcel_ods3
import pyexcel
from prisma import Prisma
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

db = Prisma()

async def parse_and_save_jobs(file: UploadFile, recruiter_id: str):
    contents = await file.read()
    filename, ext = file.filename.lower().rsplit(".", 1)
    rows = []

    # --- CSV ---
    if ext == "csv":
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        rows = df.to_dict(orient="records")

    # --- Excel ---
    elif ext in ["xls", "xlsx"]:
        df = pd.read_excel(io.BytesIO(contents), engine="openpyxl")
        rows = df.to_dict(orient="records")

    # --- ODS ---
    elif ext == "ods":
        sheet = pyexcel_ods3.get_data(io.BytesIO(contents))
        sheet_name = list(sheet.keys())[0]
        data = sheet[sheet_name]

        if not data or len(data) < 2:
            raise ValueError("ODS file is empty or missing data")

        # First row is headers
        header = data[0]
        for row in data[1:]:
            # Pad row if some cells are missing
            row += [""] * (len(header) - len(row))
            rows.append(dict(zip(header, row)))

    # --- ODT ---
    elif ext == "odt":
        sheet = pyexcel.get_sheet(file_type="odt", file_content=contents)
        header = sheet.colnames
        if not header or all([h is None or h == "" for h in header]):
            first_row = sheet.row[0]
            num_cols = len(first_row)
            header = [f"col{i+1}" for i in range(num_cols)]
        for row in sheet.to_array()[1:]:
            rows.append(dict(zip(header, row)))

    else:
        raise ValueError("Only CSV, Excel, ODT, or ODS files are supported")

    if not rows:
        raise ValueError("File has no rows")

    # --- Insert into Prisma ---
    await db.connect()
    inserted_jobs = []

    for idx, row in enumerate(rows):
        title = str(row.get("title", "")).strip()
        if not title:
            continue

        try:
            # Handle skills
            skills_raw = row.get("skills")
            skills_data = None
            if skills_raw:
                skills_list = [s.strip() for s in str(skills_raw).split(",") if s.strip()]
                if skills_list:
                    skills_data = skills_list

            # Base data
            data = {
                "title": title,
                "description": str(row.get("description", "")),
                "type": str(row.get("type", "FULL_TIME")),
                "status": "OPEN",
                # FIX: Use recruiterId directly instead of nested connect
                "recruiterId": recruiter_id,
            }

            # Add optional fields
            location = row.get("location")
            if location:
                data["location"] = str(location)

            salary = row.get("salary")
            if salary:
                try:
                    data["salary"] = int(salary)
                except:
                    pass

            if skills_data is not None:
                data["skills"] = skills_data

            job = await db.job.create(data=data)
            inserted_jobs.append(job)

        except Exception as e:
            # FIX: Check if it's a foreign key constraint error
            if "Foreign key constraint failed" in str(e) or "not found" in str(e):
                raise HTTPException(
                    status_code=404,
                    detail=f"Recruiter with ID {recruiter_id} not found"
                )
            raise HTTPException(
                status_code=400,
                detail=f"Failed to insert row {idx+1}: {str(e)}"
            )

    await db.disconnect()
    return inserted_jobs

router = APIRouter(prefix="/jobs", tags=["jobs"])

@router.post("/upload")
async def upload_jobs(file: UploadFile = File(...), recruiter_id: str = Form(...)):
    """
    Upload a file containing jobs, parse and save into database.
    """
    try:
        inserted_jobs = await parse_and_save_jobs(file, recruiter_id)
        return {
            "inserted": len(inserted_jobs),
            "jobs": [{"id": job.id, "title": job.title} for job in inserted_jobs]
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )