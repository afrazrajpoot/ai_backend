import io
import pandas as pd
import pyexcel_ods3
from prisma import Prisma

db = Prisma()

async def parse_and_save_jobs(file, recruiter_id: str):
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
        import pyexcel
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

    # --- Debug ---
    for r in rows[:5]:
        print(r)

    # --- Insert into Prisma ---
    await db.connect()
    inserted_jobs = []

    for idx, row in enumerate(rows):
        title = str(row.get("title", "")).strip()
        if not title:
            continue

        try:
            # Handle skills: extract as list if present, otherwise omit (for null)
            skills_raw = row.get("skills")
            skills_data = None
            if skills_raw:
                skills_list = [s.strip() for s in str(skills_raw).split(",") if s.strip()]
                if skills_list:
                    skills_data = skills_list  # Plain list; Prisma serializes to JSON

            # Base data without optional fields
            data = {
                "title": title,
                "description": str(row.get("description", "")),
                "type": str(row.get("type", "FULL_TIME")),
                "status": "OPEN",
                # Required relation: Use 'connect' instead of direct FK
                "recruiter": {
                    "connect": {
                        "id": recruiter_id  # Assumes recruiter_id is a valid User ID
                    }
                },
            }

            # Add optional scalars only if set
            location = row.get("location")
            if location:
                data["location"] = str(location)

            salary = row.get("salary")
            if salary:
                data["salary"] = int(salary)

            # Add skills only if present (omit for null)
            if skills_data is not None:
                data["skills"] = skills_data

            job = await db.job.create(data=data)
            inserted_jobs.append(job)

        except Exception as e:
            raise ValueError(f"Failed to insert row {idx+1}: {e}")

    await db.disconnect()
    return inserted_jobs