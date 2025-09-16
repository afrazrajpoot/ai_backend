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
    # print("[DEBUG] Number of rows:", len(rows))
    # print("[DEBUG] Columns detected:", rows[0].keys())
    # print("[DEBUG] First 5 rows:")
    for r in rows[:5]:
        print(r)

    # --- Insert into Prisma ---
    await db.connect()
    inserted_jobs = []

    for idx, row in enumerate(rows):
        title = str(row.get("title", "")).strip()
        if not title:
            # print(f"[WARNING] Row {idx+1} missing title. Skipping.")
            continue

        try:
            job = await db.job.create(
                data={
                    "title": title,
                    "description": str(row.get("description", "")),
                    "location": str(row.get("location", "")) if row.get("location") else None,
                    "salary": int(row.get("salary", 0)) if row.get("salary") else None,
                    "type": str(row.get("type", "FULL_TIME")),
                    "status": "OPEN",
                    "recruiterId": recruiter_id,
                }
            )
            inserted_jobs.append(job)
            # print(f"[INFO] Job {idx+1} inserted: {job.title}")
        except Exception as e:
            raise ValueError(f"Failed to insert row {idx+1}: {e}")
            # print(f"[ERROR] Failed to insert row {idx+1}: {e}")

    await db.disconnect()
    return inserted_jobs
