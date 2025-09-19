import io
import pandas as pd
import pyexcel_ods3
from prisma import Prisma
import bcrypt
import json  # Add json import for serialization

db = Prisma()

async def parse_and_save_employees(file, hr_id: str):
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

        header = data[0]
        for row in data[1:]:
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
    inserted_employees = []

    for idx, row in enumerate(rows):
        email = str(row.get("email", "")).strip().lower()
        if not email:
        

            continue

        existing_user = await db.user.find_first(where={"email": email})
        if existing_user:
       
            continue

        try:
            # Hash default password
            default_password = "Pa$$w0rd!"
            hashed_password = bcrypt.hashpw(default_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

            # --- Create Employee ---
            employee = await db.user.create(
                data={
                    "firstName": str(row.get("firstName", "")).strip(),
                    "lastName": str(row.get("lastName", "")).strip(),
                    "email": email,
                    "phoneNumber": str(row.get("phoneNumber", "")).strip(),
                    "position": [str(row.get("position", "")).strip()],
                    "salary": str(row.get("salary", "")) if row.get("salary") else None,
                    "department": [str(row.get("department", "")).strip()],
                    "password": hashed_password,
                    "hrId": hr_id,
                    "role": "Employee"
                }
            )

            # --- Create Department entry linked with userId ---
            dept_name = str(row.get("department", "")).strip()
            position = str(row.get("position", "")).strip()
            if dept_name and position:
                try:
                    # Convert department name to a JSON array
               
                    positions = [pos.strip() for pos in position.split(",") if pos.strip()]
                    positions_json = json.dumps(positions)  # Serialize positions to JSON string
                    userIdlIST = [employee.id]
                    userIdJson = json.dumps(userIdlIST)
                  
                    await db.department.create(
                        data={
                            "name": dept_name,  # Pass JSON string
                            "userId": employee.id,  # Link to the created user
                            "hrId": hr_id,
                        }
                    )
               
                except Exception as e:
                    # Handle the exception
                    pass
              

            # --- Append to return list ---
            inserted_employees.append({
                "id": employee.id,
                "firstName": employee.firstName,
                "lastName": employee.lastName,
                "email": employee.email,
                "position": employee.position
            })
       
        except Exception as e:
         
            raise ValueError(f"Failed to insert row {idx+1}: {e}")

    await db.disconnect()

    return {
        "inserted": len(inserted_employees),
        "employees": inserted_employees
    }