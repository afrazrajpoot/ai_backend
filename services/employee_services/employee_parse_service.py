import io
import pandas as pd
import pyexcel_ods3
from prisma import Prisma
import bcrypt
import json  # Add json import for serialization
from datetime import datetime

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

    # First, check if HR user exists and get quota
    hr_user = await db.user.find_first(where={"id": hr_id})
    if not hr_user:
        await db.disconnect()
        raise ValueError(f"HR user with ID {hr_id} not found")
    
    is_paid = hr_user.paid if hasattr(hr_user, 'paid') else False
    hr_quota = hr_user.quota if hasattr(hr_user, 'quota') and hr_user.quota is not None else 0.0
    
    # Check initial quota
    if hr_quota <= 0:
        await db.disconnect()
        raise ValueError(f"HR user has no quota available. Current quota: {hr_quota}")
    
    # Count existing employees for this HR
    existing_employee_count = await db.user.count(
        where={
            "hrId": hr_id,
            "role": "Employee"
        }
    )
    
    available_quota = hr_quota - existing_employee_count
    if available_quota <= 0:
        await db.disconnect()
        raise ValueError(f"No quota available. HR user quota: {hr_quota}, Existing employees: {existing_employee_count}")
    
    print(f"HR Quota: {hr_quota}, Existing Employees: {existing_employee_count}, Available Quota: {available_quota}")

    # Get current date and time for emailVerified field
    current_datetime = datetime.now()
    
    # Track how many employees we can actually save
    employees_to_save = min(len(rows), int(available_quota))
    employees_saved = 0
    skipped_due_to_quota = 0
    duplicate_emails = 0

    for idx, row in enumerate(rows):
        # Check if we've reached quota limit
        if employees_saved >= available_quota:
            skipped_due_to_quota = len(rows) - idx
            print(f"Quota reached. Skipping remaining {skipped_due_to_quota} employees.")
            break
            
        email = str(row.get("email", "")).strip().lower()
        if not email:
            continue

        existing_user = await db.user.find_first(where={"email": email})
        if existing_user:
            print(f"Skipping duplicate email: {email}")
            duplicate_emails += 1
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
                    "role": "Employee",
                    "paid": is_paid,
                    "emailVerified": current_datetime
                }
            )
            
            employees_saved += 1

            # --- Create Department entry linked with userId ---
            dept_name = str(row.get("department", "")).strip()
            position = str(row.get("position", "")).strip()
            if dept_name and position:
                try:
                    # Check if department already exists
                    existing_department = await db.department.find_first(
                        where={
                            "name": dept_name,
                            "hrId": hr_id
                        }
                    )
                    
                    # Only create department if it doesn't exist
                    if not existing_department:
                        positions = [pos.strip() for pos in position.split(",") if pos.strip()]
                        positions_json = json.dumps(positions)
                        userIdlIST = [employee.id]
                        userIdJson = json.dumps(userIdlIST)
                      
                        await db.department.create(
                            data={
                                "name": dept_name,
                                "userId": employee.id,
                                "hrId": hr_id,
                            }
                        )
                except Exception as e:
                    print(f"Error creating department: {e}")
                    pass

            # --- Append to return list ---
            inserted_employees.append({
                "id": employee.id,
                "firstName": employee.firstName,
                "lastName": employee.lastName,
                "email": employee.email,
                "position": employee.position,
                "paid": employee.paid,
                "emailVerified": employee.emailVerified.isoformat() if employee.emailVerified else None
            })
            print(f"Saved employee {employees_saved}/{employees_to_save}: {email}")
       
        except Exception as e:
            print(f"Error saving employee {idx+1}: {e}")
            # Continue with next employee instead of stopping completely
            continue

    # Calculate new employee count after import
    new_employee_count = await db.user.count(
        where={
            "hrId": hr_id,
            "role": "Employee"
        }
    )
    
    # Calculate remaining quota
    remaining_quota = hr_quota - new_employee_count
    
    # ✅ IMPORTANT: Update HR user's quota in the database
    if employees_saved > 0:
        try:
            await db.user.update(
                where={"id": hr_id},
                data={"quota": float(remaining_quota)}
            )
            print(f"Updated HR user quota from {hr_quota} to {remaining_quota}")
        except Exception as e:
            print(f"Error updating HR user quota: {e}")
            # Don't raise error here, just log it
    
    # Get updated HR user to verify quota was updated
    updated_hr_user = await db.user.find_first(where={"id": hr_id})
    updated_hr_quota = updated_hr_user.quota if hasattr(updated_hr_user, 'quota') and updated_hr_user.quota is not None else 0.0

    await db.disconnect()

    return {
        "inserted": len(inserted_employees),
        "availableQuota": available_quota,
        "employeesSaved": employees_saved,
        "skippedDueToQuota": skipped_due_to_quota,
        "duplicateEmails": duplicate_emails,
        "totalRowsInFile": len(rows),
        "hrPaid": is_paid,
        "hrQuota": hr_quota,  # Original quota before update
        "updatedHrQuota": updated_hr_quota,  # Quota after update
        "existingEmployeesBefore": existing_employee_count,
        "existingEmployeesAfter": new_employee_count,
        "remainingQuota": remaining_quota,
        "quotaUsed": hr_quota - updated_hr_quota,
        "emailVerifiedDate": current_datetime.isoformat(),
        "employees": inserted_employees,
        "quotaWarning": employees_saved < (len(rows) - duplicate_emails)
    }