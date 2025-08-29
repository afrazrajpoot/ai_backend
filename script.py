# import pandas as pd

# data = [
#     {"title": "Frontend Developer", "description": "Build React applications", "location": "Lahore, PK", "salary": 50000, "type": "FULL_TIME"},
#     {"title": "Backend Developer", "description": "Develop REST APIs", "location": "Karachi, PK", "salary": 60000, "type": "FULL_TIME"},
#     {"title": "UX Designer", "description": "Design user interfaces", "location": "Remote", "salary": 40000, "type": "CONTRACT"},
#     {"title": "Data Analyst", "description": "Analyze business data", "location": "Islamabad, PK", "salary": 45000, "type": "PART_TIME"},
#     {"title": "Intern Developer", "description": "Support development team", "location": "Lahore, PK", "salary": 0, "type": "INTERNSHIP"},
# ]

# df = pd.DataFrame(data)
# df.to_excel("jobs.ods", engine="odf", index=False)
# print("jobs.ods created successfully!")
import pandas as pd

# Employee data
employee_data = [
    {"firstName": "Ali", "lastName": "Khan", "email": "ali.khan@example.com", "phoneNumber": "03001234567", "position": "Frontend Developer", "salary": 50000, "department": "IT"},
    {"firstName": "Sara", "lastName": "Ahmed", "email": "sara.ahmed@example.com", "phoneNumber": "03007654321", "position": "Backend Developer", "salary": 60000, "department": "IT"},
    {"firstName": "Omar", "lastName": "Hussain", "email": "omar.hussain@example.com", "phoneNumber": "03009876543", "position": "Data Analyst", "salary": 45000, "department": "Analytics"},
    {"firstName": "Ayesha", "lastName": "Ali", "email": "ayesha.ali@example.com", "phoneNumber": "03001112233", "position": "UX Designer", "salary": 40000, "department": "Design"},
    {"firstName": "Bilal", "lastName": "Raza", "email": "bilal.raza@example.com", "phoneNumber": "03005556677", "position": "Intern Developer", "salary": 0, "department": "IT"},
]

# Create DataFrame
df_employees = pd.DataFrame(employee_data)

# Save as ODS file
df_employees.to_excel("employees.ods", engine="odf", index=False)

print("employees.ods created successfully!")
