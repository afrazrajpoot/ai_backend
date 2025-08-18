prisma migrate dev --name init

uvicorn main:app --reload --port 8000
