prisma migrate dev --name init

uvicorn main:app --reload --port 8000
uvicorn main:socket_app --reload --port 8000
