from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from routes.assessment import router as assessment_router
from routes.employee_dashboard import router as dashboard_router
from routes.assessment_analyze import router as assessment_analyze_router

from config import settings
from utils.logger import logger

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(assessment_router)
app.include_router(dashboard_router)
app.include_router(assessment_analyze_router)

@app.on_event("startup")
async def startup():
    logger.info("Application starting up...")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Application shutting down...")

@app.get("/")
async def root():
    return {"message": "Hello, FastAPI!"}
