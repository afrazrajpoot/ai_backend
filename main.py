# main.py - Complete corrected main file
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from routes.assessment import router as assessment_router
from routes.employee_dashboard import router as dashboard_router
from routes.assessment_analyze import router as assessment_analyze_router
from routes.test_db import router as test_db_router
from config import settings
from utils.logger import logger

from controllers.job_controller import router as job_router
from controllers.employee_parse_controller import router as employee_router
# from controllers.chat_controller import router as chat_router  # Use ChatController router
from controllers.employee_controllers.chat_controller import router as chat_router  # Use ChatController router
from routes.hr_routes.chat_route import router as hr_chat_router
from routes.hr_routes.intervation_routes import router as analysis_router
from routes.hr_routes.department_summary_route import router as department_router
from routes.employee_route.recommendation_course import router as recommendation_course_router

# Import applications router
from routes.hr_routes.applications import router as applications_router

# New job creation router
from routes.hr_routes.job_creation import router as job_creation_router

# Import the Socket.IO instance from socket_manager
from utils.socket_manager import sio
import socketio

# Create FastAPI app
app = FastAPI()

# CORS Middleware - MUST come first
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://geniusfactor.ai",
        "https://www.geniusfactor.ai",
        "https://api.geniusfactor.ai",
        "https://geniusfactor.ai",
        "http://127.0.0.1:3000",
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include routers
app.include_router(assessment_router)
app.include_router(dashboard_router)
app.include_router(assessment_analyze_router)
app.include_router(test_db_router, prefix="/test", tags=["testing"])
app.include_router(job_router)
app.include_router(employee_router)
app.include_router(chat_router)  # Include ChatController router
app.include_router(hr_chat_router)
app.include_router(analysis_router)
app.include_router(department_router)
app.include_router(recommendation_course_router)
app.include_router(applications_router, prefix="/api")  # Applications under /applications
app.include_router(job_creation_router, prefix="/api")  # New job creation routes under /jobs

# Create ASGI app with both FastAPI and Socket.IO
socket_app = socketio.ASGIApp(sio, app)

@app.on_event("startup")
async def startup():
    logger.info("Application starting up...")
    logger.info("Socket.IO notification system initialized")

@app.on_event("shutdown")
async def shutdown():
    logger.info("Application shutting down...")

@app.get("/")
async def root():
    return {"message": "FastAPI with Socket.IO Notification System"}

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "socket_io": "active"
    }

@app.get("/test-notification/{user_id}")
async def test_notification(user_id: str):
    """Test endpoint to send a notification"""
    from services.notification_service import NotificationService
    from datetime import datetime
    
    test_data = {
        'message': 'Test notification from API endpoint',
        'progress': 75,
        'status': 'processing',
        'stage': 'test',
        'details': {'test': True},
        'timestamp': datetime.now().isoformat()
    }
    
    await NotificationService.send_user_notification(user_id, test_data)
    
    return {"status": "test_notification_sent", "user_id": user_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000, reload=True)