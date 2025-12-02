# main.py - COMPLETE CORRECTED VERSION
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import socketio

# Import middleware FIRST
from middleware.auth_middleware import strict_auth_middleware

# Import ALL routers
from routes.assessment import router as assessment_router
from routes.employee_dashboard import router as dashboard_router
from routes.assessment_analyze import router as assessment_analyze_router
from routes.test_db import router as test_db_router
from controllers.job_controller import router as job_router
from controllers.employee_parse_controller import router as employee_router
from controllers.employee_controllers.chat_controller import router as chat_router
from routes.hr_routes.chat_route import router as hr_chat_router
from routes.hr_routes.intervation_routes import router as analysis_router
from routes.hr_routes.department_summary_route import router as department_router
from routes.employee_route.recommendation_course import router as recommendation_course_router
from routes.hr_routes.applications import router as applications_router
from routes.hr_routes.job_creation import router as job_creation_router
from controllers.simple_auth_controller import router as auth_router
from utils.socket_manager import sio

# Create FastAPI app
app = FastAPI(title="Employee Management API")

# ========== STEP 1: Add CORS Middleware ==========
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://geniusfactor.ai",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========== STEP 2: Add Authentication Middleware ==========
# ⚠️ THIS MUST BE ADDED BEFORE ANY ROUTERS ARE INCLUDED ⚠️
app.middleware("http")(strict_auth_middleware)

# ========== STEP 3: Include ALL Routers ==========
# Now include all your routers - they will ALL go through the middleware
app.include_router(auth_router)  # Has public endpoints defined in the router itself
app.include_router(assessment_router)
app.include_router(dashboard_router)
app.include_router(assessment_analyze_router)
app.include_router(job_router)
app.include_router(employee_router)  # This includes /employees/upload
app.include_router(chat_router)
app.include_router(hr_chat_router)
app.include_router(analysis_router)
app.include_router(department_router)
app.include_router(recommendation_course_router)
app.include_router(applications_router)
app.include_router(job_creation_router)
app.include_router(test_db_router, prefix="/test")

# ========== STEP 4: Root and Health Endpoints ==========
@app.get("/")
async def root():
    return {"message": "API with STRICT authentication enabled"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

# ========== STEP 5: Socket.IO ==========
socket_app = socketio.ASGIApp(sio, app)

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting FastAPI with STRICT authentication...")
    print("🔒 ALL endpoints require authentication except public ones")
    uvicorn.run(socket_app, host="0.0.0.0", port=8000, reload=True)