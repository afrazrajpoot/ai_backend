# middleware/simple_auth_middleware.py
from fastapi import Request
from fastapi.responses import JSONResponse
import jwt
import os
from datetime import datetime
import time

# ⚠️ HARDCODED JWT SECRET - MUST MATCH NEXT.JS FASTATPI_JWT_SECRET
JWT_SECRET = "flkjfd4738974932dnksmnd"
ALGORITHM = "HS256"

async def strict_auth_middleware(request: Request, call_next):
    """STRICT authentication middleware for ALL endpoints"""
    
    start_time = time.time()
    
    # Print debug info
    print(f"\n🔍 [AUTH] Request: {request.method} {request.url.path}")
    print(f"🔍 [AUTH] Using JWT_SECRET: {JWT_SECRET[:10]}...")
    
    # Public endpoints (VERY LIMITED)
    public_paths = [
        "/",
        "/health",
        "/docs",
        "/redoc", 
        "/openapi.json",
        "/api/auth/login",
        "/api/auth/verify",
        "/socket.io/",
        "/favicon.ico",
    ]
    
    current_path = request.url.path
    is_public_path = any(current_path == path for path in public_paths)
    
    # Skip auth for public paths and OPTIONS
    if is_public_path or request.method == "OPTIONS":
        print(f"🔍 [AUTH] Skipping auth for public path")
        response = await call_next(request)
        return response
    
    # Get Authorization header
    auth_header = request.headers.get("Authorization")
    
    if not auth_header or not auth_header.startswith("Bearer "):
        print(f"❌ [AUTH] Missing or invalid Authorization header")
        return JSONResponse(
            status_code=401,
            content={
                "error": "Authentication required",
                "message": "Missing or invalid Authorization header",
                "path": current_path,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # Extract token
    token = auth_header.split("Bearer ")[1].strip()
    print(f"🔍 [AUTH] Token received: {token[:30]}...")
    
    try:
        # First, decode without verification to see payload
        try:
            unverified_payload = jwt.decode(token, options={"verify_signature": False})
            print(f"🔍 [AUTH] Token payload (unverified): {unverified_payload}")
        except:
            print(f"🔍 [AUTH] Could not decode token even without verification")
        
        # Now verify with the secret
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        
        # Check expiration
        exp = payload.get("exp")
        if exp and datetime.utcnow() > datetime.fromtimestamp(exp):
            print(f"❌ [AUTH] Token expired")
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Token expired", 
                    "message": "Please login again",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Set user in request state
        request.state.user = {
            "id": payload.get("user_id"),
            "email": payload.get("email"),
            "role": payload.get("role", "Employee"),
            "hr_id": payload.get("hr_id"),
            "first_name": payload.get("first_name", ""),
            "last_name": payload.get("last_name", ""),
        }
        
        print(f"✅ [AUTH] Authenticated as: {payload.get('email')} ({payload.get('role')})")
        
    except jwt.ExpiredSignatureError:
        print(f"❌ [AUTH] Token expired (signature error)")
        return JSONResponse(
            status_code=401,
            content={
                "error": "Token expired",
                "message": "Your session has expired. Please login again.",
                "timestamp": datetime.now().isoformat()
            }
        )
    except jwt.InvalidTokenError as e:
        print(f"❌ [AUTH] Invalid token: {str(e)}")
        
        # Try to give more helpful error
        if "signature" in str(e).lower():
            error_msg = "Token signature verification failed. This usually means the JWT secret doesn't match."
        else:
            error_msg = "Invalid token format"
            
        return JSONResponse(
            status_code=401,
            content={
                "error": "Invalid token",
                "message": error_msg,
                "details": str(e),
                "timestamp": datetime.now().isoformat(),
                "debug_info": f"Using secret: {JWT_SECRET[:10]}..."
            }
        )
    except Exception as e:
        print(f"❌ [AUTH] Unexpected error: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "Authentication error",
                "message": "Internal server error during authentication",
                "timestamp": datetime.now().isoformat()
            }
        )
    
    # Process the request
    response = await call_next(request)
    elapsed = time.time() - start_time
    
    print(f"✅ [AUTH] Request completed in {elapsed:.3f}s")
    
    return response