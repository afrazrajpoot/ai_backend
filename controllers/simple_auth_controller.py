# controllers/simple_auth_controller.py
from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
import jwt
import os

router = APIRouter(prefix="/api/auth", tags=["auth"])

# ⚠️ HARDCODED JWT SECRET - MUST MATCH MIDDLEWARE
JWT_SECRET = "flkjfd4738974932dnksmnd"
ALGORITHM = "HS256"

@router.post("/login")
async def login(user_data: dict):
    """Generate JWT token for FastAPI authentication"""
    try:
        user_id = user_data.get("user_id")
        email = user_data.get("email")
        
        if not user_id or not email:
            raise HTTPException(status_code=400, detail="Missing user_id or email")
        
        print(f"🔑 [AUTH LOGIN] Generating token for: {email}")
        print(f"🔑 [AUTH LOGIN] Using secret: {JWT_SECRET[:10]}...")
        
        # Create JWT token with 24-hour expiry
        token_data = {
            "user_id": user_id,
            "email": email,
            "role": user_data.get("role", "Employee"),
            "hr_id": user_data.get("hr_id"),
            "first_name": user_data.get("first_name", ""),
            "last_name": user_data.get("last_name", ""),
            "exp": datetime.utcnow() + timedelta(hours=24)
        }
        
        token = jwt.encode(token_data, JWT_SECRET, algorithm=ALGORITHM)
        
        print(f"🔑 [AUTH LOGIN] Token generated: {token[:50]}...")
        print(f"🔑 [AUTH LOGIN] Token expires: {token_data['exp']}")
        
        return {
            "success": True,
            "token": token,
            "expires_in": 24 * 60 * 60,  # 24 hours in seconds
            "user": {
                "id": user_id,
                "email": email,
                "role": user_data.get("role", "Employee")
            }
        }
        
    except Exception as e:
        print(f"❌ [AUTH LOGIN] Error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Login failed: {str(e)}")

@router.post("/verify")
async def verify_token(token_data: dict):
    """Verify if a token is valid"""
    try:
        token = token_data.get("token")
        if not token:
            raise HTTPException(status_code=400, detail="Token is required")
        
        print(f"🔍 [AUTH VERIFY] Verifying token: {token[:30]}...")
        print(f"🔍 [AUTH VERIFY] Using secret: {JWT_SECRET[:10]}...")
        
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        
        # Check expiration
        exp_timestamp = payload.get("exp")
        if exp_timestamp and datetime.utcnow() > datetime.fromtimestamp(exp_timestamp):
            print(f"❌ [AUTH VERIFY] Token expired")
            return {
                "valid": False,
                "error": "Token expired",
                "code": "TOKEN_EXPIRED"
            }
        
        print(f"✅ [AUTH VERIFY] Token valid for: {payload.get('email')}")
        
        return {
            "valid": True,
            "user": {
                "id": payload.get("user_id"),
                "email": payload.get("email"),
                "role": payload.get("role")
            }
        }
        
    except jwt.ExpiredSignatureError:
        print(f"❌ [AUTH VERIFY] Token expired (signature)")
        return {
            "valid": False,
            "error": "Token expired",
            "code": "TOKEN_EXPIRED"
        }
    except jwt.InvalidTokenError as e:
        print(f"❌ [AUTH VERIFY] Invalid token: {str(e)}")
        return {
            "valid": False,
            "error": f"Invalid token: {str(e)}",
            "code": "INVALID_TOKEN",
            "debug": f"Using secret: {JWT_SECRET[:10]}..."
        }
    except Exception as e:
        print(f"❌ [AUTH VERIFY] Error: {str(e)}")
        return {
            "valid": False,
            "error": str(e),
            "code": "VERIFICATION_ERROR"
        }