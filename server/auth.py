from fastapi import Depends, HTTPException, status, Request, Cookie
from fastapi.security import OAuth2PasswordBearer
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from datetime import datetime, timedelta
from pydantic import BaseModel
from typing import Optional
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Secret key and algorithm for JWT
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-should-be-in-env-file")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Admin credentials from .env
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "password")  # Default for development only

# Token model
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

# Simple OAuth2 with cookie
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login", auto_error=False)

def authenticate_user(username: str, password: str):
    """Verifies username and password against .env values"""
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """Creates a JWT token with specified expiration"""
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(request: Request):
    """Validates the user's token from cookie"""
    access_token = request.cookies.get("access_token")
    
    if not access_token:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login"}
        )
    
    # Strip "Bearer " prefix if present
    if isinstance(access_token, str) and access_token.startswith("Bearer "):
        access_token = access_token[7:]
    
    try:
        payload = jwt.decode(access_token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if not username or username != ADMIN_USERNAME:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return username
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_307_TEMPORARY_REDIRECT,
            headers={"Location": "/login?error=Session+expired"}
        )

# Use this as a dependency for protected routes
async def get_current_active_user(request: Request):
    """Use this dependency to protect routes"""
    return await get_current_user(request)