from fastapi import Security, HTTPException, Depends
from fastapi.security.api_key import APIKeyHeader
from passlib.context import CryptContext
from starlette.status import HTTP_403_FORBIDDEN
import os
import secrets

# Initialize password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# API key header field
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

def generate_api_key():
    """Generate a random API key."""
    return secrets.token_urlsafe(32)

def get_hashed_api_key(api_key: str):
    """Hash an API key."""
    return pwd_context.hash(api_key)

def verify_api_key(plain_api_key: str, hashed_api_key: str):
    """Verify an API key against its hash."""
    return pwd_context.verify(plain_api_key, hashed_api_key)

async def get_api_key(api_key_header: str = Security(API_KEY_HEADER)):
    """Dependency to validate API key in requests."""
    if not api_key_header:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="No API key provided"
        )

    # Get the hashed API key from environment variable
    hashed_key = os.getenv("HASHED_API_KEY")
    if not hashed_key:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="API authentication not configured"
        )

    if not verify_api_key(api_key_header, hashed_key):
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Invalid API key"
        )

    return api_key_header
