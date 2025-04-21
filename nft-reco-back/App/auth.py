"""Authentication and authorization functionality."""
from __future__ import annotations

import os
import json
import uuid
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

import jwt
import bcrypt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import OAuth2PasswordBearer, HTTPBearer, HTTPAuthorizationCredentials
from pydantic import ValidationError

from .config import (
    JWT_SECRET_KEY,
    JWT_ALGORITHM,
    JWT_EXPIRATION_MINUTES,
    PASSWORD_HASH_ALGORITHM,
    PASSWORD_HASH_ROUNDS,
    USERS_FILE,
    AUTH_RATE_LIMIT
)
from .schemas import UserRole, UserCreate, UserResponse, TokenData, UserLogin

# Setup OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")
# Setup HTTP Bearer scheme for token authentication
http_bearer = HTTPBearer()

# Thread lock for user operations
user_lock = threading.Lock()

# Rate limiting data structure
login_attempts = {}  # {username: [(timestamp, ip_address), ...]}


class UserManager:
    """User management and authentication."""
    
    def __init__(self, users_file: Path = USERS_FILE):
        self.users_file = users_file
        self.users: Dict[str, Dict[str, Any]] = {}
        self.email_to_id: Dict[str, str] = {}
        self.username_to_id: Dict[str, str] = {}
        self._load_users()
    
    def _load_users(self):
        """Load users from file if it exists."""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.users = data.get("users", {})
                    
                    # Build lookup dictionaries
                    self.email_to_id = {
                        user["email"].lower(): user_id 
                        for user_id, user in self.users.items()
                    }
                    
                    self.username_to_id = {
                        user["username"].lower(): user_id
                        for user_id, user in self.users.items()
                    }
            except json.JSONDecodeError:
                print(f"Warning: Could not parse users file {self.users_file}")
                self.users = {}
                self.email_to_id = {}
                self.username_to_id = {}
    
    def _save_users(self):
        """Save users to file."""
        os.makedirs(os.path.dirname(self.users_file), exist_ok=True)
        
        with open(self.users_file, "w", encoding="utf-8") as f:
            json.dump({"users": self.users}, f, ensure_ascii=False, indent=2, default=str)
    
    def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create a new user."""
        with user_lock:
            # Check if email already exists
            if user_data.email.lower() in self.email_to_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Email already registered"
                )
            
            # Check if username already exists
            if user_data.username.lower() in self.username_to_id:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Username already taken"
                )
            
            # Generate user ID
            user_id = str(uuid.uuid4())
            
            # Hash password
            hashed_password = self._hash_password(user_data.password)
            
            # Create user
            user = {
                "id": user_id,
                "username": user_data.username,
                "email": user_data.email,
                "full_name": user_data.full_name,
                "hashed_password": hashed_password.decode("utf-8"),
                "role": UserRole.USER.value,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat(),
                "is_active": True
            }
            
            # Save user
            self.users[user_id] = user
            self.email_to_id[user_data.email.lower()] = user_id
            self.username_to_id[user_data.username.lower()] = user_id
            
            self._save_users()
            
            # Return user data (without password)
            return UserResponse(
                id=user_id,
                username=user_data.username,
                email=user_data.email,
                full_name=user_data.full_name,
                role=UserRole(user["role"]),
                created_at=datetime.fromisoformat(user["created_at"])
            )
    
    def authenticate_user(self, username_or_email: str, password: str) -> Optional[UserResponse]:
        """Authenticate a user by username/email and password."""
        user_id = None
        
        # Check if input is an email
        if "@" in username_or_email:
            # Lookup by email
            user_id = self.email_to_id.get(username_or_email.lower())
        else:
            # Lookup by username
            user_id = self.username_to_id.get(username_or_email.lower())
        
        if not user_id:
            return None
        
        user = self.users.get(user_id)
        if not user:
            return None
        
        # Verify password
        if not self._verify_password(password, user["hashed_password"].encode("utf-8")):
            return None
        
        # Check if user is active
        if not user.get("is_active", True):
            return None
        
        # Return user data
        return UserResponse(
            id=user_id,
            username=user["username"],
            email=user["email"],
            full_name=user.get("full_name"),
            role=UserRole(user["role"]),
            created_at=datetime.fromisoformat(user["created_at"])
        )
    
    def get_user_by_id(self, user_id: str) -> Optional[UserResponse]:
        """Get user by ID."""
        user = self.users.get(user_id)
        if not user:
            return None
        
        return UserResponse(
            id=user_id,
            username=user["username"],
            email=user["email"],
            full_name=user.get("full_name"),
            role=UserRole(user["role"]),
            created_at=datetime.fromisoformat(user["created_at"])
        )
    
    def change_password(self, user_id: str, current_password: str, new_password: str) -> bool:
        """Change a user's password."""
        with user_lock:
            user = self.users.get(user_id)
            if not user:
                return False
            
            # Verify current password
            if not self._verify_password(current_password, user["hashed_password"].encode("utf-8")):
                return False
            
            # Hash new password
            hashed_password = self._hash_password(new_password)
            
            # Update password
            user["hashed_password"] = hashed_password.decode("utf-8")
            user["updated_at"] = datetime.now().isoformat()
            
            self.users[user_id] = user
            self._save_users()
            
            return True
    
    def _hash_password(self, password: str) -> bytes:
        """Hash a password using bcrypt."""
        salt = bcrypt.gensalt(rounds=PASSWORD_HASH_ROUNDS)
        return bcrypt.hashpw(password.encode("utf-8"), salt)
    
    def _verify_password(self, plain_password: str, hashed_password: bytes) -> bool:
        """Verify a password against a hash."""
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password)


# Create global instance
user_manager = UserManager()


def create_access_token(user: UserResponse) -> str:
    """Create a JWT access token."""
    expires_delta = timedelta(minutes=JWT_EXPIRATION_MINUTES)
    expire = datetime.utcnow() + expires_delta
    
    to_encode = {
        "sub": user.id,
        "exp": expire,
        "role": user.role.value
    }
    
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> TokenData:
    """Verify a JWT token and return the decoded data."""
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        token_data = TokenData(
            sub=payload["sub"],
            exp=datetime.fromtimestamp(payload["exp"]),
            role=UserRole(payload["role"])
        )
        
        # Check if token is expired
        if token_data.exp < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return token_data
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except ValidationError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token format",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(token: str = Depends(oauth2_scheme)) -> UserResponse:
    """Get the current user from a token."""
    token_data = verify_token(token)
    
    user = user_manager.get_user_by_id(token_data.sub)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user


async def get_current_user_from_bearer(credentials: HTTPAuthorizationCredentials = Security(http_bearer)) -> UserResponse:
    """Get the current user from HTTP Bearer token."""
    return await get_current_user(credentials.credentials)


async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))) -> Optional[UserResponse]:
    """Get the current user from HTTP Bearer token but make it optional."""
    if not credentials:
        return None
    
    try:
        token_data = verify_token(credentials.credentials)
        user = user_manager.get_user_by_id(token_data.sub)
        return user
    except (HTTPException, ValidationError, jwt.PyJWTError):
        return None


def check_rate_limit(username: str, ip_address: str) -> bool:
    """Check rate limiting for login attempts."""
    now = datetime.utcnow()
    window_start = now - timedelta(seconds=AUTH_RATE_LIMIT["window_seconds"])
    
    # Initialize if not exists
    if username not in login_attempts:
        login_attempts[username] = []
    
    # Clean up old attempts
    login_attempts[username] = [
        (timestamp, ip) for timestamp, ip in login_attempts[username]
        if timestamp >= window_start
    ]
    
    # Check if user is locked out
    if len(login_attempts[username]) >= AUTH_RATE_LIMIT["login_attempts"]:
        return False
    
    # Add current attempt
    login_attempts[username].append((now, ip_address))
    
    return True 