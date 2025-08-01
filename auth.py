from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from fastapi import Depends, FastAPI, HTTPException, status,  APIRouter
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jwt.exceptions import InvalidTokenError
from passlib.context import CryptContext
from pydantic import BaseModel
from database import get_db
from sqlalchemy.orm import Session
from models import User as DBUser
from langchain_core.runnables import RunnableLambda


def component(func):
    return RunnableLambda(func)


router = APIRouter()

# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "95e3e35c3cd46d3585b0d1b3a55fd6a13cf8783357eb985422eeb02be6ca7e8a"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class UserCreate(BaseModel):
    email: str
    username: str
    password: str


class SignupResponse(BaseModel):
    message: str
    username: str


class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    is_verified: bool
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# STATE MODELS
# ============================================================================

class TokenValidationState(BaseModel):
    """State model for token validation pipeline"""
    token: str
    username: str | None = None
    user: DBUser | None = None
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True


class AuthenticationState(BaseModel):
    """State model for authentication pipeline"""
    username: str
    password: str
    expires_delta: timedelta | None = None
    user: DBUser | None = None
    access_token: str | None = None

    class Config:
        arbitrary_types_allowed = True


class SignupState(BaseModel):
    """State model for signup pipeline"""
    username: str
    email: str
    password: str
    user_exists: bool = False
    hashed_password: str | None = None
    created_user: DBUser | None = None

    class Config:
        arbitrary_types_allowed = True


pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


# ============================================================================
# SHARED COMPONENTS
# ============================================================================

@component
def get_user(state: dict) -> dict:
    db = state["db"]
    username = state["username"]
    user = db.query(DBUser).filter(DBUser.username == username).first()
    return {**state, "user": user}


# ============================================================================
# TOKEN VALIDATION ENDPOINT
# ============================================================================

@component
def decode_and_validate_token(state: dict) -> dict:
    token = state["token"]
    db = state["db"]
    
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        if username is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token payload",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database
        user = db.query(DBUser).filter(DBUser.username == username).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return {**state, "username": username, "user": user}
        
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Token validation pipeline
token_pipeline = decode_and_validate_token


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Session = Depends(get_db)
):
    # Create initial state using the state model
    initial_state = TokenValidationState(token=token)
    result = token_pipeline.invoke({**initial_state.model_dump(), "db": db})
    
    return result["user"]


async def get_current_active_user(
    current_user: Annotated[DBUser, Depends(get_current_user)],
):
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# ============================================================================
# LOGIN ENDPOINT
# ============================================================================

@component
def check_user_and_authenticate(state: dict) -> dict:
    user = state.get("user")
    password = state.get("password")
    
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    is_valid = pwd_context.verify(password, user.hashed_password)
    if not is_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return state


@component
def create_access_token(state: dict) -> dict:
    user = state["user"]
    expires_delta = state.get("expires_delta")
    
    data = {"sub": user.username}
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return {**state, "access_token": encoded_jwt}


@component
def format_token_response(state: dict) -> Token:
    access_token = state["access_token"]
    return Token(access_token=access_token, token_type="bearer")


# Authentication pipeline
auth_pipeline = (
    get_user
    | check_user_and_authenticate
    | create_access_token
    | format_token_response
)


from fastapi.responses import JSONResponse

@router.post("/token", response_model=Token)
async def login_for_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Session = Depends(get_db)
):
    initial_state = AuthenticationState(
        username=form_data.username,
        password=form_data.password,
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    result = auth_pipeline.invoke({**initial_state.model_dump(), "db": db})
    
    # Create the response
    response = JSONResponse(content=result.model_dump())  # still returns token in JSON if needed
    
    # Set the JWT token as a cookie
    response.set_cookie(
        key="jwt",
        value=result.access_token,
        httponly=False,  # True = not accessible in JS, set to False only if you read it in client-side JS
        secure=False,    # Set to True if using HTTPS (recommended in production)
        samesite="Lax",  # "Strict" or "None" depending on cross-site needs
        max_age=60 * ACCESS_TOKEN_EXPIRE_MINUTES,
        path="/"
    )

    return response



# ============================================================================
# USER INFO ENDPOINTS
# ============================================================================

@router.get("/users/me/", response_model=User)
async def read_users_me(
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
):
    return current_user


@router.get("/users/me/items/")
async def read_own_items(
    current_user: Annotated[DBUser, Depends(get_current_active_user)],
):
    return [{"item_id": "Foo", "owner": current_user.username}]


# ============================================================================
# SIGNUP ENDPOINT
# ============================================================================

@component
def check_user_exists(state: dict) -> dict:
    db = state["db"]
    username = state["username"]
    email = state["email"]
    
    existing_user = db.query(DBUser).filter(
        (DBUser.username == username) | (DBUser.email == email)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Username or email already registered"
        )
    
    return {**state, "user_exists": False}


@component
def hash_user_password(state: dict) -> dict:
    password = state["password"]
    hashed_password = pwd_context.hash(password)
    return {**state, "hashed_password": hashed_password}


@component
def create_user_in_db(state: dict) -> dict:
    db = state["db"]
    username = state["username"]
    email = state["email"]
    hashed_password = state["hashed_password"]
    
    user = DBUser(
        username=username,
        email=email,
        hashed_password=hashed_password
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    return {**state, "created_user": user}


@component
def format_signup_response(state: dict) -> SignupResponse:
    user = state["created_user"]
    return SignupResponse(message="User created successfully", username=user.username)


# Signup pipeline
signup_pipeline = (
    check_user_exists
    | hash_user_password
    | create_user_in_db
    | format_signup_response
)


@router.post("/signup", response_model=SignupResponse)
async def signup(user_data: UserCreate, db: Session = Depends(get_db)):
    # Create initial state using the state model
    initial_state = SignupState(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password
    )
    result = signup_pipeline.invoke({**initial_state.model_dump(), "db": db})
    
    return result




from fastapi import Request, HTTPException

@router.post("/logout")
async def logout(request: Request):
    jwt_token = request.cookies.get("jwt")
    if not jwt_token:
        raise HTTPException(status_code=401, detail="Not logged in")

    response = JSONResponse(content={"message": "Logged out"})
    response.delete_cookie(key="jwt")
    return response

