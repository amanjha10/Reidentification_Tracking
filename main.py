from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import OAuth2PasswordRequestForm
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from jose import jwt
from datetime import datetime, timedelta
from database import get_db_connection
from urllib.parse import quote

app = FastAPI()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

from dotenv import load_dotenv
import os
load_dotenv()

SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES"))


# =============================
# Pydantic Models
# =============================
class UserRegister(BaseModel):
    email: EmailStr
    password: str

class CameraInfo(BaseModel):
    camera_id: str      # ex: 192.168.1.5
    username: str
    password: str
    port: int           # ex: 554
    stream: str         # ex: stream1


# =============================
# Helper functions
# =============================
def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(plain, hashed):
    return pwd_context.verify(plain, hashed)


def hash_password(password: str):
    return pwd_context.hash(password)


# =============================
# User Registration API
# =============================
@app.post("/register")
def register_user(user: UserRegister):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT * FROM users WHERE email=%s", (user.email,))
    existing = cur.fetchone()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    hashed = hash_password(user.password)

    cur.execute("INSERT INTO users (email, password) VALUES (%s, %s)",
                (user.email, hashed))
    conn.commit()

    cur.close()
    conn.close()

    return {"message": "User registered successfully"}


# =============================
# Login API (Email + Password)
# =============================
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("SELECT id, email, password FROM users WHERE email=%s",
                (form_data.username,))
    user = cur.fetchone()

    if not user:
        raise HTTPException(status_code=400, detail="Invalid email or password")

    user_id, email, hashed_password = user

    if not verify_password(form_data.password, hashed_password):
        raise HTTPException(status_code=400, detail="Invalid email or password")

    access_token = create_access_token(
        data={"sub": email},
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    )

    cur.close()
    conn.close()

    return {"access_token": access_token, "token_type": "bearer"}


# =============================
# RTSP URL Generator API
# =============================
@app.post("/generate_rtsp")
def generate_rtsp(info: CameraInfo):
    """
    Example Input:
    {
        "camera_id": "192.168.1.5",
        "username": "admin",
        "password": "14562@",
        "port": 554,
        "stream": "stream1"
    }
    """

    # Encode password to be URL safe
    encoded_pass = quote(info.password)

    rtsp_url = f"rtsp://{info.username}:{encoded_pass}@{info.camera_id}:{info.port}/{info.stream}"

    return {
        "rtsp_url": rtsp_url
    }


# =============================
# Root Test
# =============================
@app.get("/")
def root():
    return {"message": "FastAPI backend is running!"}
