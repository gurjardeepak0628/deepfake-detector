from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from database import get_connection
import bcrypt
import jwt
import os
from datetime import datetime, timedelta

router = APIRouter()
SECRET = os.getenv("JWT_SECRET", "deepguard_secret_key_2024")

class RegisterRequest(BaseModel):
    name: str
    email: EmailStr
    password: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

def make_token(user_id: int, name: str, email: str):
    payload = {
        "user_id": user_id,
        "name": name,
        "email": email,
        "exp": datetime.utcnow() + timedelta(days=7)
    }
    return jwt.encode(payload, SECRET, algorithm="HS256")

@router.post("/register")
def register(body: RegisterRequest):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT id FROM users WHERE email = %s", (body.email,))
        if cursor.fetchone():
            raise HTTPException(status_code=400, detail="Email already registered.")

        hashed = bcrypt.hashpw(body.password.encode(), bcrypt.gensalt()).decode()
        cursor.execute(
            "INSERT INTO users (name, email, password) VALUES (%s, %s, %s)",
            (body.name, body.email, hashed)
        )
        conn.commit()
        user_id = cursor.lastrowid
        token = make_token(user_id, body.name, body.email)
        return {
            "token": token,
            "user": {"id": user_id, "name": body.name, "email": body.email}
        }
    finally:
        cursor.close()
        conn.close()

@router.post("/login")
def login(body: LoginRequest):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute("SELECT * FROM users WHERE email = %s", (body.email,))
        user = cursor.fetchone()
        if not user or not bcrypt.checkpw(body.password.encode(), user["password"].encode()):
            raise HTTPException(status_code=401, detail="Invalid email or password.")

        token = make_token(user["id"], user["name"], user["email"])
        return {
            "token": token,
            "user": {"id": user["id"], "name": user["name"], "email": user["email"]}
        }
    finally:
        cursor.close()
        conn.close()
