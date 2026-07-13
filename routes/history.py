from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from database import get_connection
import jwt, os

router = APIRouter()
security = HTTPBearer()
SECRET = os.getenv("JWT_SECRET", "deepguard_secret_key_2024")

def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        return jwt.decode(credentials.credentials, SECRET, algorithms=["HS256"])
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token.")

@router.get("/history")
def get_history(user=Depends(get_current_user)):
    conn = get_connection()
    cursor = conn.cursor(dictionary=True)
    try:
        cursor.execute(
            """SELECT id, filename, file_type, prediction, confidence, detection_time, created_at
               FROM detections
               WHERE user_id = %s
               ORDER BY created_at DESC
               LIMIT 100""",
            (user["user_id"],)
        )
        records = cursor.fetchall()
        # Convert datetime to string for JSON
        for r in records:
            r["created_at"] = str(r["created_at"])
        return records
    finally:
        cursor.close()
        conn.close()
