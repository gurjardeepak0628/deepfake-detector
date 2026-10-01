from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from database import get_connection
import jwt, os
import psycopg2.extras

router = APIRouter()
security = HTTPBearer()
SECRET = os.getenv("JWT_SECRET", "deepguard_secret_key_2024")

ADMIN_EMAIL = "gurjardeepak808@gmail.com"

def get_admin_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET, algorithms=["HS256"])
        if payload.get("email") != ADMIN_EMAIL:
            raise HTTPException(status_code=403, detail="Admin access only.")
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired.")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token.")

@router.get("/stats")
def get_stats(admin=Depends(get_admin_user)):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cursor.execute("SELECT COUNT(*) as total FROM users")
        total_users = cursor.fetchone()["total"]
        cursor.execute("SELECT COUNT(*) as total FROM detections")
        total_detections = cursor.fetchone()["total"]
        cursor.execute("SELECT COUNT(*) as total FROM detections WHERE prediction='FAKE'")
        total_fake = cursor.fetchone()["total"]
        cursor.execute("SELECT COUNT(*) as total FROM detections WHERE prediction='REAL'")
        total_real = cursor.fetchone()["total"]
        return {
            "total_users": total_users,
            "total_detections": total_detections,
            "total_fake": total_fake,
            "total_real": total_real,
            "fake_rate": round((total_fake / total_detections * 100), 1) if total_detections > 0 else 0
        }
    finally:
        cursor.close()
        conn.close()

@router.get("/users")
def get_all_users(admin=Depends(get_admin_user)):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cursor.execute("""
            SELECT u.id, u.name, u.email, u.created_at,
                   COUNT(d.id) as total_detections,
                   SUM(CASE WHEN d.prediction='FAKE' THEN 1 ELSE 0 END) as fake_count,
                   SUM(CASE WHEN d.prediction='REAL' THEN 1 ELSE 0 END) as real_count
            FROM users u
            LEFT JOIN detections d ON u.id = d.user_id
            GROUP BY u.id
            ORDER BY u.created_at DESC
        """)
        users = cursor.fetchall()
        for u in users:
            u["created_at"] = str(u["created_at"])
            u["fake_count"] = int(u["fake_count"] or 0)
            u["real_count"] = int(u["real_count"] or 0)
        return users
    finally:
        cursor.close()
        conn.close()

@router.get("/detections")
def get_all_detections(admin=Depends(get_admin_user)):
    conn = get_connection()
    cursor = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
    try:
        cursor.execute("""
            SELECT d.id, u.name, u.email, d.filename, d.file_type,
                   d.prediction, d.confidence, d.detection_time, d.created_at
            FROM detections d
            JOIN users u ON d.user_id = u.id
            ORDER BY d.created_at DESC
            LIMIT 200
        """)
        records = cursor.fetchall()
        for r in records:
            r["created_at"] = str(r["created_at"])
        return records
    finally:
        cursor.close()
        conn.close()

@router.delete("/detections/{detection_id}")
def delete_detection(detection_id: int, admin=Depends(get_admin_user)):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM detections WHERE id = %s", (detection_id,))
        conn.commit()
        return {"message": "Detection deleted successfully"}
    finally:
        cursor.close()
        conn.close()

@router.delete("/users/{user_id}")
def delete_user(user_id: int, admin=Depends(get_admin_user)):
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM users WHERE id = %s", (user_id,))
        conn.commit()
        return {"message": "User deleted successfully"}
    finally:
        cursor.close()
        conn.close()