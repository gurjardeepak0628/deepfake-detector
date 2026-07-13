from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import auth, detect, history, admin
from database import create_tables

app = FastAPI(title="DeepGuard AI API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

create_tables()

app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(detect.router, prefix="/api", tags=["Detection"])
app.include_router(history.router, prefix="/api", tags=["History"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])

@app.get("/")
def root():
    return {"message": "DeepGuard AI API is running"}