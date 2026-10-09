"""
AI Squad Backend Entry Point
Initializes FastAPI, sets up CORS, and includes the API routes.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.v1.chat import router as chat_router

app = FastAPI(title="AI Squad API", version="1.0")

# Allow frontend to communicate with backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(chat_router, prefix="/api/v1")

@app.get("/")
def health_check():
    return {"status": "AI Squad Backend is running"}