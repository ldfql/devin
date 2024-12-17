from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from .routers import market_analysis, chinese_platforms

# Load environment variables
load_dotenv()

app = FastAPI(
    title="Crypto Trading System",
    description="Real-time cryptocurrency trading monitoring and prediction system",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Verify Binance API configuration
if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
    raise HTTPException(
        status_code=500,
        detail="Binance API configuration missing"
    )

# Include routers
app.include_router(market_analysis.router)
app.include_router(chinese_platforms.router)

@app.get("/")
async def root():
    return {"status": "ok", "message": "Crypto Trading System API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
