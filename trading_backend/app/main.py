from fastapi import FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from .routers import market_analysis, chinese_platforms, screenshot, notification
from starlette.websockets import WebSocketDisconnect

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
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]  # Required for WebSocket upgrade
)

# Verify Binance API configuration
if not os.getenv("BINANCE_API_KEY") or not os.getenv("BINANCE_API_SECRET"):
    # Use mock data if no API keys are present
    os.environ["BINANCE_API_KEY"] = "mock_key"
    os.environ["BINANCE_API_SECRET"] = "mock_secret"

# Include routers (remove prefixes as they're defined in the routers)
app.include_router(market_analysis.router)
app.include_router(chinese_platforms.router)
app.include_router(screenshot.router)
app.include_router(notification.router)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Keep connection alive and wait for messages
            data = await websocket.receive_text()
            # Echo back the message for now
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")

@app.get("/")
async def root():
    return {"status": "ok", "message": "Crypto Trading System API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
