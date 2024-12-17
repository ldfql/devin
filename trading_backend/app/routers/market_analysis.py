from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from pydantic import BaseModel
from ..services.market_analysis.market_cycle import MarketCycleAnalyzer
from ..services.market_analysis.exceptions import MarketAnalysisError
import os

router = APIRouter(
    prefix="/api/market-analysis",
    tags=["market-analysis"]
)

class PositionSizeRequest(BaseModel):
    account_balance: float
    risk_level: float
    market_data: Dict[str, Any]

async def get_market_analyzer() -> MarketCycleAnalyzer:
    """Dependency to get market analyzer instance."""
    api_key = os.getenv("BINANCE_API_KEY", "test_key")
    api_secret = os.getenv("BINANCE_API_SECRET", "test_secret")
    try:
        analyzer = MarketCycleAnalyzer(api_key, api_secret)
        return analyzer
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initialize market analyzer: {str(e)}")

@router.get("/analyze/{symbol}")
async def analyze_market(symbol: str, analyzer: MarketCycleAnalyzer = Depends(get_market_analyzer)):
    """Analyze market conditions for a given symbol."""
    try:
        return await analyzer.analyze_market(symbol)
    except MarketAnalysisError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/position-size")
async def calculate_position_size(
    request: PositionSizeRequest,
    analyzer: MarketCycleAnalyzer = Depends(get_market_analyzer)
):
    """Calculate position size based on account balance and market conditions."""
    try:
        return analyzer.calculate_position_size(
            request.account_balance,
            request.risk_level,
            request.market_data
        )
    except MarketAnalysisError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
