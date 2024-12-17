from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
import os
from ..services.market_analysis.market_cycle import MarketCycleAnalyzer
from ..services.market_analysis.exceptions import MarketAnalysisError

router = APIRouter(
    prefix="/api/market-analysis",
    tags=["market-analysis"]
)

async def get_market_analyzer() -> MarketCycleAnalyzer:
    """Dependency to get market analyzer instance."""
    api_key = os.getenv("BINANCE_API_KEY")
    api_secret = os.getenv("BINANCE_API_SECRET")

    if not api_key or not api_secret:
        raise HTTPException(
            status_code=500,
            detail="Binance API configuration missing"
        )

    return MarketCycleAnalyzer(api_key, api_secret)

@router.get("/analyze/{symbol}")
async def analyze_market(
    symbol: str,
    analyzer: MarketCycleAnalyzer = Depends(get_market_analyzer)
) -> Dict[str, Any]:
    """
    Analyze market conditions for a given symbol.

    Parameters:
    - symbol: Trading pair symbol (e.g., 'BTCUSDT')

    Returns:
    - Market analysis including prediction, position sizing, and technical indicators
    """
    try:
        return await analyzer.analyze_market(symbol)
    except MarketAnalysisError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/supported-pairs")
async def get_supported_pairs(
    analyzer: MarketCycleAnalyzer = Depends(get_market_analyzer)
) -> List[str]:
    """Get list of supported trading pairs."""
    try:
        # Focus on major pairs initially
        return [
            "BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT",
            "ADAUSDT", "DOGEUSDT", "MATICUSDT", "SOLUSDT"
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
