from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, List, Optional
from pydantic import BaseModel
from ..services.trading_strategy.strategy import TradingStrategy
from ..services.market_analysis.market_cycle import MarketCycleAnalyzer

router = APIRouter(prefix="/api/trading", tags=["trading"])

class TradeRequest(BaseModel):
    symbol: str
    position_size: float
    leverage: Optional[int] = 1
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None

class TradeResponse(BaseModel):
    trade_id: str
    symbol: str
    entry_price: float
    position_size: float
    leverage: int
    estimated_liquidation_price: Optional[float]
    status: str
    message: str

@router.post("/short-term/open", response_model=TradeResponse)
async def open_short_term_position(
    trade_request: TradeRequest,
    strategy: TradingStrategy = Depends(),
    market_analyzer: MarketCycleAnalyzer = Depends()
) -> TradeResponse:
    """Open a new short-term trading position."""
    try:
        # Validate market conditions
        market_state = await market_analyzer.analyze_market_state(trade_request.symbol)
        if not market_state["is_tradeable"]:
            raise HTTPException(
                status_code=400,
                detail=f"Market conditions unfavorable: {market_state['reason']}"
            )

        # Execute trade
        trade_result = await strategy.open_position(
            symbol=trade_request.symbol,
            position_size=trade_request.position_size,
            leverage=trade_request.leverage,
            stop_loss=trade_request.stop_loss,
            take_profit=trade_request.take_profit
        )

        return TradeResponse(**trade_result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/short-term/close/{trade_id}")
async def close_short_term_position(
    trade_id: str,
    strategy: TradingStrategy = Depends()
) -> Dict[str, any]:
    """Close an existing short-term trading position."""
    try:
        result = await strategy.close_position(trade_id)
        return {
            "status": "success",
            "trade_id": trade_id,
            "closing_price": result["closing_price"],
            "pnl": result["pnl"],
            "message": "Position closed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/positions/active")
async def get_active_positions(
    strategy: TradingStrategy = Depends()
) -> List[Dict[str, any]]:
    """Get all active trading positions."""
    try:
        positions = await strategy.get_active_positions()
        return positions
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analysis/short-term/{symbol}")
async def get_short_term_analysis(
    symbol: str,
    market_analyzer: MarketCycleAnalyzer = Depends()
) -> Dict[str, any]:
    """Get short-term market analysis for a symbol."""
    try:
        analysis = await market_analyzer.get_short_term_analysis(symbol)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
