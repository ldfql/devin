from typing import Dict, List, Optional
import uuid
from datetime import datetime
from ..market_analysis.market_cycle import MarketCycleAnalyzer

class TradingStrategy:
    def __init__(self):
        self.active_positions: Dict[str, Dict] = {}
        self.market_analyzer = MarketCycleAnalyzer()

    async def open_position(
        self,
        symbol: str,
        position_size: float,
        leverage: int = 1,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Dict[str, any]:
        """Open a new trading position."""
        # Generate unique trade ID
        trade_id = str(uuid.uuid4())

        try:
            # Get current market price
            market_data = await self.market_analyzer.get_market_data(symbol)
            entry_price = market_data["current_price"]

            # Calculate estimated liquidation price
            estimated_liq_price = self._calculate_liquidation_price(
                entry_price=entry_price,
                position_size=position_size,
                leverage=leverage,
                is_long=True  # Default to long position for now
            )

            # Store position details
            position = {
                "trade_id": trade_id,
                "symbol": symbol,
                "entry_price": entry_price,
                "position_size": position_size,
                "leverage": leverage,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "estimated_liquidation_price": estimated_liq_price,
                "entry_time": datetime.utcnow().isoformat(),
                "status": "open"
            }

            self.active_positions[trade_id] = position
            return {
                **position,
                "message": "Position opened successfully",
                "status": "success"
            }

        except Exception as e:
            raise Exception(f"Failed to open position: {str(e)}")

    async def close_position(self, trade_id: str) -> Dict[str, any]:
        """Close an existing trading position."""
        if trade_id not in self.active_positions:
            raise Exception("Position not found")

        position = self.active_positions[trade_id]
        try:
            # Get current market price
            market_data = await self.market_analyzer.get_market_data(position["symbol"])
            closing_price = market_data["current_price"]

            # Calculate PnL
            entry_price = position["entry_price"]
            position_size = position["position_size"]
            leverage = position["leverage"]

            pnl = (closing_price - entry_price) * position_size * leverage

            # Update position status
            position["status"] = "closed"
            position["closing_price"] = closing_price
            position["closing_time"] = datetime.utcnow().isoformat()
            position["pnl"] = pnl

            # Remove from active positions
            del self.active_positions[trade_id]

            return {
                "trade_id": trade_id,
                "closing_price": closing_price,
                "pnl": pnl,
                "status": "success"
            }

        except Exception as e:
            raise Exception(f"Failed to close position: {str(e)}")

    async def get_active_positions(self) -> List[Dict[str, any]]:
        """Get all active trading positions."""
        return list(self.active_positions.values())

    def _calculate_liquidation_price(
        self,
        entry_price: float,
        position_size: float,
        leverage: int,
        is_long: bool,
        maintenance_margin: float = 0.01
    ) -> float:
        """Calculate estimated liquidation price."""
        # Simple liquidation price calculation
        # For more accurate results, consider exchange-specific formulas
        margin = position_size / leverage
        price_move_to_liquidation = (margin * (1 - maintenance_margin)) / position_size

        if is_long:
            return entry_price - price_move_to_liquidation
        else:
            return entry_price + price_move_to_liquidation
