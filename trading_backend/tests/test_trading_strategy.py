import pytest
from datetime import datetime
from app.services.trading_strategy.strategy import TradingStrategy
from app.services.market_analysis.market_cycle import MarketCycleAnalyzer

@pytest.fixture
async def trading_strategy():
    return TradingStrategy()

@pytest.fixture
async def market_analyzer():
    return MarketCycleAnalyzer()

@pytest.mark.asyncio
async def test_open_position(trading_strategy):
    """Test opening a new trading position."""
    position = await trading_strategy.open_position(
        symbol="BTCUSDT",
        position_size=1.0,
        leverage=2,
        stop_loss=19000.0,
        take_profit=21000.0
    )

    assert position["symbol"] == "BTCUSDT"
    assert position["position_size"] == 1.0
    assert position["leverage"] == 2
    assert position["status"] == "success"
    assert "trade_id" in position
    assert "entry_price" in position
    assert "estimated_liquidation_price" in position
    assert isinstance(position["entry_time"], str)

    # Verify position is stored
    active_positions = await trading_strategy.get_active_positions()
    assert len(active_positions) == 1
    assert active_positions[0]["trade_id"] == position["trade_id"]

@pytest.mark.asyncio
async def test_close_position(trading_strategy):
    """Test closing a trading position."""
    # First open a position
    position = await trading_strategy.open_position(
        symbol="BTCUSDT",
        position_size=1.0
    )
    trade_id = position["trade_id"]

    # Close the position
    result = await trading_strategy.close_position(trade_id)
    assert result["status"] == "success"
    assert result["trade_id"] == trade_id
    assert "closing_price" in result
    assert "pnl" in result

    # Verify position is removed from active positions
    active_positions = await trading_strategy.get_active_positions()
    assert len(active_positions) == 0

@pytest.mark.asyncio
async def test_get_active_positions(trading_strategy):
    """Test retrieving active positions."""
    # Open multiple positions
    positions = []
    for i in range(3):
        position = await trading_strategy.open_position(
            symbol=f"BTC{i}USDT",
            position_size=1.0
        )
        positions.append(position)

    active_positions = await trading_strategy.get_active_positions()
    assert len(active_positions) == 3

    # Verify all positions are present
    position_ids = {p["trade_id"] for p in active_positions}
    expected_ids = {p["trade_id"] for p in positions}
    assert position_ids == expected_ids

@pytest.mark.asyncio
async def test_liquidation_price_calculation(trading_strategy):
    """Test liquidation price calculation."""
    position = await trading_strategy.open_position(
        symbol="BTCUSDT",
        position_size=1.0,
        leverage=10
    )

    assert position["estimated_liquidation_price"] < position["entry_price"]

    # Test with higher leverage
    high_leverage_position = await trading_strategy.open_position(
        symbol="BTCUSDT",
        position_size=1.0,
        leverage=20
    )

    # Higher leverage should result in liquidation price closer to entry
    assert (high_leverage_position["entry_price"] - high_leverage_position["estimated_liquidation_price"]) < \
           (position["entry_price"] - position["estimated_liquidation_price"])

@pytest.mark.asyncio
async def test_market_validation(trading_strategy, market_analyzer):
    """Test market condition validation before trading."""
    # Mock unfavorable market conditions
    market_analyzer.is_tradeable = lambda x: False
    market_analyzer.get_market_reason = lambda: "High volatility"

    with pytest.raises(Exception) as exc_info:
        await trading_strategy.open_position(
            symbol="BTCUSDT",
            position_size=1.0
        )
    assert "Market conditions unfavorable" in str(exc_info.value)
