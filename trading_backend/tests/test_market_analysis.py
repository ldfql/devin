import pytest
import pytest_asyncio
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from app.services.market_analysis.market_cycle import MarketCycleAnalyzer
from app.services.market_analysis.exceptions import MarketAnalysisError
from binance.client import Client

@pytest_asyncio.fixture(scope="function")
async def mock_binance_client():
    """Create a mocked Binance client."""
    with patch('app.services.market_analysis.market_cycle.Client') as mock_client:
        # Mock successful ping
        mock_client.return_value.ping.return_value = {}
        # Mock other necessary methods
        mock_client.return_value.get_klines.return_value = []
        yield mock_client.return_value

@pytest.fixture(scope="function")
def market_analyzer(mock_binance_client):
    """Create a market analyzer instance with mocked client for testing."""
    with patch('xgboost.XGBClassifier') as mock_xgb:
        mock_model = Mock()
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])  # 90% confidence
        mock_xgb.return_value = mock_model

        analyzer = MarketCycleAnalyzer(
            api_key="test_key",
            api_secret="test_secret"
        )
        analyzer.client = mock_binance_client
        analyzer.model = mock_model
        return analyzer

@pytest.fixture(scope="function")
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
    data = {
        'open': np.random.uniform(45000, 50000, len(dates)),
        'high': np.random.uniform(45000, 50000, len(dates)),
        'low': np.random.uniform(45000, 50000, len(dates)),
        'close': np.random.uniform(45000, 50000, len(dates)),
        'volume': np.random.uniform(1000, 5000, len(dates))
    }
    return pd.DataFrame(data, index=dates)

@pytest.mark.asyncio
async def test_market_prediction_confidence(market_analyzer, sample_market_data):
    """Test that market predictions meet confidence threshold."""
    # Create bullish market conditions
    bull_data = sample_market_data.copy()
    bull_data['close'] = bull_data['close'] * 1.1  # Create uptrend

    with patch.object(market_analyzer.model, 'predict_proba') as mock_predict:
        mock_predict.return_value = np.array([[0.1, 0.9]])  # 90% confidence
        prediction = await market_analyzer.predict_market_direction(bull_data)

        # Verify prediction results
        assert prediction['confidence'] >= 0.85, "Bullish predictions must meet confidence threshold"
        assert prediction['is_bullish'], "Should be bullish with high confidence"
        assert not prediction.get('is_bearish', False), "Should not be bearish"
        assert not prediction.get('is_uncertain', False), "Should not be uncertain"

@pytest.mark.asyncio
async def test_position_sizing(market_analyzer, sample_market_data):
    """Test position sizing logic."""
    test_balances = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]

    market_data = {
        'confidence': 0.9,
        'technical_indicators': {
            'volatility': 0.02
        }
    }

    for balance in test_balances:
        position_info = market_analyzer.calculate_position_size(
            account_balance=balance,
            risk_level=0.8,
            market_data=market_data
        )

        # Test minimum position size
        assert position_info['position_size'] >= 100, "Position size must be at least 100 USDT"
        # Test maximum position size
        assert position_info['position_size'] <= 100000000, "Position size must not exceed 100M USDT"
        # Test profit allocation
        assert abs(position_info['reinvestment_amount'] - position_info['position_size'] * 0.7) < 0.01, \
            "70% of profits should be allocated for reinvestment"
        assert abs(position_info['withdrawal_amount'] - position_info['position_size'] * 0.3) < 0.01, \
            "30% of profits should be allocated for withdrawal"

@pytest.mark.asyncio
async def test_market_phase_detection(market_analyzer, sample_market_data):
    """Test dynamic market phase detection."""
    # Test bullish conditions
    bull_data = sample_market_data.copy()
    bull_data['close'] = bull_data['close'] * 1.1  # Create uptrend

    with patch.object(market_analyzer.model, 'predict_proba') as mock_predict:
        mock_predict.return_value = np.array([[0.1, 0.8]])  # 80% confidence
        prediction = await market_analyzer.predict_market_direction(bull_data)
        assert prediction['is_bullish'], "Should detect bull market"
        assert not prediction.get('is_bearish', False), "Should not be bearish"
        assert not prediction.get('is_uncertain', False), "Should not be uncertain"

    # Test bearish conditions
    bear_data = sample_market_data.copy()
    bear_data['close'] = bear_data['close'] * 0.9  # Create downtrend

    with patch.object(market_analyzer.model, 'predict_proba') as mock_predict:
        mock_predict.return_value = np.array([[0.8, 0.2]])  # 20% confidence
        prediction = await market_analyzer.predict_market_direction(bear_data)
        assert prediction['is_bearish'], "Should detect bear market"
        assert not prediction['is_bullish'], "Should not be bullish"
        assert not prediction.get('is_uncertain', False), "Should not be uncertain"

    # Test uncertain conditions
    uncertain_data = sample_market_data.copy()
    uncertain_data['close'] = uncertain_data['close'] * np.random.uniform(0.98, 1.02, len(uncertain_data))

    with patch.object(market_analyzer.model, 'predict_proba') as mock_predict:
        mock_predict.return_value = np.array([[0.5, 0.5]])  # 50% confidence
        prediction = await market_analyzer.predict_market_direction(uncertain_data)
        assert prediction.get('is_uncertain', False), "Should detect uncertainty"
        assert prediction['confidence'] == 0.5, "Should have neutral confidence"

@pytest.mark.asyncio
async def test_enhanced_position_sizing(market_analyzer, sample_market_data):
    """Test position sizing adjustments for different market conditions."""
    base_market_data = {
        'confidence': 0.8,
        'technical_indicators': {
            'volatility': 0.02,
            'rsi': 60,
            'macd': 0.001,
            'bb_width': 0.04
        }
    }

    # Test bull market position sizing
    bull_market = base_market_data.copy()
    bull_market['is_bullish'] = True
    bull_market['is_bearish'] = False
    bull_market['is_uncertain'] = False

    position_bull = market_analyzer.calculate_position_size(
        account_balance=100000,
        risk_level=0.8,
        market_data=bull_market
    )

    # Test bear market position sizing
    bear_market = base_market_data.copy()
    bear_market['is_bullish'] = False
    bear_market['is_bearish'] = True
    bear_market['is_uncertain'] = False

    position_bear = market_analyzer.calculate_position_size(
        account_balance=100000,
        risk_level=0.8,
        market_data=bear_market
    )

    # Test uncertain market position sizing
    uncertain_market = base_market_data.copy()
    uncertain_market['is_bullish'] = False
    uncertain_market['is_bearish'] = False
    uncertain_market['is_uncertain'] = True

    position_uncertain = market_analyzer.calculate_position_size(
        account_balance=100000,
        risk_level=0.8,
        market_data=uncertain_market
    )

    # Verify position size relationships
    assert position_bull['position_size'] > position_bear['position_size'], \
        "Bull market positions should be larger than bear market positions"
    assert position_bear['position_size'] > position_uncertain['position_size'], \
        "Bear market positions should be larger than uncertain market positions"
    assert position_uncertain['position_size'] >= 100, \
        "Position size should respect minimum limit"

@pytest.mark.asyncio
async def test_technical_indicators(market_analyzer, sample_market_data):
    """Test enhanced technical indicator calculations."""
    features = market_analyzer.calculate_technical_features(sample_market_data)

    required_indicators = ['rsi', 'rsi_long', 'macd', 'macd_signal', 'macd_hist',
                         'volatility', 'bb_width']
    for indicator in required_indicators:
        assert indicator in features.columns, f"{indicator} should be calculated"

    assert features['rsi'].min() >= 0 and features['rsi'].max() <= 100, \
        "RSI should be between 0 and 100"
    assert features['rsi_long'].min() >= 0 and features['rsi_long'].max() <= 100, \
        "Long-term RSI should be between 0 and 100"

    assert (features['bb_width'] > 0).all(), "BB width should be positive"
