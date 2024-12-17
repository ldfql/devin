import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from app.services.market_analysis.market_cycle import MarketCycleAnalyzer
from app.services.market_analysis.exceptions import MarketAnalysisError
from binance.client import Client

@pytest.fixture
def mock_binance_client():
    """Create a mocked Binance client."""
    with patch('app.services.market_analysis.market_cycle.Client') as mock_client:
        # Mock successful ping
        mock_client.return_value.ping.return_value = {}
        # Mock other necessary methods
        mock_client.return_value.get_klines.return_value = []
        yield mock_client.return_value

@pytest.fixture
def market_analyzer():
    """Create a market analyzer instance with mocked client for testing."""
    with patch('app.services.market_analysis.market_cycle.Client') as mock_client:
        mock_client.return_value.ping.return_value = {}
        mock_client.return_value.get_klines.return_value = []

        with patch('xgboost.XGBClassifier') as mock_xgb:
            mock_model = Mock()
            mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])  # 90% confidence
            mock_xgb.return_value = mock_model

            analyzer = MarketCycleAnalyzer(
                api_key="test_key",
                api_secret="test_secret"
            )
            return analyzer

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range(start='2024-01-01', end='2024-02-01', freq='D')
    data = {
        'open': np.random.uniform(45000, 50000, len(dates)),
        'high': np.random.uniform(45000, 50000, len(dates)),
        'low': np.random.uniform(45000, 50000, len(dates)),
        'close': np.random.uniform(45000, 50000, len(dates)),
        'volume': np.random.uniform(1000, 5000, len(dates))
    }
    return pd.DataFrame(data, index=dates)

async def test_market_prediction_confidence(market_analyzer, sample_market_data):
    """Test that market predictions meet confidence threshold."""
    with patch.object(market_analyzer.model, 'predict_proba') as mock_predict:
        mock_predict.return_value = np.array([[0.1, 0.9]])  # 90% confidence
        prediction = await market_analyzer.predict_market_direction(sample_market_data)
        assert prediction['confidence'] >= 0.85, "Bullish predictions must meet confidence threshold"
        assert prediction['is_bullish'] is True, "Should be bullish with high confidence"

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

async def test_bull_market_bias(market_analyzer, sample_market_data):
    """Test bull market bias until May 2024."""
    # Test current period (before May 2024)
    current_data = sample_market_data.copy()
    current_data.index = pd.date_range(start='2024-02-01', end='2024-03-02', freq='D')

    with patch.object(market_analyzer.model, 'predict_proba') as mock_predict:
        mock_predict.return_value = np.array([[0.1, 0.75]])  # 75% base confidence
        prediction = await market_analyzer.predict_market_direction(current_data)
        assert prediction['is_bull_period'], "Should be in bull period before May 2024"
        assert prediction['confidence'] > 0.85, "Confidence should be boosted during bull period"

    # Test future period (after May 2024)
    future_data = sample_market_data.copy()
    future_data.index = pd.date_range(start='2024-06-01', end='2024-07-02', freq='D')

    with patch.object(market_analyzer.model, 'predict_proba') as mock_predict:
        mock_predict.return_value = np.array([[0.1, 0.75]])  # 75% base confidence
        prediction = await market_analyzer.predict_market_direction(future_data)
        assert not prediction['is_bull_period'], "Should not be in bull period after May 2024"
        assert prediction['confidence'] <= 0.85, "Confidence should not be boosted after bull period"

async def test_technical_indicators(market_analyzer, sample_market_data):
    """Test technical indicator calculations."""
    features = market_analyzer.calculate_technical_features(sample_market_data)

    assert 'rsi' in features.columns, "RSI should be calculated"
    assert 'macd' in features.columns, "MACD should be calculated"
    assert 'volatility' in features.columns, "Volatility should be calculated"

    # Test RSI bounds
    assert features['rsi'].min() >= 0 and features['rsi'].max() <= 100, "RSI should be between 0 and 100"
