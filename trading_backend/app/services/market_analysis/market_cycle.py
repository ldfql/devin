from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from binance.client import Client
import xgboost as xgb
import pandas_ta as ta

class MarketCycleAnalyzer:
    def __init__(self, api_key: str, api_secret: str):
        """Initialize the market cycle analyzer."""
        self.client = Client(api_key, api_secret)
        self.model = xgb.XGBClassifier()
        # Initialize with default model weights
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the XGBoost model with default parameters."""
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            objective='binary:logistic'
        )

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for market analysis."""
        features = pd.DataFrame(index=df.index)

        # Calculate RSI
        features['rsi'] = ta.rsi(df['close'], length=14)

        # Calculate MACD manually using EMAs
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal

        # Calculate volatility (ATR)
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        features['volatility'] = atr / df['close']

        # Fill any NaN values with forward fill then backward fill
        features = features.fillna(method='ffill').fillna(method='bfill')

        return features

    def calculate_position_size(
        self,
        account_balance: float,
        risk_level: float,
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate position size based on account balance and market conditions."""
        # Base position size on account balance and risk level
        base_position = account_balance * risk_level

        # Adjust based on market confidence
        confidence_factor = market_data.get('confidence', 0.5)
        volatility = market_data.get('technical_indicators', {}).get('volatility', 0.02)

        # Adjust position size based on volatility and confidence
        adjusted_position = base_position * confidence_factor * (1 / volatility)

        # Enforce position size limits
        position_size = max(100, min(adjusted_position, 100_000_000))

        # Calculate profit allocation
        reinvestment_amount = position_size * 0.7  # 70% for reinvestment
        withdrawal_amount = position_size * 0.3    # 30% for withdrawal

        return {
            'position_size': position_size,
            'reinvestment_amount': reinvestment_amount,
            'withdrawal_amount': withdrawal_amount
        }

    async def predict_market_direction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict market direction with confidence score."""
        # Calculate technical features
        features = self.calculate_technical_features(df)

        # Get prediction probability
        pred_proba = self.model.predict_proba(features.iloc[[-1]])
        base_confidence = pred_proba[0][1]  # Probability of bullish prediction

        # Check if we're in the bull market period (before May 2024)
        current_date = df.index[-1]
        bull_end = pd.Timestamp('2024-05-01')
        is_bull_period = current_date <= bull_end

        # Apply bull market bias
        if is_bull_period:
            confidence = min(base_confidence + 0.2, 1.0)  # Add 20% confidence during bull period
        else:
            confidence = base_confidence

        return {
            'is_bullish': confidence >= 0.5,
            'confidence': confidence,
            'is_bull_period': is_bull_period,
            'technical_indicators': {
                'rsi': float(features['rsi'].iloc[-1]),
                'macd': float(features['macd'].iloc[-1]),
                'volatility': float(features['volatility'].iloc[-1])
            }
        }
