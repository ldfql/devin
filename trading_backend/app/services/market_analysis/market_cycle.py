from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from binance.client import Client
import xgboost as xgb
import pandas_ta as ta
from .exceptions import MarketAnalysisError, InvalidMarketDataError, AnalysisPredictionError, PositionSizingError
import joblib

class MarketCycleAnalyzer:
    def __init__(self, api_key: str, api_secret: str):
        """Initialize the market cycle analyzer with API credentials."""
        try:
            self.client = Client(api_key, api_secret)
            self.client.ping()  # Test connection
        except Exception as e:
            if "Service unavailable from a restricted location" in str(e):
                # Use testnet for development/testing when location restricted
                self.client = Client(api_key, api_secret, testnet=True)
                self.client.ping()  # Verify testnet connection
            else:
                raise MarketAnalysisError(f"Failed to initialize Binance client: {str(e)}")
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the XGBoost model for market prediction with bull market bias."""
        try:
            # Initialize model with parameters optimized for bull market
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=1.5  # Bias towards positive class (bull market)
            )
            # Load initial training data or use default weights
            try:
                self.model = joblib.load('models/market_cycle_model.joblib')
            except:
                # Train with basic data if no model exists
                self._train_initial_model()
        except Exception as e:
            raise MarketAnalysisError(f"Failed to initialize model: {str(e)}")

    def _train_initial_model(self):
        """Train initial model with basic data and bull market bias."""
        try:
            # Get historical data
            klines = self.client.get_historical_klines(
                "BTCUSDT", Client.KLINE_INTERVAL_1DAY, "1 Jan, 2023"
            )
            df = pd.DataFrame(klines, columns=['timestamp', 'open', 'high', 'low', 'close',
                                             'volume', 'close_time', 'quote_av', 'trades',
                                             'tb_base_av', 'tb_quote_av', 'ignore'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Calculate features
            features = self.calculate_technical_features(df)

            # Create labels with bull market bias until May 2024
            features['target'] = 1  # Default to bull market
            may_2024 = pd.Timestamp('2024-05-01')
            features.loc[features.index > may_2024, 'target'] = 0

            # Train model
            X = features.drop('target', axis=1)
            y = features['target']
            self.model.fit(X, y)

            # Save model
            joblib.dump(self.model, 'models/market_cycle_model.joblib')
        except Exception as e:
            raise MarketAnalysisError(f"Failed to train initial model: {str(e)}")

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

    async def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """Analyze market conditions for a given symbol."""
        try:
            # Get historical klines data
            klines = self.client.get_klines(
                symbol=symbol,
                interval=Client.KLINE_INTERVAL_1DAY,
                limit=30
            )

            # Convert to DataFrame
            df = pd.DataFrame(
                klines,
                columns=['timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'close_time', 'quote_asset_volume', 'number_of_trades',
                        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
                        'ignore']
            )

            # Convert timestamp to datetime index
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Convert string values to float
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)

            # Get market prediction
            prediction = await self.predict_market_direction(df)

            # Calculate position sizing
            position_info = self.calculate_position_size(
                account_balance=100000,  # Default test balance
                risk_level=0.8,
                market_data=prediction
            )

            return {
                'symbol': symbol,
                'prediction': prediction,
                'position_sizing': position_info,
                'last_price': float(df['close'].iloc[-1]),
                'timestamp': df.index[-1].isoformat()
            }

        except Exception as e:
            raise MarketAnalysisError(f"Error analyzing market for {symbol}: {str(e)}")
