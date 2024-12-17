from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from binance.client import Client
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketCycleAnalyzer:
    def __init__(self, api_key: str, api_secret: str):
        """Initialize the market cycle analyzer with Binance API credentials."""
        self.client = Client(api_key, api_secret)
        self.scaler = StandardScaler()
        self.model = xgb.XGBClassifier(
            objective='binary:logistic',
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.bull_end = pd.Timestamp('2024-05-01')
        self.confidence_threshold = 0.85
        self.min_position = 100  # Minimum position size in USDT
        self.max_position = 100_000_000  # Maximum position size in USDT

    async def get_market_data(self, symbol: str, interval: str = '1d', limit: int = 500) -> pd.DataFrame:
        """Fetch market data from Binance."""
        try:
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                limit=limit
            )
            df = pd.DataFrame(klines, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignored'
            ])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            raise

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for market analysis."""
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        df['rsi'] = self._calculate_rsi(df['close'])
        df['macd'], df['macd_signal'] = self._calculate_macd(df['close'])
        df['atr'] = self._calculate_atr(df)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std()
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD and Signal line."""
        exp1 = prices.ewm(span=12, adjust=False).mean()
        exp2 = prices.ewm(span=26, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=9, adjust=False).mean()
        return macd, signal

    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(period).mean()

    async def predict_market_direction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict market direction with bull market bias until May 2024."""
        current_date = df.index[-1]
        is_bull_period = current_date <= self.bull_end

        # Calculate prediction probability
        features = self.calculate_technical_features(df)
        features = features.dropna()

        # Add bull market bias
        if is_bull_period:
            prediction_confidence = self.model.predict_proba(
                self.scaler.transform(features.iloc[-1:])
            )[0][1] * 1.2  # 20% boost during bull market
        else:
            prediction_confidence = self.model.predict_proba(
                self.scaler.transform(features.iloc[-1:])
            )[0][1]

        # Determine market direction
        is_bullish = prediction_confidence >= self.confidence_threshold

        return {
            'is_bullish': is_bullish,
            'confidence': prediction_confidence,
            'is_bull_period': is_bull_period,
            'technical_indicators': {
                'rsi': features['rsi'].iloc[-1],
                'macd': features['macd'].iloc[-1],
                'volatility': features['volatility'].iloc[-1]
            }
        }

    def calculate_position_size(self,
                              account_balance: float,
                              risk_level: float,
                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate position size based on account balance and market conditions."""
        # Base position size on account growth
        base_position = min(max(account_balance * 0.1, self.min_position), self.max_position)

        # Adjust for market confidence
        confidence_multiplier = market_data['confidence'] if market_data['confidence'] >= self.confidence_threshold else 0.5

        # Adjust for volatility
        volatility = market_data['technical_indicators']['volatility']
        volatility_multiplier = 1 - (volatility * 2)  # Reduce position size in high volatility

        # Calculate final position size
        position_size = base_position * confidence_multiplier * volatility_multiplier * risk_level

        # Apply progressive scaling
        if account_balance > 10000:
            position_size *= 1.2  # 20% increase for larger accounts

        # Calculate stop loss and take profit
        stop_loss_pct = 0.02  # 2% stop loss
        take_profit_pct = 0.04  # 4% take profit

        return {
            'position_size': position_size,
            'stop_loss_pct': stop_loss_pct,
            'take_profit_pct': take_profit_pct,
            'reinvestment_amount': position_size * 0.7,  # 70% for reinvestment
            'withdrawal_amount': position_size * 0.3  # 30% for withdrawal
        }

    async def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """Perform complete market analysis."""
        try:
            # Fetch market data
            df = await self.get_market_data(symbol)

            # Predict market direction
            market_prediction = await self.predict_market_direction(df)

            # Calculate position sizing
            position_info = self.calculate_position_size(
                account_balance=10000,  # Example balance
                risk_level=0.8 if market_prediction['is_bull_period'] else 0.5,
                market_data=market_prediction
            )

            return {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'market_prediction': market_prediction,
                'position_info': position_info,
                'technical_data': {
                    'current_price': df['close'].iloc[-1],
                    'sma_20': df['close'].rolling(window=20).mean().iloc[-1],
                    'sma_50': df['close'].rolling(window=50).mean().iloc[-1]
                }
            }
        except Exception as e:
            logger.error(f"Error in market analysis: {e}")
            raise
