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
        """Initialize the XGBoost model for market prediction without bias."""
        try:
            self.model = xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=100,
                max_depth=4,
                learning_rate=0.1,
                scale_pos_weight=1.0  # Remove bull market bias
            )
            try:
                self.model = joblib.load('models/market_cycle_model.joblib')
            except:
                self._train_initial_model()
        except Exception as e:
            raise MarketAnalysisError(f"Failed to initialize model: {str(e)}")

    def _train_initial_model(self):
        """Train initial model with historical data."""
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

            # Create labels based on technical indicators
            features['target'] = 0  # Default to neutral
            features.loc[
                (features['rsi'] > 55) &
                (features['macd'] > 0) &
                (features['volatility'] < 0.03),
                'target'
            ] = 1  # Bullish conditions

            # Train model
            X = features.drop('target', axis=1)
            y = features['target']
            self.model.fit(X, y)

            # Save model
            joblib.dump(self.model, 'models/market_cycle_model.joblib')
        except Exception as e:
            raise MarketAnalysisError(f"Failed to train initial model: {str(e)}")

    def calculate_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators for market analysis with enhanced volatility metrics."""
        features = pd.DataFrame(index=df.index)

        # Calculate RSI with standard and longer periods
        features['rsi'] = ta.rsi(df['close'], length=14)
        features['rsi_long'] = ta.rsi(df['close'], length=28)  # Longer period for trend confirmation

        # Calculate MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = macd - signal

        # Enhanced volatility metrics
        atr = ta.atr(df['high'], df['low'], df['close'], length=14)
        features['volatility'] = atr / df['close']

        # Add Bollinger Bands volatility
        bb = ta.bbands(df['close'], length=20)
        features['bb_width'] = (bb['BBU_20_2.0'] - bb['BBL_20_2.0']) / bb['BBM_20_2.0']

        # Fill NaN values
        features = features.fillna(method='ffill').fillna(method='bfill')

        return features

    def calculate_position_size(
        self,
        account_balance: float,
        risk_level: float,
        market_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Calculate position size based on market conditions and risk management."""
        # Base position size on account balance and risk level
        base_position = account_balance * risk_level

        # Get market conditions
        confidence_factor = market_data.get('confidence', 0.5)
        volatility = market_data.get('technical_indicators', {}).get('volatility', 0.02)
        is_uncertain = market_data.get('is_uncertain', False)
        is_bullish = market_data.get('is_bullish', False)

        # Adjust risk based on market conditions
        if is_uncertain:
            # Reduce position size significantly during uncertainty
            confidence_factor *= 0.3
        elif not is_bullish:
            # Reduce position size in bear markets
            confidence_factor *= 0.5

        # Calculate adjusted position size
        adjusted_position = base_position * confidence_factor * (1 / volatility)

        # Enforce position size limits (100 USDT to 100M USDT)
        position_size = max(100, min(adjusted_position, 100_000_000))

        return {
            'position_size': position_size,
            'reinvestment_amount': position_size * 0.7,
            'withdrawal_amount': position_size * 0.3
        }

    async def predict_market_direction(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Predict market direction with enhanced market phase detection."""
        # Calculate technical features
        features = self.calculate_technical_features(df)

        # Get prediction probability
        pred_proba = self.model.predict_proba(features.iloc[[-1]])
        base_confidence = pred_proba[0][1]  # Probability of bullish prediction

        # Get latest technical indicators
        rsi = features['rsi'].iloc[-1]
        rsi_long = features['rsi_long'].iloc[-1]
        macd = features['macd'].iloc[-1]
        volatility = features['volatility'].iloc[-1]
        bb_width = features['bb_width'].iloc[-1]

        # Enhanced market phase detection with adjusted thresholds
        is_bull = (
            base_confidence >= 0.8 and  # High bullish probability
            rsi > 50 and  # Less strict RSI threshold
            rsi_long > 50 and
            macd > 0 and
            volatility < 0.04
        )

        # Enhanced bear market detection with adjusted thresholds
        is_bear = (
            base_confidence <= 0.2 or  # Very low bullish probability
            (
                base_confidence < 0.3 and  # Low bullish probability
                (rsi < 45 or rsi_long < 45) and  # Either RSI below threshold
                (macd < 0 or volatility > 0.03)  # Either bearish MACD or high volatility
            )
        )

        # Enhanced uncertainty detection
        is_uncertain = (
            not is_bull and not is_bear and  # Not clearly bull or bear
            (45 <= rsi <= 55) and  # RSI in middle range
            abs(macd) < 0.0001 and  # MACD near zero
            volatility > 0.02  # Some volatility present
        )

        # Adjust confidence based on market conditions
        if is_uncertain:
            confidence = 0.5
        elif is_bear:
            confidence = max(0.1, base_confidence - 0.2)  # Ensure minimum confidence
        elif is_bull:
            confidence = min(1.0, base_confidence + 0.1)  # Cap at maximum confidence
        else:
            confidence = base_confidence

        return {
            'is_bullish': is_bull or (not is_bear and not is_uncertain and confidence >= 0.85),
            'is_bearish': is_bear,
            'is_uncertain': is_uncertain,
            'confidence': confidence,
            'technical_indicators': {
                'rsi': float(rsi),
                'rsi_long': float(rsi_long),
                'macd': float(macd),
                'volatility': float(volatility),
                'bb_width': float(bb_width)
            }
        }

    async def analyze_market(self, symbol: str) -> Dict[str, Any]:
        """Analyze market conditions and determine optimal position sizing."""
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

            # Validate price data
            if not self._validate_price_data(df):
                raise InvalidMarketDataError("Invalid price data detected")

            # Get market prediction with enhanced features
            prediction = await self.predict_market_direction(df)

            # Determine base risk level based on market conditions
            base_risk = self._calculate_base_risk(prediction)

            # Calculate position sizing with dynamic risk adjustment
            position_info = self.calculate_position_size(
                account_balance=100_000,  # Default balance
                risk_level=base_risk,
                market_data=prediction
            )

            return {
                'symbol': symbol,
                'prediction': prediction,
                'position_sizing': position_info,
                'last_price': float(df['close'].iloc[-1]),
                'timestamp': df.index[-1].isoformat(),
                'market_conditions': {
                    'trend': 'bearish' if prediction.get('is_bearish') else 'bullish',
                    'risk_level': base_risk,
                    'volatility': prediction['technical_indicators']['volatility']
                }
            }

        except Exception as e:
            raise MarketAnalysisError(f"Error analyzing market for {symbol}: {str(e)}")

    def _validate_price_data(self, df: pd.DataFrame) -> bool:
        """Validate price data for accuracy and consistency."""
        try:
            # Check for extreme price movements
            price_change = abs(df['close'].pct_change())
            if (price_change > 0.5).any():  # 50% price change threshold
                return False

            # Check for zero or negative prices
            if (df[['open', 'high', 'low', 'close']] <= 0).any().any():
                return False

            # Check for high-low relationship
            if not ((df['high'] >= df['low']) & (df['high'] >= df['open']) & (df['high'] >= df['close'])).all():
                return False

            return True
        except Exception:
            return False

    def _calculate_base_risk(self, prediction: Dict[str, Any]) -> float:
        """Calculate base risk level based on market conditions."""
        base_risk = 0.8  # Default risk level

        # Adjust risk based on market conditions
        if prediction.get('is_uncertain', False):
            base_risk *= 0.3  # Significant reduction in uncertain markets
        elif prediction.get('is_bearish', False):
            base_risk *= 0.5  # Moderate reduction in bear markets
        elif not prediction.get('is_bullish', False):
            base_risk *= 0.7  # Slight reduction in neutral markets

        # Further adjust based on volatility
        volatility = prediction['technical_indicators']['volatility']
        if volatility > 0.05:  # High volatility
            base_risk *= 0.8

        return max(0.1, min(base_risk, 0.9))  # Keep risk between 10% and 90%
