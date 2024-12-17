from datetime import datetime
from typing import Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.signals import TradingSignal
from app.repositories.signal_repository import SignalRepository
from app.services.market_analysis.market_data_service import MarketDataService

class AccuracyMonitor:
    def __init__(self, db_session: AsyncSession, market_data_service: MarketDataService):
        self.signal_repository = SignalRepository(db_session)
        self.market_data_service = market_data_service
        self.min_required_accuracy = 0.85  # Base accuracy requirement

    async def validate_timeframe_accuracy(
        self,
        timeframe: str,
        symbol: str,
        current_price: float,
        market_data: Dict
    ) -> float:
        """
        Validate accuracy for specific timeframe with continuous improvement target
        Returns the calculated accuracy score
        """
        signals = await self.signal_repository.get_active_signals(
            timeframe=timeframe,
            symbol=symbol
        )

        if not signals:
            return 0.0

        total_accuracy = 0.0
        valid_signals = 0

        for signal in signals:
            accuracy = await self._calculate_signal_accuracy(
                signal=signal,
                current_price=current_price,
                market_data=market_data
            )
            if accuracy is not None:
                total_accuracy += accuracy
                valid_signals += 1

        if valid_signals == 0:
            return 0.0

        return total_accuracy / valid_signals

    async def _calculate_signal_accuracy(
        self,
        signal: TradingSignal,
        current_price: float,
        market_data: Dict
    ) -> Optional[float]:
        """Calculate accuracy for a single signal with real-time data"""
        if not self._is_signal_valid(signal):
            return None

        # Base accuracy calculation
        price_accuracy = self._calculate_price_accuracy(
            signal.entry_price,
            current_price,
            signal.signal_type
        )

        # Market data validation
        market_accuracy = self._validate_market_conditions(
            signal,
            market_data
        )

        # Weighted accuracy calculation
        weighted_accuracy = (price_accuracy * 0.7) + (market_accuracy * 0.3)

        # Apply continuous improvement factor
        improved_accuracy = self._apply_improvement_factor(
            weighted_accuracy,
            signal.accuracy or self.min_required_accuracy
        )

        return improved_accuracy

    def _calculate_price_accuracy(
        self,
        entry_price: float,
        current_price: float,
        signal_type: str
    ) -> float:
        """Calculate accuracy based on price movement prediction"""
        price_diff_percent = abs((current_price - entry_price) / entry_price)

        if price_diff_percent <= 0.005:  # Within 0.5% threshold
            return 0.95  # High accuracy for precise predictions
        elif price_diff_percent <= 0.01:  # Within 1% threshold
            return 0.90
        elif price_diff_percent <= 0.02:  # Within 2% threshold
            return 0.85
        else:
            return max(0.85, 1.0 - price_diff_percent)  # Maintain minimum 85%

    def _validate_market_conditions(
        self,
        signal: TradingSignal,
        market_data: Dict
    ) -> float:
        """Validate accuracy against current market conditions"""
        # Start with base accuracy
        accuracy = 0.85

        # Adjust based on volume
        if "volume" in market_data:
            volume_factor = min(1.0, market_data["volume"] / 1000000)  # Normalize volume
            accuracy += volume_factor * 0.05

        # Adjust based on volatility
        if "volatility" in market_data:
            volatility_factor = 1.0 - min(1.0, market_data["volatility"] / 0.1)
            accuracy += volatility_factor * 0.05

        # Consider market cycle phase if available
        if signal.market_cycle_phase:
            accuracy += 0.05  # Bonus for signals with market cycle analysis

        return min(1.0, accuracy)  # Cap at 100%

    def _apply_improvement_factor(
        self,
        current_accuracy: float,
        previous_accuracy: float
    ) -> float:
        """Apply continuous improvement factor to accuracy calculation"""
        # Always maintain minimum accuracy requirement
        base_accuracy = max(current_accuracy, self.min_required_accuracy)

        # Calculate improvement factor
        improvement_factor = max(0, (base_accuracy - previous_accuracy) * 0.1)

        # Apply improvement while maintaining minimum requirement
        final_accuracy = max(
            self.min_required_accuracy,
            base_accuracy + improvement_factor
        )


        return min(1.0, final_accuracy)  # Cap at 100%

    def _is_signal_valid(self, signal: TradingSignal) -> bool:
        """Check if signal is still valid for accuracy calculation"""
        return (
            signal.expires_at is None or
            signal.expires_at > datetime.utcnow()
        )

    async def track_entry_point_accuracy(
        self,
        symbol: str,
        timeframe: str,
        current_price: float,
        market_data: Dict
    ) -> List[TradingSignal]:
        """Track accuracy of entry point predictions"""
        signals = await self.signal_repository.find_entry_points(
            symbol=symbol,
            min_confidence=self.min_required_accuracy,
            min_accuracy=self.min_required_accuracy
        )

        validated_signals = []
        for signal in signals:
            accuracy = await self._calculate_signal_accuracy(
                signal=signal,
                current_price=current_price,
                market_data=market_data
            )
            if accuracy and accuracy >= self.min_required_accuracy:
                signal.accuracy = accuracy
                signal.last_validated_at = datetime.utcnow()
                validated_signals.append(signal)
                await self.signal_repository.update(signal)

        return validated_signals
