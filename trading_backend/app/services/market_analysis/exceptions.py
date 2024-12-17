class MarketAnalysisError(Exception):
    """Base exception for market analysis errors."""
    pass

class InvalidMarketDataError(MarketAnalysisError):
    """Raised when market data is invalid or incomplete."""
    pass

class AnalysisPredictionError(MarketAnalysisError):
    """Raised when prediction fails."""
    pass

class PositionSizingError(MarketAnalysisError):
    """Raised when position sizing calculation fails."""
    pass
