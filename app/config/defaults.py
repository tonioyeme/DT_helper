"""
Default configuration settings for the Day Trade Helper application.
"""

from typing import Dict, Any

# Default configuration for the application
DEFAULT_CONFIG: Dict[str, Any] = {
    "general": {
        "default_symbol": "SPY",
        "default_exchange": "NASDAQ",
        "default_timeframe": "5m",
        "default_period": "1d",
        "theme": "light",
    },
    
    "indicators": {
        "ema": {
            "fast_period": 9,
            "slow_period": 21,
            "signal_period": 9,
            "weight": 1.0,
        },
        "rsi": {
            "period": 14,
            "overbought": 70,
            "oversold": 30,
            "weight": 1.0,
        },
        "macd": {
            "fast_period": 12,
            "slow_period": 26,
            "signal_period": 9,
            "weight": 1.0,
        },
        "vwap": {
            "reset_period": "day",
            "weight": 1.0,
        },
        "bollinger_bands": {
            "period": 20,
            "std_dev": 2.0,
            "weight": 1.0,
        },
    },
    
    "strategies": {
        "orb": {
            "enabled": True,
            "minutes": 5,
            "weight": 1.0,
        },
        "vwap_crossover": {
            "enabled": True,
            "weight": 1.0,
        },
        "ema_crossover": {
            "enabled": True,
            "weight": 1.0,
        },
    },
    
    "risk_management": {
        "max_risk_per_trade": 2.0,  # percentage
        "default_stop_loss": 1.0,   # percentage
        "default_take_profit": 2.0, # percentage
        "use_atr_for_stops": True,
        "atr_multiplier": 2.0,
    },
    
    "data": {
        "cache_data": True,
        "cache_expiry": 60,  # minutes
        "default_data_source": "yahoo",
    },
}

# Export the default configuration
__all__ = ["DEFAULT_CONFIG"] 