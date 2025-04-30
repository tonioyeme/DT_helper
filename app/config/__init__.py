"""
Configuration module for the Day Trade Helper application.
Contains settings, parameters, and configurations for different instruments and strategies.
"""

# Import all configuration objects
try:
    from app.config.instruments.spy import SPY_CONFIG
except ImportError:
    # Define a default SPY config if the file doesn't exist
    SPY_CONFIG = {
        'indicators': {
            'ema': {'fast_period': 5, 'slow_period': 13},
            'rsi': {'period': 14, 'overbought': 70, 'oversold': 30},
            'vwap': {'reset_period': 'day'},
            'bollinger': {'period': 20, 'std_dev': 2.0}
        },
        'strategies': {
            'orb': {'minutes': 5, 'confirmation_candles': 1},
            'vwap_bollinger': {'enabled': True},
            'ema_vwap': {'enabled': True}
        },
        'signals': {
            'threshold': 0.6,
            'confirmation_required': False
        }
    }

from .defaults import DEFAULT_CONFIG

# Export config objects
__all__ = [
    'DEFAULT_CONFIG',
    'SPY_CONFIG'
] 