# Import common types and functions
from .signal_functions import SignalStrength, generate_standard_signals, is_market_hours, calculate_adaptive_rsi
from .generator import generate_signals_multi_timeframe, generate_signals_advanced

# Define exported symbols
__all__ = [
    'SignalStrength',
    'generate_standard_signals',
    'is_market_hours',
    'calculate_adaptive_rsi',
    'generate_signals_multi_timeframe',
    'generate_signals_advanced'
]

# Make the signals directory a Python package 