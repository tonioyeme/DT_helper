"""
Pattern detection modules for technical analysis.
"""

from app.patterns.candlestick import (
    identify_doji,
    identify_hammer,
    identify_engulfing,
    identify_morning_star,
    identify_evening_star
)

from app.patterns.adapter import (
    identify_head_and_shoulders,
    identify_inverse_head_and_shoulders,
    identify_double_top,
    identify_double_bottom,
    identify_triple_top,
    identify_triple_bottom,
    identify_rectangle,
    identify_channel,
    identify_triangle,
    identify_flag
)

__all__ = [
    # Candlestick patterns
    'identify_doji',
    'identify_hammer',
    'identify_engulfing',
    'identify_morning_star',
    'identify_evening_star',
    
    # Price action patterns
    'identify_head_and_shoulders',
    'identify_inverse_head_and_shoulders',
    'identify_double_top',
    'identify_double_bottom',
    'identify_triple_top',
    'identify_triple_bottom',
    'identify_rectangle',
    'identify_channel',
    'identify_triangle',
    'identify_flag'
] 