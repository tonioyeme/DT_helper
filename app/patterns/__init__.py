"""
Pattern detection modules for technical analysis.
"""

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