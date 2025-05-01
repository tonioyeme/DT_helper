"""
Instrument-specific configurations for optimized trading.
"""

try:
    from .spy import SPY_CONFIG
except ImportError:
    import os
    print(f"Error importing SPY_CONFIG from {os.path.dirname(__file__)}/spy.py")
    # Create a placeholder so importers don't fail
    SPY_CONFIG = {}

__all__ = ["SPY_CONFIG"] 