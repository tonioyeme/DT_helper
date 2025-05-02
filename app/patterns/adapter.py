"""
Adapter module to convert price_action.py pattern detection to match the interface 
expected by the signal generator.
"""

import pandas as pd
from .price_action import (
    detect_head_and_shoulders,
    detect_inverted_head_and_shoulders,
    detect_double_top,
    detect_double_bottom,
    detect_triple_top,
    detect_triple_bottom,
    detect_rectangle,
    detect_channel,
    detect_triangle,
    detect_flag
)

def identify_head_and_shoulders(data, window=20):
    """Adapter for head and shoulders pattern"""
    return detect_head_and_shoulders(data, window)

def identify_inverse_head_and_shoulders(data, window=20):
    """Adapter for inverse head and shoulders pattern"""
    return detect_inverted_head_and_shoulders(data, window)

def identify_double_top(data, window=15):
    """Adapter for double top pattern"""
    return detect_double_top(data, window)

def identify_double_bottom(data, window=15):
    """Adapter for double bottom pattern"""
    return detect_double_bottom(data, window)

def identify_triple_top(data, window=20):
    """Adapter for triple top pattern"""
    return detect_triple_top(data, window)

def identify_triple_bottom(data, window=20):
    """Adapter for triple bottom pattern"""
    return detect_triple_bottom(data, window)

def identify_rectangle(data, window=15):
    """
    Adapter for rectangle pattern
    Returns bullish and bearish rectangles based on trend direction
    """
    rectangles = detect_rectangle(data, window)
    
    # Determine if bullish or bearish based on trend before rectangle
    bullish_rectangle = pd.Series(False, index=data.index)
    bearish_rectangle = pd.Series(False, index=data.index)
    
    for idx in data.index[rectangles]:
        # Look at prior trend (10 bars before)
        start_idx = max(0, data.index.get_loc(idx) - 15)
        end_idx = data.index.get_loc(idx)
        
        if start_idx >= end_idx:
            continue
            
        prior_trend = data.iloc[start_idx:end_idx]
        
        # If prior trend is up, this is a bullish continuation rectangle
        if prior_trend['close'].iloc[-1] > prior_trend['close'].iloc[0]:
            bullish_rectangle.loc[idx] = True
        else:
            bearish_rectangle.loc[idx] = True
    
    return bullish_rectangle, bearish_rectangle

def identify_channel(data, window=20):
    """
    Adapter for channel pattern
    Returns ascending and descending channels
    """
    channels = detect_channel(data, window)
    
    # Determine if ascending or descending based on slope
    ascending_channel = pd.Series(False, index=data.index)
    descending_channel = pd.Series(False, index=data.index)
    
    for idx in data.index[channels]:
        # Get section of data
        start_idx = max(0, data.index.get_loc(idx) - window)
        end_idx = data.index.get_loc(idx)
        
        if start_idx >= end_idx:
            continue
            
        section = data.iloc[start_idx:end_idx]
        
        # Calculate slope of prices
        prices = section['close']
        x = range(len(prices))
        
        try:
            slope, _ = pd.Series(prices).corr(pd.Series(x))
            
            if slope > 0:
                ascending_channel.loc[idx] = True
            else:
                descending_channel.loc[idx] = True
        except:
            continue
    
    return ascending_channel, descending_channel

def identify_triangle(data, window=20):
    """
    Adapter for triangle pattern
    Returns ascending, descending, and symmetric triangles
    """
    triangles = detect_triangle(data, window)
    
    # Determine triangle type based on price action
    ascending_triangle = pd.Series(False, index=data.index)
    descending_triangle = pd.Series(False, index=data.index)
    symmetric_triangle = pd.Series(False, index=data.index)
    
    for idx in data.index[triangles]:
        # Get section of data
        start_idx = max(0, data.index.get_loc(idx) - window)
        end_idx = data.index.get_loc(idx)
        
        if start_idx >= end_idx:
            continue
            
        section = data.iloc[start_idx:end_idx]
        
        # Analyze highs and lows to determine triangle type
        highs = [h for h in section['high'] if h > section['high'].mean()]
        lows = [l for l in section['low'] if l < section['low'].mean()]
        
        if len(highs) < 2 or len(lows) < 2:
            continue
            
        # Calculate slopes
        high_x = range(len(highs))
        low_x = range(len(lows))
        
        try:
            high_slope, _ = pd.Series(highs).corr(pd.Series(high_x))
            low_slope, _ = pd.Series(lows).corr(pd.Series(low_x))
            
            # Ascending triangle: flat top (high_slope near 0), rising bottom (low_slope > 0)
            if abs(high_slope) < 0.3 and low_slope > 0.3:
                ascending_triangle.loc[idx] = True
            
            # Descending triangle: flat bottom (low_slope near 0), falling top (high_slope < 0)
            elif abs(low_slope) < 0.3 and high_slope < -0.3:
                descending_triangle.loc[idx] = True
            
            # Symmetric triangle: converging slopes of opposite signs
            elif high_slope < -0.2 and low_slope > 0.2:
                symmetric_triangle.loc[idx] = True
        except:
            continue
    
    return ascending_triangle, descending_triangle, symmetric_triangle

def identify_flag(data, window=15):
    """
    Adapter for flag pattern
    Returns bull and bear flags
    """
    flags = detect_flag(data, window)
    
    # Determine if bull flag or bear flag based on prior trend
    bull_flag = pd.Series(False, index=data.index)
    bear_flag = pd.Series(False, index=data.index)
    
    for idx in data.index[flags]:
        # Get prior trend (looking at the pole before the flag)
        start_idx = max(0, data.index.get_loc(idx) - window)
        end_idx = data.index.get_loc(idx)
        
        if start_idx >= end_idx:
            continue
            
        prior_trend = data.iloc[start_idx:end_idx]
        
        # Bull flag: preceded by strong uptrend
        if prior_trend['close'].iloc[-1] > prior_trend['close'].iloc[0] * 1.02:  # 2% minimum rise
            bull_flag.loc[idx] = True
        
        # Bear flag: preceded by strong downtrend
        elif prior_trend['close'].iloc[-1] < prior_trend['close'].iloc[0] * 0.98:  # 2% minimum drop
            bear_flag.loc[idx] = True
    
    return bull_flag, bear_flag 