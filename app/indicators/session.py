import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time

def calculate_opening_range(data, minutes=5):
    """
    Calculate opening range for SPY based on first X minutes
    
    Args:
        data (pd.DataFrame): OHLCV data with datetime index
        minutes (int): Opening range duration in minutes
        
    Returns:
        tuple: (Opening Range High, Opening Range Low)
    """
    # Filter market open data (9:30-9:35 for 5-min ORB)
    market_open = data.between_time('09:30', '09:' + str(30 + minutes))
    if len(market_open) == 0:
        return None, None
        
    opening_high = market_open['high'].max()
    opening_low = market_open['low'].min()
    
    return opening_high, opening_low

def detect_orb_breakout(data, high, low, direction=None, buffer_percent=0.1):
    """
    Detect Opening Range Breakout signals
    
    Args:
        data (pd.DataFrame): OHLCV data with datetime index
        high (float): Opening range high
        low (float): Opening range low
        direction (str, optional): 'long', 'short', or None for both
        buffer_percent (float): Percentage buffer to avoid false breakouts
        
    Returns:
        pd.DataFrame: DataFrame with 'orb_long' and 'orb_short' columns
    """
    if high is None or low is None:
        return pd.DataFrame(index=data.index, columns=['orb_long', 'orb_short'])
    
    # Calculate buffer amounts
    high_buffer = high + (high * buffer_percent / 100)
    low_buffer = low - (low * buffer_percent / 100)
    
    # Initialize result DataFrame
    result = pd.DataFrame(index=data.index, columns=['orb_long', 'orb_short'])
    result['orb_long'] = False
    result['orb_short'] = False
    
    # Get data after opening range
    market_time = data.index.time
    for idx, row in data.iterrows():
        hour, minute = idx.hour, idx.minute
        
        # Skip opening range time
        if hour == 9 and minute <= 30 + minutes:
            continue
            
        # Check breakout conditions
        if direction in [None, 'long'] and row['high'] > high_buffer:
            result.loc[idx, 'orb_long'] = True
            
        if direction in [None, 'short'] and row['low'] < low_buffer:
            result.loc[idx, 'orb_short'] = True
    
    return result

def analyze_session_data(data, session_type='regular'):
    """
    Analyze market session data (regular, pre-market, after-hours)
    
    Args:
        data (pd.DataFrame): OHLCV data with datetime index
        session_type (str): 'regular', 'pre_market', or 'after_hours'
        
    Returns:
        pd.DataFrame: DataFrame with session analysis
    """
    # Convert index to Eastern time if it has timezone info
    if data.index.tzinfo is not None:
        eastern = pytz.timezone('US/Eastern')
        data = data.tz_convert(eastern)
    
    # Filter data based on session type
    if session_type == 'regular':
        session_data = data.between_time('09:30', '16:00')
    elif session_type == 'pre_market':
        session_data = data.between_time('04:00', '09:30')
    elif session_type == 'after_hours':
        session_data = data.between_time('16:00', '20:00')
    else:
        raise ValueError("session_type must be 'regular', 'pre_market', or 'after_hours'")
    
    # Calculate session metrics
    result = {}
    if len(session_data) > 0:
        result['open'] = session_data.iloc[0]['open']
        result['high'] = session_data['high'].max()
        result['low'] = session_data['low'].min()
        result['close'] = session_data.iloc[-1]['close']
        result['volume'] = session_data['volume'].sum()
        result['range'] = result['high'] - result['low']
        result['change'] = result['close'] - result['open']
        result['percent_change'] = (result['change'] / result['open']) * 100
    
    return result 