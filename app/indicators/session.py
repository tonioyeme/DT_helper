import pandas as pd
import numpy as np
import pytz
from datetime import datetime, time

def calculate_opening_range(data, minutes=5):
    """
    Calculate opening range for SPY based on first X minutes with proper session tracking
    
    Args:
        data (pd.DataFrame): OHLCV data with datetime index
        minutes (int): Opening range duration in minutes
        
    Returns:
        tuple: (Opening Range High Series, Opening Range Low Series)
    """
    # Convert to Eastern Time if needed
    eastern = pytz.timezone('US/Eastern')
    if data.index.tzinfo is not None:
        data = data.tz_convert(eastern)
    else:
        data = data.tz_localize('UTC').tz_convert(eastern)
    
    # Initialize storage for ORB levels
    orb_high = pd.Series(index=data.index, dtype='float64')
    orb_low = pd.Series(index=data.index, dtype='float64')
    
    # Track current trading day
    current_date = None
    current_high = -np.inf
    current_low = np.inf
    
    for idx, row in data.iterrows():
        # Reset on new trading day
        if current_date is None or idx.date() != current_date:
            current_date = idx.date()
            current_high = -np.inf
            current_low = np.inf
            session_start = idx.replace(hour=9, minute=30)
            session_end = session_start + pd.Timedelta(minutes=minutes)
        
        # Only update ORB during first X minutes
        if session_start <= idx <= session_end:
            current_high = max(current_high, row['high'])
            current_low = min(current_low, row['low'])
        
        orb_high[idx] = current_high
        orb_low[idx] = current_low
    
    # If no valid data found, return None values
    if current_high == -np.inf or current_low == np.inf:
        return None, None
        
    return orb_high, orb_low
    
def detect_orb_breakout(data, orb_high, orb_low, direction=None, buffer_percent=0.03):
    """
    Detect Opening Range Breakout signals with proper confirmation and session handling
    
    Args:
        data (pd.DataFrame): OHLCV data with datetime index
        orb_high (pd.Series): Opening range high for each timestamp
        orb_low (pd.Series): Opening range low for each timestamp
        direction (str, optional): 'long', 'short', or None for both
        buffer_percent (float): Percentage buffer to avoid false breakouts
        
    Returns:
        pd.DataFrame: DataFrame with 'orb_long' and 'orb_short' columns
    """
    # Handle None values
    if orb_high is None or orb_low is None:
        return pd.DataFrame(index=data.index, columns=['orb_long', 'orb_short'])
    
    # Ensure we have Series objects
    if not isinstance(orb_high, pd.Series):
        orb_high = pd.Series(orb_high, index=data.index)
    if not isinstance(orb_low, pd.Series):
        orb_low = pd.Series(orb_low, index=data.index)
    
    # Initialize signals DataFrame
    signals = pd.DataFrame(index=data.index)
    signals['orb_long'] = False
    signals['orb_short'] = False
    
    # Calculate buffers
    high_buffer = orb_high * (1 + buffer_percent)
    low_buffer = orb_low * (1 - buffer_percent)
    
    # Ensure data is in Eastern Time
    eastern = pytz.timezone('US/Eastern')
    if data.index.tzinfo is not None:
        data_et = data.tz_convert(eastern)
    else:
        data_et = data.tz_localize('UTC').tz_convert(eastern)
    
    # Track confirmed breakouts
    confirmed_long = False
    confirmed_short = False
    current_date = None
    
    # Skip first entry since we need to compare with previous
    for i in range(1, len(data)):
        idx = data.index[i]
        prev_idx = data.index[i-1]
        
        # Reset confirmations on new trading day
        if current_date is None or idx.date() != current_date:
            current_date = idx.date()
            confirmed_long = False
            confirmed_short = False
            # Skip ORB formation period (typically first 5-15 minutes)
            if idx.time() < time(9, 35):
                continue
        
        # Ignore signals outside market hours (9:30 AM - 4:00 PM ET)
        if not (time(9, 30) <= idx.time() <= time(16, 0)):
            continue
            
        # Get current and previous values
        prev_close = data.iloc[i-1]['close']
        curr_close = data.iloc[i]['close']
        
        # Check for long breakout (if not confirmed yet)
        if (not confirmed_long and (direction in [None, 'long'])):
            # Require consecutive closes above buffer
            long_condition = (
                (curr_close > high_buffer.iloc[i]) and 
                (prev_close > high_buffer.iloc[i-1])
            )
            if long_condition:
                signals.iloc[i, signals.columns.get_loc('orb_long')] = True
                confirmed_long = True
        
        # Check for short breakout (if not confirmed yet)
        if (not confirmed_short and (direction in [None, 'short'])):
            # Require consecutive closes below buffer
            short_condition = (
                (curr_close < low_buffer.iloc[i]) and 
                (prev_close < low_buffer.iloc[i-1])
            )
            if short_condition:
                signals.iloc[i, signals.columns.get_loc('orb_short')] = True
                confirmed_short = True
        
        # Reset confirmation at end of session
        if idx.time() >= time(16, 0):
            confirmed_long = False
            confirmed_short = False
    
    return signals

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