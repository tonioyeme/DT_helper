import pandas as pd
import numpy as np

def calculate_vwap(data, reset_period='D'):
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Args:
        data (pd.DataFrame): DataFrame with 'high', 'low', 'close', and 'volume' columns
        reset_period (str): Period to reset VWAP calculation ('D' for daily)
        
    Returns:
        pd.Series: VWAP values
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input must be a DataFrame")
    
    required_cols = ['high', 'low', 'close', 'volume']
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain these columns: {required_cols}")
    
    # Ensure data has a datetime index for period resets
    if not isinstance(data.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be DatetimeIndex")
    
    # Calculate typical price
    data = data.copy()
    data['typical_price'] = (data['high'] + data['low'] + data['close']) / 3
    
    # Calculate price * volume
    data['pv'] = data['typical_price'] * data['volume']
    
    # Group by the reset period and calculate cumulative values
    if reset_period:
        data['cumulative_pv'] = data.groupby(pd.Grouper(freq=reset_period))['pv'].cumsum()
        data['cumulative_volume'] = data.groupby(pd.Grouper(freq=reset_period))['volume'].cumsum()
    else:
        data['cumulative_pv'] = data['pv'].cumsum()
        data['cumulative_volume'] = data['volume'].cumsum()
    
    # Calculate VWAP
    vwap = data['cumulative_pv'] / data['cumulative_volume']
    
    return vwap 