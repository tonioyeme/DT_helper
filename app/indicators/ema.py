import pandas as pd
import numpy as np

def calculate_ema(data, period=20):
    """
    Calculate Exponential Moving Average
    
    Args:
        data (pd.DataFrame): DataFrame with 'close' price column
        period (int): EMA period
        
    Returns:
        pd.Series: EMA values
    """
    if isinstance(data, pd.DataFrame) and 'close' in data.columns:
        return data['close'].ewm(span=period, adjust=False).mean()
    elif isinstance(data, pd.Series):
        return data.ewm(span=period, adjust=False).mean()
    else:
        raise ValueError("Input must be DataFrame with 'close' column or Series")

def calculate_ema_cloud(data, fast_period=20, slow_period=50):
    """
    Calculate EMA Cloud (fast and slow EMAs)
    
    Args:
        data (pd.DataFrame): DataFrame with 'close' price column
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        
    Returns:
        tuple: (Fast EMA Series, Slow EMA Series)
    """
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    
    return fast_ema, slow_ema 