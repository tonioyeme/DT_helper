import pandas as pd
import numpy as np

def calculate_ema(data, period=9):
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        data: DataFrame with 'close' column or Series
        period: EMA period
    
    Returns:
        Series: EMA values
    """
    if isinstance(data, pd.DataFrame) and 'close' in data.columns:
        return data['close'].ewm(span=period, adjust=False).mean()
    elif isinstance(data, pd.Series):
        return data.ewm(span=period, adjust=False).mean()
    else:
        raise ValueError("data must be a DataFrame with 'close' column or a Series")

def calculate_ema_cloud(data, fast_period=5, slow_period=13):
    """
    Calculate EMA cloud (fast and slow EMAs)
    
    Args:
        data: DataFrame with 'close' column or Series
        fast_period: Fast EMA period (default: 5 for better sensitivity on 5-minute charts)
        slow_period: Slow EMA period (default: 13 for better sensitivity on 5-minute charts)
    
    Returns:
        tuple: (fast_ema, slow_ema) - tuple of two Series
    """
    fast_ema = calculate_ema(data, fast_period)
    slow_ema = calculate_ema(data, slow_period)
    return fast_ema, slow_ema 