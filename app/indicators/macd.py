import pandas as pd
import numpy as np

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD)
    
    Args:
        data (pd.DataFrame): DataFrame with 'close' price column
        fast_period (int): Fast EMA period
        slow_period (int): Slow EMA period
        signal_period (int): Signal line period
        
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    if isinstance(data, pd.DataFrame) and 'close' in data.columns:
        price = data['close']
    elif isinstance(data, pd.Series):
        price = data
    else:
        raise ValueError("Input must be DataFrame with 'close' column or Series")
    
    # Calculate fast and slow EMAs
    fast_ema = price.ewm(span=fast_period, adjust=False).mean()
    slow_ema = price.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram 