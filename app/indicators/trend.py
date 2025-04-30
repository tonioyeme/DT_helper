import pandas as pd
import numpy as np

def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Average Directional Index (ADX)
    
    The ADX is a trend strength indicator that uses the Directional Movement 
    Index (DMI) to determine if a security is trending. It ranges from 0 to 100,
    with higher values indicating stronger trends.
    
    Args:
        data: DataFrame with OHLCV data (must have 'high', 'low', 'close' columns)
        period: Period for ADX calculation (default: 14)
        
    Returns:
        pd.Series: ADX values
    """
    # Validate data
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")
        
    if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
        raise ValueError("Input DataFrame must have 'high', 'low', and 'close' columns")
    
    # Make a copy of the dataframe to avoid modifying original
    df = data.copy()
    
    # Calculate True Range (TR)
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Calculate Directional Movement (DM)
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    
    # Positive DM
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    pos_dm = pd.Series(pos_dm, index=df.index)
    
    # Negative DM
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    neg_dm = pd.Series(neg_dm, index=df.index)
    
    # Calculate smoothed TR, +DM, and -DM using Wilder's smoothing
    # First elements
    smoothed_tr = tr.rolling(window=period).sum()
    smoothed_pos_dm = pos_dm.rolling(window=period).sum()
    smoothed_neg_dm = neg_dm.rolling(window=period).sum()
    
    # Calculate +DI and -DI
    pos_di = 100 * (smoothed_pos_dm / smoothed_tr)
    neg_di = 100 * (smoothed_neg_dm / smoothed_tr)
    
    # Calculate DX
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
    
    # Calculate ADX
    adx = dx.rolling(window=period).mean()
    
    return adx

def calculate_directional_indicators(data: pd.DataFrame, period: int = 14) -> tuple:
    """
    Calculate ADX along with +DI and -DI directional indicators
    
    Args:
        data: DataFrame with OHLCV data (must have 'high', 'low', 'close' columns)
        period: Period for calculations (default: 14)
        
    Returns:
        tuple: (ADX, +DI, -DI)
    """
    # Validate data
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame")
        
    if 'high' not in data.columns or 'low' not in data.columns or 'close' not in data.columns:
        raise ValueError("Input DataFrame must have 'high', 'low', and 'close' columns")
    
    # Make a copy of the dataframe to avoid modifying original
    df = data.copy()
    
    # Calculate True Range (TR)
    high_low = df['high'] - df['low']
    high_prev_close = abs(df['high'] - df['close'].shift(1))
    low_prev_close = abs(df['low'] - df['close'].shift(1))
    
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    
    # Calculate Directional Movement (DM)
    up_move = df['high'] - df['high'].shift(1)
    down_move = df['low'].shift(1) - df['low']
    
    # Positive DM
    pos_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    pos_dm = pd.Series(pos_dm, index=df.index)
    
    # Negative DM
    neg_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    neg_dm = pd.Series(neg_dm, index=df.index)
    
    # Calculate smoothed TR, +DM, and -DM using Wilder's smoothing
    # First elements
    smoothed_tr = tr.rolling(window=period).sum()
    smoothed_pos_dm = pos_dm.rolling(window=period).sum()
    smoothed_neg_dm = neg_dm.rolling(window=period).sum()
    
    # Calculate +DI and -DI
    pos_di = 100 * (smoothed_pos_dm / smoothed_tr)
    neg_di = 100 * (smoothed_neg_dm / smoothed_tr)
    
    # Calculate DX
    dx = 100 * abs(pos_di - neg_di) / (pos_di + neg_di)
    
    # Calculate ADX
    adx = dx.rolling(window=period).mean()
    
    return adx, pos_di, neg_di 