def calculate_enhanced_atr(data, period=14, smoothing='ema', scaling_factor=1.0):
    """
    Calculate an enhanced version of Average True Range (ATR) with additional parameters
    for more responsive exits and improved volatility measurement
    
    Args:
        data (pd.DataFrame): OHLCV data
        period (int): Period for ATR calculation
        smoothing (str): Smoothing method ('ema', 'wma', 'sma')
        scaling_factor (float): Factor to adjust ATR sensitivity
        
    Returns:
        pd.Series: Enhanced ATR values
    """
    import pandas as pd
    import numpy as np
    
    # Calculate true range
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Previous close for TR calculation
    prev_close = close.shift(1)
    
    # Calculate all three TR components
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    
    # Find the maximum of the three components
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    # Apply smoothing based on parameter
    if smoothing == 'ema':
        atr = true_range.ewm(span=period, adjust=False).mean()
    elif smoothing == 'wma':
        # Weighted moving average puts more weight on recent values
        weights = np.arange(1, period + 1)
        atr = true_range.rolling(window=period).apply(
            lambda x: np.sum(weights * x) / np.sum(weights), raw=True
        )
    else:  # Default to SMA
        atr = true_range.rolling(window=period).mean()
    
    # Apply scaling factor to make the ATR more or less sensitive
    return atr * scaling_factor 