import pandas as pd
import numpy as np
from typing import Union

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        data: DataFrame with price data (must have 'high', 'low', 'close' columns)
        period: Look-back period
        
    Returns:
        Series with ATR values
    """
    high = data['high']
    low = data['low']
    close = data['close'].shift(1)
    
    tr1 = high - low
    tr2 = (high - close).abs()
    tr3 = (low - close).abs()
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    
    return atr

def calculate_bollinger_bands(data: pd.DataFrame, 
                            period: int = 20, 
                            std_dev: float = 2.0,
                            source_column: str = 'close') -> tuple:
    """
    Calculate Bollinger Bands
    
    Args:
        data: DataFrame with price data
        period: Look-back period
        std_dev: Number of standard deviations
        source_column: Column name to use for calculations
        
    Returns:
        tuple: (upper_band, middle_band, lower_band) as pandas Series
    """
    source = data[source_column]
    
    # Calculate middle band (simple moving average)
    middle_band = source.rolling(window=period).mean()
    
    # Calculate standard deviation
    rolling_std = source.rolling(window=period).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (rolling_std * std_dev)
    lower_band = middle_band - (rolling_std * std_dev)
    
    return upper_band, middle_band, lower_band

def calculate_keltner_channels(data: pd.DataFrame,
                             period: int = 20,
                             atr_multiplier: float = 2.0,
                             atr_period: int = 14) -> tuple:
    """
    Calculate Keltner Channels
    
    Args:
        data: DataFrame with OHLC price data
        period: Look-back period for the EMA
        atr_multiplier: Multiplier for the ATR
        atr_period: Period for ATR calculation
        
    Returns:
        tuple: (upper_channel, middle_channel, lower_channel) as pandas Series
    """
    # Calculate the middle channel (EMA of typical price)
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    middle_channel = typical_price.ewm(span=period, adjust=False).mean()
    
    # Calculate ATR
    atr = calculate_atr(data, period=atr_period)
    
    # Calculate upper and lower channels
    upper_channel = middle_channel + (atr * atr_multiplier)
    lower_channel = middle_channel - (atr * atr_multiplier)
    
    return upper_channel, middle_channel, lower_channel 