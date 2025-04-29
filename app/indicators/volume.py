import pandas as pd
import numpy as np

def calculate_obv(data):
    """
    Calculate On-Balance Volume (OBV)
    
    OBV is a cumulative total of volume based on price movement:
    - If today's close > yesterday's close, OBV = yesterday's OBV + today's volume
    - If today's close < yesterday's close, OBV = yesterday's OBV - today's volume
    - If today's close = yesterday's close, OBV = yesterday's OBV
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC and volume data
        
    Returns:
        pd.Series: On-Balance Volume values
    """
    close = data['close']
    volume = data['volume']
    
    # Initialize OBV with first period's volume
    obv = pd.Series(0, index=data.index)
    obv.iloc[0] = volume.iloc[0]
    
    # Calculate OBV for each period after the first
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_ad_line(data):
    """
    Calculate Accumulation/Distribution Line
    
    The A/D Line measures the flow of money into and out of a security.
    It uses the close price relative to the high-low range to determine
    buying or selling pressure.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC and volume data
        
    Returns:
        pd.Series: Accumulation/Distribution Line values
    """
    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']
    
    # Calculate Money Flow Multiplier
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low)
    # Handle division by zero
    money_flow_multiplier = money_flow_multiplier.replace([np.inf, -np.inf], 0)
    money_flow_multiplier.fillna(0, inplace=True)
    
    # Calculate Money Flow Volume
    money_flow_volume = money_flow_multiplier * volume
    
    # Calculate A/D Line (cumulative sum of Money Flow Volume)
    ad_line = money_flow_volume.cumsum()
    
    return ad_line 