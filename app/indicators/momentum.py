import pandas as pd
import numpy as np

def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI)
    
    RSI is a momentum oscillator that measures the speed and change of price movements
    on a scale from 0 to 100. RSI is considered overbought when above 70 and oversold
    when below 30.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        period (int): Lookback period for RSI calculation, default is 14
        
    Returns:
        pd.Series: RSI values
    """
    close = data['close']
    delta = close.diff()
    
    # Separate gains and losses
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # Calculate average gain and loss over the period
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    # Calculate RS (Relative Strength)
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_stochastic(data, k_period=14, d_period=3):
    """
    Calculate Stochastic Oscillator
    
    The Stochastic Oscillator is a momentum indicator that shows the location of
    the close relative to high-low range over a set number of periods.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        k_period (int): Period for %K calculation, default is 14
        d_period (int): Period for %D calculation, default is 3
        
    Returns:
        tuple: (stoch_k, stoch_d) - %K and %D values
    """
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Calculate %K: (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    low_min = low.rolling(window=k_period).min()
    high_max = high.rolling(window=k_period).max()
    
    stoch_k = 100 * ((close - low_min) / (high_max - low_min))
    stoch_k.fillna(0, inplace=True)  # Handle division by zero
    
    # Calculate %D: Simple moving average of %K
    stoch_d = stoch_k.rolling(window=d_period).mean()
    
    return stoch_k, stoch_d

def calculate_fibonacci_sma(data):
    """
    Calculate Fibonacci-based Simple Moving Averages (5-8-13)
    
    This calculates three SMAs based on Fibonacci numbers (5, 8, 13),
    which are often used in combination for day trading strategies.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        
    Returns:
        tuple: (sma5, sma8, sma13) - The three SMAs
    """
    close = data['close']
    
    sma5 = close.rolling(window=5).mean()
    sma8 = close.rolling(window=8).mean()
    sma13 = close.rolling(window=13).mean()
    
    return sma5, sma8, sma13

def calculate_pain(data):
    """
    Calculate Price Action Indicator (PAIN)
    
    This indicator calculates three values:
    - Intraday Momentum: Close - Open
    - Late Selling Pressure: Close - Low
    - Late Buying Pressure: Close - High
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        
    Returns:
        tuple: (intraday_momentum, late_selling, late_buying) - PAIN components
    """
    open_price = data['open']
    high = data['high']
    low = data['low']
    close = data['close']
    
    # Calculate the three components
    intraday_momentum = close - open_price
    late_selling = close - low
    late_buying = close - high  # This will be negative when close < high
    
    return intraday_momentum, late_selling, late_buying 