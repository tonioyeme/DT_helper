import pandas as pd
import numpy as np
from typing import Tuple, Union, Dict, Any, Optional
from .volatility import calculate_atr

def calculate_rsi(data, period=9):
    """
    Calculate Relative Strength Index (RSI)
    
    RSI is a momentum oscillator that measures the speed and change of price movements
    on a scale from 0 to 100. RSI is considered overbought when above 70 and oversold
    when below 30.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        period (int): Lookback period for RSI calculation, default is 9 (more sensitive for day trading)
        
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

def calculate_stochastic(data, k_period=8, d_period=3):
    """
    Calculate Stochastic Oscillator
    
    The Stochastic Oscillator is a momentum indicator that shows the location of
    the close relative to high-low range over a set number of periods.
    
    Args:
        data (pd.DataFrame): DataFrame with OHLC data
        k_period (int): Period for %K calculation, default is 8 (more sensitive for day trading) 
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

def calculate_adaptive_rsi(data: Union[pd.DataFrame, pd.Series], 
                          base_period: int = 14, 
                          min_period: int = 6, 
                          max_period: int = 24, 
                          lookback: int = 20,
                          atr_period: int = 14) -> pd.Series:
    """
    Calculate adaptive RSI where the period adjusts based on market volatility.
    
    In volatile markets (high ATR), a shorter period is used to be more responsive.
    In calm markets (low ATR), a longer period is used to reduce noise.
    
    Args:
        data: DataFrame with OHLC data or Series with price data
        base_period: The base RSI period (default period when volatility is average)
        min_period: Minimum RSI period during high volatility
        max_period: Maximum RSI period during low volatility
        lookback: Lookback period for ATR normalization
        atr_period: Period for ATR calculation
        
    Returns:
        Series containing the adaptive RSI values
    """
    # Ensure data is a DataFrame with required columns
    if isinstance(data, pd.Series):
        # If it's a Series, convert to DataFrame
        df = pd.DataFrame({"close": data})
    else:
        df = data.copy()
    
    # Calculate ATR for volatility measurement
    atr = calculate_atr(df, period=atr_period)
    
    # Normalize ATR over the lookback period to get relative volatility
    # We'll use a rolling window to compute the normalized ATR
    normalized_atr = pd.Series(index=atr.index, dtype=float)
    
    for i in range(len(atr)):
        if i >= lookback:
            # Get the ATR values for the lookback window
            window = atr.iloc[i-lookback:i+1]
            # Calculate min and max ATR in the window
            min_atr = window.min()
            max_atr = window.max()
            
            # Avoid division by zero
            if max_atr - min_atr > 0:
                # Normalize current ATR between 0 and 1
                norm_val = (atr.iloc[i] - min_atr) / (max_atr - min_atr)
            else:
                norm_val = 0.5  # Default to middle if range is zero
                
            normalized_atr.iloc[i] = norm_val
        else:
            # Not enough data for normalization
            normalized_atr.iloc[i] = 0.5  # Default to middle
    
    # Calculate adaptive periods: high volatility (norm_atr=1) → shorter period
    # low volatility (norm_atr=0) → longer period
    adaptive_periods = max_period - normalized_atr * (max_period - min_period)
    adaptive_periods = adaptive_periods.round().astype(int)
    
    # Calculate RSI with adaptive periods
    adaptive_rsi = pd.Series(index=df.index, dtype=float)
    adaptive_rsi_metadata = {}
    
    for i in range(len(df)):
        if i < max_period:
            # Not enough data for calculation with max period
            adaptive_rsi.iloc[i] = np.nan
            continue
            
        # Get the period for this specific point
        period = adaptive_periods.iloc[i]
        
        # Ensure we have enough data for this period
        if i >= period:
            # Calculate RSI for this point using the adaptive period
            window = df.iloc[i-period:i+1]
            
            # Get price data
            if 'close' in window.columns:
                price = window['close']
            else:
                price = window.iloc[:, 0]  # Use the first column
                
            # Calculate price changes
            delta = price.diff(1)
            
            # Calculate gains and losses
            gains = delta.where(delta > 0, 0)
            losses = -delta.where(delta < 0, 0)
            
            # Calculate average gains and losses
            avg_gain = gains.iloc[1:].mean()
            avg_loss = losses.iloc[1:].mean()
            
            if avg_loss == 0:
                # Avoid division by zero
                rsi_value = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi_value = 100 - (100 / (1 + rs))
                
            adaptive_rsi.iloc[i] = rsi_value
            
            # Store the period used for this point for reference
            adaptive_rsi_metadata[df.index[i]] = period
        else:
            adaptive_rsi.iloc[i] = np.nan
    
    # Add metadata to the series
    adaptive_rsi.name = "adaptive_rsi"
    adaptive_rsi.metadata = {
        "base_period": base_period,
        "min_period": min_period,
        "max_period": max_period,
        "periods_used": adaptive_rsi_metadata,
        "description": "Adaptive RSI that adjusts period based on market volatility"
    }
    
    return adaptive_rsi 

def calculate_stochastic_rsi(data: pd.DataFrame, 
                           rsi_period: int = 14, 
                           stoch_period: int = 14, 
                           k_period: int = 3, 
                           d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Stochastic RSI
    
    This is an indicator of an indicator - it applies the Stochastic oscillator
    formula to RSI values rather than price values, creating a more sensitive
    indicator that combines momentum and speed of movement.
    
    Args:
        data: DataFrame with OHLC data
        rsi_period: Period for RSI calculation
        stoch_period: Period for Stochastic calculation
        k_period: Smoothing period for %K
        d_period: Smoothing period for %D
        
    Returns:
        tuple: (stoch_rsi_k, stoch_rsi_d) - %K and %D values
    """
    # First calculate RSI
    rsi = calculate_rsi(data, period=rsi_period)
    
    # Then apply stochastic formula to RSI values
    # Lowest low and highest high of RSI values
    rsi_low = rsi.rolling(window=stoch_period).min()
    rsi_high = rsi.rolling(window=stoch_period).max()
    
    # Calculate %K
    stoch_rsi_k = 100 * ((rsi - rsi_low) / (rsi_high - rsi_low))
    
    # Handle division by zero
    stoch_rsi_k = stoch_rsi_k.replace([np.inf, -np.inf], np.nan).fillna(50)
    
    # Apply smoothing
    if k_period > 1:
        stoch_rsi_k = stoch_rsi_k.rolling(window=k_period).mean()
    
    # Calculate %D (SMA of %K)
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_period).mean()
    
    return stoch_rsi_k, stoch_rsi_d

def calculate_momentum_signals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate momentum-based trading signals
    
    Generates a set of signals based on various momentum indicators:
    - RSI: Overbought/Oversold conditions
    - Stochastic: Overbought/Oversold conditions and crossovers
    - StochasticRSI: Extreme readings and crossovers
    
    Args:
        data: DataFrame with OHLC price data
        
    Returns:
        DataFrame with added columns for momentum signals
    """
    if data is None or len(data) < 50:
        return data
        
    result = data.copy()
    
    # Calculate RSI
    result['rsi'] = calculate_rsi(data)
    
    # RSI signals
    result['rsi_overbought'] = result['rsi'] > 70
    result['rsi_oversold'] = result['rsi'] < 30
    result['rsi_bullish'] = (result['rsi'] > 50) & (result['rsi'].shift(1) <= 50)
    result['rsi_bearish'] = (result['rsi'] < 50) & (result['rsi'].shift(1) >= 50)
    
    # Calculate Stochastic
    stoch_k, stoch_d = calculate_stochastic(data)
    result['stoch_k'] = stoch_k
    result['stoch_d'] = stoch_d
    
    # Stochastic signals
    result['stoch_overbought'] = stoch_k > 80
    result['stoch_oversold'] = stoch_k < 20
    result['stoch_bullish_cross'] = (stoch_k > stoch_d) & (stoch_k.shift(1) <= stoch_d.shift(1))
    result['stoch_bearish_cross'] = (stoch_k < stoch_d) & (stoch_k.shift(1) >= stoch_d.shift(1))
    
    # Calculate Stochastic RSI
    stoch_rsi_k, stoch_rsi_d = calculate_stochastic_rsi(data)
    result['stoch_rsi_k'] = stoch_rsi_k
    result['stoch_rsi_d'] = stoch_rsi_d
    
    # Stochastic RSI signals
    result['stoch_rsi_overbought'] = stoch_rsi_k > 80
    result['stoch_rsi_oversold'] = stoch_rsi_k < 20
    result['stoch_rsi_bullish_cross'] = (stoch_rsi_k > stoch_rsi_d) & (stoch_rsi_k.shift(1) <= stoch_rsi_d.shift(1))
    result['stoch_rsi_bearish_cross'] = (stoch_rsi_k < stoch_rsi_d) & (stoch_rsi_k.shift(1) >= stoch_rsi_d.shift(1))
    
    # Momentum divergences
    if len(result) > 30:
        # Price series for comparing peaks and troughs
        price = result['close'].iloc[-30:]
        rsi_series = result['rsi'].iloc[-30:]
        
        # Check for divergences (simplified for demonstration)
        price_higher_high = price.iloc[-1] > price.iloc[-5:-2].max()
        price_lower_low = price.iloc[-1] < price.iloc[-5:-2].min()
        
        rsi_higher_high = rsi_series.iloc[-1] > rsi_series.iloc[-5:-2].max()
        rsi_lower_low = rsi_series.iloc[-1] < rsi_series.iloc[-5:-2].min()
        
        # Bearish divergence: price makes higher high but RSI makes lower high
        result.loc[result.index[-1], 'rsi_bearish_divergence'] = price_higher_high and not rsi_higher_high
        
        # Bullish divergence: price makes lower low but RSI makes higher low
        result.loc[result.index[-1], 'rsi_bullish_divergence'] = price_lower_low and not rsi_lower_low
    
    return result 