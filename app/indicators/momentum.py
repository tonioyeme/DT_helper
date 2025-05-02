import pandas as pd
import numpy as np
from typing import Tuple, Union, Dict, Any, Optional
from .volatility import calculate_atr

def calculate_rsi(data: pd.DataFrame, period: int = 14, column: str = 'close') -> pd.Series:
    """
    Calculate the Relative Strength Index (RSI)
    
    Args:
        data: DataFrame with OHLCV data
        period: Look-back period for RSI calculation
        column: Column to use for calculation
        
    Returns:
        Series with RSI values
    """
    # Get price data
    prices = data[column]
    
    # Calculate price changes
    deltas = prices.diff()
    
    # Get gains and losses
    gain = deltas.clip(lower=0)
    loss = -deltas.clip(upper=0)
    
    # Calculate average gain and loss
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    
    # Handle division by zero
    avg_loss = avg_loss.replace(0, 0.001)
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
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

def calculate_adaptive_rsi(data: pd.DataFrame, period: int = 14, volatility_factor: float = 0.3) -> pd.Series:
    """
    Calculate an adaptive RSI that adjusts period based on volatility
    
    Args:
        data: DataFrame with OHLCV data
        period: Base period for RSI calculation
        volatility_factor: Factor to adjust period based on volatility
        
    Returns:
        Series with adaptive RSI values
    """
    # Calculate standard RSI
    base_rsi = calculate_rsi(data, period)
    
    # Calculate volatility (ATR as % of price)
    if 'atr' in data.columns:
        atr = data['atr']
    else:
        # Calculate a simple ATR if not available
        tr = pd.DataFrame()
        tr['h-l'] = data['high'] - data['low']
        tr['h-pc'] = abs(data['high'] - data['close'].shift(1))
        tr['l-pc'] = abs(data['low'] - data['close'].shift(1))
        tr['tr'] = tr.max(axis=1)
        atr = tr['tr'].rolling(period).mean()
    
    # Calculate volatility as % of price
    volatility = atr / data['close']
    
    # Calculate adaptive period
    avg_volatility = volatility.rolling(50).mean()
    period_adjustment = 1.0 + (volatility / avg_volatility - 1.0) * volatility_factor
    
    # Ensure we have enough data for the adaptive calculation
    min_period = 5
    max_period = 30
    
    # Adjust the RSI series based on the adaptive period
    # For simplicity, we'll just return the base RSI
    # In a full implementation, this would recalculate RSI with dynamic periods
    
    return base_rsi

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

def calculate_macd(data: pd.DataFrame, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9,
                  column: str = 'close') -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate the Moving Average Convergence Divergence (MACD)
    
    Args:
        data: DataFrame with OHLCV data
        fast_period: Period for fast EMA
        slow_period: Period for slow EMA
        signal_period: Period for signal line
        column: Column to use for calculation
        
    Returns:
        Tuple of (macd, signal, histogram)
    """
    # Get price data
    prices = data[column]
    
    # Calculate EMAs
    fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
    slow_ema = prices.ewm(span=slow_period, adjust=False).mean()
    
    # Calculate MACD line
    macd = fast_ema - slow_ema
    
    # Calculate signal line
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd - signal
    
    return macd, signal, histogram

def calculate_adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate the Average Directional Index (ADX)
    
    Args:
        data: DataFrame with OHLCV data
        period: Look-back period for ADX calculation
        
    Returns:
        Series with ADX values
    """
    # Calculate True Range
    tr = pd.DataFrame()
    tr['h-l'] = data['high'] - data['low']
    tr['h-pc'] = abs(data['high'] - data['close'].shift(1))
    tr['l-pc'] = abs(data['low'] - data['close'].shift(1))
    tr['tr'] = tr.max(axis=1)
    
    # Calculate Directional Movement
    data['up_move'] = data['high'] - data['high'].shift(1)
    data['down_move'] = data['low'].shift(1) - data['low']
    
    # Calculate +DM and -DM
    data['plus_dm'] = ((data['up_move'] > data['down_move']) & (data['up_move'] > 0)) * data['up_move']
    data['minus_dm'] = ((data['down_move'] > data['up_move']) & (data['down_move'] > 0)) * data['down_move']
    
    # Calculate smoothed values
    smoothed_tr = tr['tr'].rolling(period).sum()
    smoothed_plus_dm = data['plus_dm'].rolling(period).sum()
    smoothed_minus_dm = data['minus_dm'].rolling(period).sum()
    
    # Calculate Directional Indicators
    data['plus_di'] = 100 * smoothed_plus_dm / smoothed_tr
    data['minus_di'] = 100 * smoothed_minus_dm / smoothed_tr
    
    # Calculate Directional Index
    data['dx'] = 100 * abs(data['plus_di'] - data['minus_di']) / (data['plus_di'] + data['minus_di'])
    
    # Calculate ADX
    adx = data['dx'].rolling(period).mean()
    
    return adx

def calculate_trend_strength(data: pd.DataFrame) -> pd.Series:
    """
    Calculate overall trend strength as a normalized value between 0 and 1
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        Series with trend strength values
    """
    try:
        # Calculate ADX if not already present
        if 'adx' not in data.columns:
            adx = calculate_adx(data)
        else:
            adx = data['adx']
        
        # Calculate EMAs if not already present
        if 'ema20' not in data.columns:
            data['ema20'] = data['close'].ewm(span=20, adjust=False).mean()
        if 'ema50' not in data.columns:
            data['ema50'] = data['close'].ewm(span=50, adjust=False).mean()
        
        # Calculate EMA difference as % of price
        ema_diff = (data['ema20'] - data['ema50']) / data['close'] * 100
        
        # Normalize ADX (0-100 scale, but typical range is 0-50)
        normalized_adx = adx / 50.0
        normalized_adx = normalized_adx.clip(0, 1)
        
        # Normalize EMA difference (typically -5% to +5%)
        normalized_ema_diff = abs(ema_diff) / 5.0
        normalized_ema_diff = normalized_ema_diff.clip(0, 1)
        
        # Combine indicators (70% ADX, 30% EMA difference)
        trend_strength = normalized_adx * 0.7 + normalized_ema_diff * 0.3
        
        return trend_strength
    
    except Exception as e:
        print(f"Error calculating trend strength: {str(e)}")
        return pd.Series(0.5, index=data.index)  # Return moderate strength on error 