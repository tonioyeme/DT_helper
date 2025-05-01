import numpy as np
import pandas as pd
from scipy import stats
from typing import Tuple, Optional

def calculate_roc(data: pd.DataFrame, period: int = 14, source_column: str = 'close') -> pd.Series:
    """
    Calculate Price Rate of Change (ROC)
    
    Args:
        data: DataFrame with price data
        period: Look-back period
        source_column: Column to use for calculation
        
    Returns:
        Series with ROC values
    """
    return ((data[source_column] - data[source_column].shift(period)) / 
            data[source_column].shift(period)) * 100

def calculate_hull_moving_average(data: pd.DataFrame, period: int = 20, source_column: str = 'close') -> pd.Series:
    """
    Calculate Hull Moving Average (HMA)
    
    Args:
        data: DataFrame with price data
        period: Look-back period
        source_column: Column to use for calculation
        
    Returns:
        Series with HMA values
    """
    def wma(series, window):
        weights = np.arange(1, window+1)
        return series.rolling(window).apply(
            lambda x: np.sum(weights * x) / weights.sum() if len(x) == window else np.nan, 
            raw=True
        )
    
    half_period = int(period / 2)
    sqrt_period = int(np.sqrt(period))
    
    wma_half = wma(data[source_column], half_period)
    wma_full = wma(data[source_column], period)
    
    raw_hma = 2 * wma_half - wma_full
    hma = wma(raw_hma, sqrt_period)
    
    return hma

def calculate_ttm_squeeze(data: pd.DataFrame, 
                         bb_period: int = 20, 
                         bb_std: float = 2.0, 
                         kc_period: int = 20, 
                         kc_mult: float = 1.5) -> pd.DataFrame:
    """
    Calculate TTM Squeeze Indicator (Bollinger Bands vs Keltner Channels)
    
    Args:
        data: DataFrame with price data
        bb_period: Bollinger Bands period
        bb_std: Number of standard deviations for Bollinger Bands
        kc_period: Keltner Channel period
        kc_mult: Keltner Channel multiplier
        
    Returns:
        DataFrame with TTM Squeeze columns (BB_Upper, BB_Lower, KC_Upper, KC_Lower, Squeeze_On)
    """
    result = pd.DataFrame(index=data.index)
    
    # Bollinger Bands
    basis = data['close'].rolling(bb_period).mean()
    dev = data['close'].rolling(bb_period).std()
    
    result['bb_upper'] = basis + bb_std * dev
    result['bb_lower'] = basis - bb_std * dev
    result['bb_mid'] = basis
    
    # Keltner Channel
    typical_price = (data['high'] + data['low'] + data['close']) / 3
    atr = calculate_atr(data, kc_period)
    
    result['kc_upper'] = basis + kc_mult * atr
    result['kc_lower'] = basis - kc_mult * atr
    result['kc_mid'] = basis
    
    # Squeeze Detection (True when BB is inside KC)
    result['squeeze_on'] = (result['bb_upper'] < result['kc_upper']) & (result['bb_lower'] > result['kc_lower'])
    
    # Momentum calculation (John Carter's method)
    mean_val = data['close'].rolling(bb_period).mean()
    highest = data['high'].rolling(bb_period).max()
    lowest = data['low'].rolling(bb_period).min()
    
    m = (data['close'] - ((highest + lowest) / 2 + mean_val) / 2)
    
    # Normalize the momentum
    result['squeeze_momentum'] = (m - m.rolling(bb_period).mean()) / m.rolling(bb_period).std()
    
    # Additional signals
    result['squeeze_exit'] = result['squeeze_on'].shift(1) & ~result['squeeze_on']
    result['squeeze_fire'] = result['squeeze_exit'] & (result['squeeze_momentum'] > 0)
    
    return result

def calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR)
    
    Args:
        data: DataFrame with price data
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

def detect_hidden_divergence(price: pd.Series, 
                           indicator: pd.Series, 
                           lookback: int = 14,
                           threshold: float = 0.02) -> Tuple[bool, bool]:
    """
    Detect hidden bullish and bearish divergences
    
    Hidden Bullish Divergence: Price making higher lows but indicator making lower lows
    Hidden Bearish Divergence: Price making lower highs but indicator making higher highs
    
    Args:
        price: Series of price data
        indicator: Series of indicator data (like RSI, MACD, etc)
        lookback: Number of periods to look back for divergence
        threshold: Minimum percentage change to consider a high/low
        
    Returns:
        Tuple of (hidden_bullish, hidden_bearish) booleans
    """
    if len(price) < lookback or len(indicator) < lookback:
        return False, False
    
    # Extract the segment we're analyzing
    price_segment = price.iloc[-lookback:]
    indicator_segment = indicator.iloc[-lookback:]
    
    # Find local minima and maxima in price
    price_peaks = find_peaks(price_segment, threshold)
    price_troughs = find_troughs(price_segment, threshold)
    
    # Find local minima and maxima in indicator
    indicator_peaks = find_peaks(indicator_segment, threshold=0)
    indicator_troughs = find_troughs(indicator_segment, threshold=0)
    
    # Need at least 2 peaks/troughs to detect divergence
    if len(price_peaks) < 2 or len(price_troughs) < 2 or len(indicator_peaks) < 2 or len(indicator_troughs) < 2:
        return False, False
    
    # Get the last two peaks and troughs
    last_price_peaks = sorted(price_peaks[-2:])
    last_price_troughs = sorted(price_troughs[-2:])
    last_indicator_peaks = sorted(indicator_peaks[-2:])
    last_indicator_troughs = sorted(indicator_troughs[-2:])
    
    # Check for hidden bullish divergence (price making higher lows, indicator making lower lows)
    hidden_bullish = False
    if (price_segment.iloc[last_price_troughs[-1]] > price_segment.iloc[last_price_troughs[-2]] and
        indicator_segment.iloc[last_indicator_troughs[-1]] < indicator_segment.iloc[last_indicator_troughs[-2]]):
        hidden_bullish = True
    
    # Check for hidden bearish divergence (price making lower highs, indicator making higher highs)
    hidden_bearish = False
    if (price_segment.iloc[last_price_peaks[-1]] < price_segment.iloc[last_price_peaks[-2]] and
        indicator_segment.iloc[last_indicator_peaks[-1]] > indicator_segment.iloc[last_indicator_peaks[-2]]):
        hidden_bearish = True
    
    return hidden_bullish, hidden_bearish

def find_peaks(data: pd.Series, threshold: float = 0.02) -> list:
    """
    Find peaks in a series
    
    Args:
        data: Series of data
        threshold: Minimum percentage change to consider a peak
        
    Returns:
        List of peak indices
    """
    peaks = []
    for i in range(1, len(data) - 1):
        if (data.iloc[i] > data.iloc[i-1] and data.iloc[i] > data.iloc[i+1] and
            (data.iloc[i] / data.iloc[i+1] - 1) > threshold):
            peaks.append(i)
    return peaks

def find_troughs(data: pd.Series, threshold: float = 0.02) -> list:
    """
    Find troughs in a series
    
    Args:
        data: Series of data
        threshold: Minimum percentage change to consider a trough
        
    Returns:
        List of trough indices
    """
    troughs = []
    for i in range(1, len(data) - 1):
        if (data.iloc[i] < data.iloc[i-1] and data.iloc[i] < data.iloc[i+1] and
            (data.iloc[i+1] / data.iloc[i] - 1) > threshold):
            troughs.append(i)
    return troughs

# Advanced Indicator Integration Utility Function
def add_advanced_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add all advanced indicators to a price DataFrame
    
    Args:
        data: DataFrame with OHLCV price data
        
    Returns:
        DataFrame with added indicators
    """
    if data is None or len(data) < 50:
        return None, None, None
    
    result = data.copy()
    
    # Add ROC
    result['roc_14'] = calculate_roc(data, period=14)
    
    # Add Hull Moving Average
    result['hma_20'] = calculate_hull_moving_average(data, period=20)
    result['hma_50'] = calculate_hull_moving_average(data, period=50)
    
    # Add TTM Squeeze
    ttm = calculate_ttm_squeeze(data)
    for col in ttm.columns:
        result[col] = ttm[col]
    
    # Add divergence calculations using RSI
    # First calculate RSI if not already present
    if 'rsi' not in result.columns:
        from app.indicators import calculate_rsi
        result['rsi'] = calculate_rsi(data)
    
    # Check for hidden divergences on the latest bar
    if len(result) > 20:
        hidden_bullish, hidden_bearish = detect_hidden_divergence(
            data['close'].iloc[-20:], 
            result['rsi'].iloc[-20:],
            lookback=20
        )
        
        # Add divergence signals
        result['hidden_bullish_div'] = False
        result['hidden_bearish_div'] = False
        
        if hidden_bullish:
            result.loc[result.index[-1], 'hidden_bullish_div'] = True
            
        if hidden_bearish:
            result.loc[result.index[-1], 'hidden_bearish_div'] = True
    
    return result 